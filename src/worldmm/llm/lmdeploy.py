"""OpenAI-compatible LMDeploy provider used for local Qwen3.5 inference."""

from __future__ import annotations

import ast
import base64
import io
import json
import logging
import os
from typing import Any

from openai import OpenAI
from pydantic import ValidationError

logger = logging.getLogger(__name__)

# Preprocessing runs thousands of requests per phase; httpx logs every one at
# INFO and floods the run logs, so keep transport noise out of the handlers.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

IMAGE_MAX_SIZE = (512, 512)


class LMDeployConfigurationError(ValueError):
    """Raised when the local LMDeploy endpoint is not configured."""


class StructuredOutputTruncatedError(ValueError):
    """The output cap was reached; retrying the same input cannot ensure complete JSON."""


class LMDeployModel:
    """Small adapter from WorldMM prompts to LMDeploy chat completions."""

    def __init__(
        self,
        model_name: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        client: Any | None = None,
        **request_options: Any,
    ) -> None:
        # Local video backends accept fps as a model-construction hint.  It is
        # not an OpenAI chat-completions parameter and LMDeploy rejects it.
        request_options.pop("fps", None)
        self.model_name = os.environ.get("WORLDMM_LMDEPLOY_MODEL", model_name)
        self.base_url = base_url or os.environ.get("WORLDMM_LMDEPLOY_BASE_URL")
        if not self.base_url:
            raise LMDeployConfigurationError(
                "Set WORLDMM_LMDEPLOY_BASE_URL, e.g. http://127.0.0.1:23333/v1"
            )
        self.api_key = api_key or os.environ.get("WORLDMM_LMDEPLOY_API_KEY", "EMPTY")
        self.request_options = request_options
        self.request_options.setdefault(
            "extra_body",
            {
                "enable_thinking": os.environ.get(
                    "WORLDMM_LMDEPLOY_ENABLE_THINKING", "false"
                ).strip().lower() == "true"
            },
        )
        if "max_tokens" not in self.request_options and os.environ.get("WORLDMM_LMDEPLOY_MAX_TOKENS"):
            self.request_options["max_tokens"] = int(os.environ["WORLDMM_LMDEPLOY_MAX_TOKENS"])
        self.sync_client = client or OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.kwargs = request_options
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    @staticmethod
    def _image_data_url(image: Any) -> str:
        if isinstance(image, str):
            return image
        from PIL import Image

        # Same thumbnail bound as the upstream WorldMM Qwen3-VL/GPT clients (512x512, aspect kept),
        # i.e. VideoSpy max_image_pixels=262144; native frames overflow the 262k context.
        image = image.convert("RGB")
        image.thumbnail(IMAGE_MAX_SIZE, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

    def _messages(self, prompt: Any) -> list[dict[str, Any]]:
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]

        result: list[dict[str, Any]] = []
        for message in prompt:
            content = message["content"]
            if not isinstance(content, list):
                result.append({"role": message["role"], "content": content})
                continue
            converted: list[dict[str, Any]] = []
            for item in content:
                if item["type"] == "text":
                    converted.append({"type": "text", "text": item["text"]})
                elif item["type"] == "image":
                    converted.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": self._image_data_url(item["image"])},
                        }
                    )
                else:
                    raise ValueError(f"Unsupported LMDeploy content type: {item['type']}")
            result.append({"role": message["role"], "content": converted})
        return result

    def _add_usage(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        for name in self.usage:
            value = getattr(usage, name, None)
            if value is not None:
                self.usage[name] += int(value)

    def usage_snapshot(self) -> dict[str, int]:
        return dict(self.usage)

    @staticmethod
    def _json_payload(content: str) -> str:
        """Best-effort extraction of the JSON document from a model reply.

        Qwen models sometimes wrap JSON in markdown fences or add prose around
        it; ``model_validate_json`` rejects both, so strip them before parsing.
        """
        text = content.strip()
        start = min((i for i in (text.find("{"), text.find("[")) if i != -1), default=-1)
        if start >= 0:
            candidate = text[start:]
            try:
                _, end = json.JSONDecoder().raw_decode(candidate)
                return candidate[:end]
            except json.JSONDecodeError:
                # Preserve the original invalid document for validation/limited literal repair.
                return candidate.rstrip("`\n ")
        return text

    @staticmethod
    def _repair_json_text(text: str) -> str | None:
        """Last-resort repair for near-JSON replies that json.loads rejects.

        Small models occasionally emit Python-style literals (single-quoted
        strings, True/False/None, trailing commas) inside an otherwise
        well-formed object; ``ast.literal_eval`` parses those exactly.
        """
        try:
            obj = ast.literal_eval(text)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            return None
        if isinstance(obj, (dict, list)):
            try:
                return json.dumps(obj, ensure_ascii=False)
            except (TypeError, ValueError):
                return None
        return None

    def generate(self, prompt: Any, text_format: Any | None = None, **kwargs: Any) -> Any:
        options = {**self.request_options, **kwargs}
        messages = self._messages(prompt)
        if text_format is not None and text_format.__name__ in ("NerRawOutput", "TripleRawOutput"):
            schema = text_format.model_json_schema()
            schema["additionalProperties"] = False
            if text_format.__name__ == "TripleRawOutput":
                schema["properties"]["triples"]["items"].update(minItems=3, maxItems=3)
                schema["properties"]["triples"]["items"]["items"]["minLength"] = 1
            options["response_format"] = {"type": "json_schema", "json_schema": {
                "name": text_format.__name__, "strict": True, "schema": schema}}
            options["temperature"] = 0
            messages = [dict(message) for message in messages]
            instruction = "Return only compact JSON on one line without Markdown or explanation."
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] += "\n" + instruction
            else:
                messages.insert(0, {"role": "system", "content": instruction})
        response = self.sync_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **options,
        )
        self._add_usage(response)
        content = response.choices[0].message.content or ""
        if text_format is None:
            return content
        if getattr(response.choices[0], "finish_reason", None) == "length":
            raise StructuredOutputTruncatedError(
                f"Structured output truncated at max_tokens={options.get('max_tokens')}; reduce the input length")
        try:
            return text_format.model_validate_json(content)
        except ValidationError:
            payload = self._json_payload(content)
            try:
                return text_format.model_validate_json(payload)
            except ValidationError:
                repaired = self._repair_json_text(payload)
                if repaired is not None:
                    try:
                        return text_format.model_validate_json(repaired)
                    except ValidationError:
                        pass
                logger.warning(
                    "Unparseable structured output for %s (first 500 chars): %r",
                    getattr(text_format, "__name__", text_format),
                    content[:500],
                )
                raise
