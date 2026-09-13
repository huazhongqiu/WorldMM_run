"""OpenAI-compatible LMDeploy provider used for local Qwen3.5 inference."""

from __future__ import annotations

import base64
import io
import logging
import os
from typing import Any

from openai import OpenAI
from pydantic import ValidationError

# Preprocessing runs thousands of requests per phase; httpx logs every one at
# INFO and floods the run logs, so keep transport noise out of the handlers.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


class LMDeployConfigurationError(ValueError):
    """Raised when the local LMDeploy endpoint is not configured."""


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
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG")
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
        if text.startswith("```"):
            text = text.strip("`").lstrip()
            if "\n" in text:
                text = text.split("\n", 1)[1]
        start = min((i for i in (text.find("{"), text.find("[")) if i != -1), default=-1)
        if start > 0:
            end = max(text.rfind("}"), text.rfind("]"))
            if end > start:
                text = text[start : end + 1]
        return text

    def generate(self, prompt: Any, text_format: Any | None = None, **kwargs: Any) -> Any:
        options = {**self.request_options, **kwargs}
        response = self.sync_client.chat.completions.create(
            model=self.model_name,
            messages=self._messages(prompt),
            **options,
        )
        self._add_usage(response)
        content = response.choices[0].message.content or ""
        if text_format is None:
            return content
        try:
            return text_format.model_validate_json(content)
        except ValidationError:
            return text_format.model_validate_json(self._json_payload(content))
