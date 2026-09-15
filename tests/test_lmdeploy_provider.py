import importlib.util

import sys

from pathlib import Path

from types import SimpleNamespace





SOURCE = Path(__file__).resolve().parents[1] / "src" / "worldmm" / "llm" / "llm_wrapper.py"

SPEC = importlib.util.spec_from_file_location("worldmm_llm_wrapper", SOURCE)

assert SPEC is not None

assert SPEC.loader is not None


LMDEPLOY_SOURCE = Path(__file__).resolve().parents[1] / "src" / "worldmm" / "llm" / "lmdeploy.py"
LMDEPLOY_SPEC = importlib.util.spec_from_file_location("worldmm_lmdeploy", LMDEPLOY_SOURCE)
assert LMDEPLOY_SPEC is not None
assert LMDEPLOY_SPEC.loader is not None
LMDEPLOY_MODULE = importlib.util.module_from_spec(LMDEPLOY_SPEC)
sys.modules[LMDEPLOY_SPEC.name] = LMDEPLOY_MODULE
LMDEPLOY_SPEC.loader.exec_module(LMDEPLOY_MODULE)


class FakeClient:
    def __init__(self) -> None:
        self.requests = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ready"))],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7, total_tokens=18),
        )


def test_lmdeploy_uses_chat_completions_and_accumulates_usage() -> None:
    client = FakeClient()
    model = LMDEPLOY_MODULE.LMDeployModel(
        model_name="Qwen3.5-4B",
        base_url="http://127.0.0.1:23333/v1",
        client=client,
    )

    assert model.generate("hello") == "ready"
    assert client.requests == [
        {
            "model": "Qwen3.5-4B",
            "messages": [{"role": "user", "content": "hello"}],
            "extra_body": {"enable_thinking": False},
        }
    ]
    assert model.usage == {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
    }

def test_lmdeploy_reads_configured_token_cap(monkeypatch) -> None:
    monkeypatch.setenv("WORLDMM_LMDEPLOY_MAX_TOKENS", "4096")
    client = FakeClient()
    model = LMDEPLOY_MODULE.LMDeployModel(model_name="Qwen3.5-4B", base_url="http://127.0.0.1:23333/v1", client=client)
    model.generate("hello")
    assert client.requests[0]["max_tokens"] == 4096


def test_lmdeploy_does_not_forward_local_fps_option() -> None:
    client = FakeClient()
    model = LMDEPLOY_MODULE.LMDeployModel(
        model_name="Qwen3.5-4B",
        base_url="http://127.0.0.1:23333/v1",
        client=client,
        fps=1,
    )

    model.generate("hello")

    assert "fps" not in client.requests[0]

MODULE = importlib.util.module_from_spec(SPEC)

sys.modules[SPEC.name] = MODULE

SPEC.loader.exec_module(MODULE)



LLMModel = MODULE.LLMModel





def test_qwen35_uses_lmdeploy_provider() -> None:

    model = LLMModel.__new__(LLMModel)
    assert model._detect_provider("Qwen3.5-4B", None) == "lmdeploy"

def test_llm_wrapper_dispatches_to_lmdeploy() -> None:
    sys.path.insert(0, str(SOURCE.parents[2]))
    from worldmm.llm.llm_wrapper import LLMModel as PackageLLMModel

    client = FakeClient()
    model = PackageLLMModel(
        "Qwen3.5-4B",
        base_url="http://127.0.0.1:23333/v1",
        client=client,
    )
    assert model.generate("hello") == "ready"
    assert client.requests[0]["model"] == "Qwen3.5-4B"

    assert model.provider == "lmdeploy"
# End of LMDeploy provider regression tests.
