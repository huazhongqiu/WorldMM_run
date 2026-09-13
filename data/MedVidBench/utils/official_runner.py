"""Run the external MedVidBench evaluator with local LLM configuration.

Clone of ``videospy/benchmark/medvidbench/official_runner.py``: monkey-patches
``openai.OpenAI`` so the official leaderboard scripts (written for GPT-4.1)
call a local OpenAI-compatible endpoint instead, forcing the model name on
every chat.completions.create call.
"""

import argparse
import os
import runpy
import sys


def _configure_openai(model: str, api_base: str | None) -> None:
    if not os.getenv("OPENAI_API_KEY"):
        return

    import openai

    original_openai = openai.OpenAI

    class CompletionsProxy:
        def __init__(self, completions):
            self._completions = completions

        def create(self, *args, **kwargs):
            kwargs["model"] = model
            return self._completions.create(*args, **kwargs)

    class ChatProxy:
        def __init__(self, chat):
            self._chat = chat
            self.completions = CompletionsProxy(chat.completions)

        def __getattr__(self, name):
            return getattr(self._chat, name)

    class ClientProxy:
        def __init__(self, client):
            self._client = client
            self.chat = ChatProxy(client.chat)

        def __getattr__(self, name):
            return getattr(self._client, name)

    def configured_openai(*args, **kwargs):
        if api_base:
            kwargs.setdefault("base_url", api_base)
        return ClientProxy(original_openai(*args, **kwargs))

    openai.OpenAI = configured_openai


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument("--api-base", default=None)
    parser.add_argument("script")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    _configure_openai(args.model, args.api_base)
    sys.argv = [args.script, *args.arguments]
    runpy.run_path(args.script, run_name="__main__")


if __name__ == "__main__":
    main()
