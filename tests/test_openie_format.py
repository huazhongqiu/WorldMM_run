import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
from worldmm.llm.lmdeploy import LMDeployModel
from worldmm.memory.episodic.utils import NerRawOutput, TripleRawOutput


class Client:
    def __init__(self, text, finish="stop"):
        self.text, self.finish, self.requests = text, finish, []
        self.chat = SimpleNamespace(completions=self)
    def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.text), finish_reason=self.finish)], usage=None)


def test_openie_schema_constrains_generation_without_changing_qa():
    client = Client('{"triples": [["I", "hold", "cup"]]}')
    model = LMDeployModel("Qwen3.5-4B", base_url="http://localhost/v1", client=client)
    assert model.generate("extract", text_format=TripleRawOutput).triples == [["I", "hold", "cup"]]
    options = client.requests[0]
    schema = options["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["triples"]["items"]["minItems"] == 3
    assert schema["properties"]["triples"]["items"]["maxItems"] == 3
    assert options["temperature"] == 0
    model.generate("answer")
    assert "response_format" not in client.requests[1]
    assert "temperature" not in client.requests[1]


def test_complete_fenced_json_with_explanation_is_accepted():
    client = Client('```json\n{"named_entities": ["Jake"]}\n```\nExplanation: people in the paragraph.')
    model = LMDeployModel("Qwen3.5-4B", base_url="http://localhost/v1", client=client)
    assert model.generate("extract", text_format=NerRawOutput).named_entities == ["Jake"]


def test_truncation_is_explicit_even_if_partial_output_parses():
    client = Client('{"named_entities": []}', finish="length")
    model = LMDeployModel("Qwen3.5-4B", base_url="http://localhost/v1", client=client)
    with pytest.raises(ValueError, match="truncated"):
        model.generate("extract", text_format=NerRawOutput)


def test_long_passage_extraction_preserves_original_passage_and_deduplicates():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
    from egolife_precompute import extract_caption
    passage = ("First paragraph holds a cup. " * 80) + "\n\n" + ("Second paragraph opens a door. " * 80)
    calls = []
    class Extractor:
        def ner(self, key, text):
            return SimpleNamespace(unique_entities=["I"], metadata={})
        def triple_extraction(self, key, text, entities):
            calls.append(text)
            return SimpleNamespace(triples=[["I", "hold", "cup"]], metadata={})
    doc = extract_caption(Extractor(), "original-id", passage)
    assert doc["passage"] == passage and doc["idx"] == "original-id"
    assert doc["extracted_triples"] == [["I", "hold", "cup"]]
    assert len(calls) > 1 and max(map(len, calls)) <= 1200


def test_truncated_segment_is_split_without_identical_retries():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
    from egolife_precompute import extract_caption
    calls = []
    class Extractor:
        def ner(self, key, text):
            return SimpleNamespace(unique_entities=["I"], metadata={})
        def triple_extraction(self, key, text, entities):
            calls.append(text)
            if len(text) > 10:
                return SimpleNamespace(triples=[], metadata={"error": "truncated", "output_truncated": True})
            return SimpleNamespace(triples=[["I", "hold", "cup"]], metadata={})
    assert extract_caption(Extractor(), "id", "one two three four five six")["extracted_triples"]
    assert len(calls) == len(set(calls))


def test_openie_does_not_retry_identical_truncated_request():
    from worldmm.memory.episodic.openie import OpenIE
    client = Client('{"named_entities": []}', finish="length")
    model = LMDeployModel("Qwen3.5-4B", base_url="http://localhost/v1", client=client)
    result = OpenIE(model).ner("chunk", "a passage")
    assert result.metadata["output_truncated"] is True
    assert len(client.requests) == 1


def test_failed_caption_does_not_discard_successful_caption_in_batch(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
    from egolife_precompute import prepare_openie
    memory = SimpleNamespace(captions={"3min": [SimpleNamespace(text=text, timestamp_int=(0,1)) for text in ["bad", "good"]]},
                             llm_model=SimpleNamespace(model_name="llm"), base_granularity="30sec",
                             _snapshot_identity=lambda t: {"base_openie": "hash"})
    class Extractor:
        def ner(self, key, text):
            return SimpleNamespace(unique_entities=["I"], metadata={})
        def triple_extraction(self, key, text, entities):
            if text == "bad": return SimpleNamespace(triples=[], metadata={"error": "invalid reply"})
            return SimpleNamespace(triples=[["I", "hold", "cup"]], metadata={})
    with pytest.raises(RuntimeError, match="saved successful captions"):
        prepare_openie(memory, tmp_path, 1, extractor=Extractor())
    docs = json.loads((tmp_path / "3min.json").read_text())["docs"]
    assert [d["passage"] for d in docs] == ["good"]


def test_openie_keeps_one_initial_system_message_and_does_not_mutate_prompt():
    client = Client('{"named_entities": ["Jake"]}')
    model = LMDeployModel("Qwen3.5-4B", base_url="http://localhost/v1", client=client)
    prompt = [{"role": "system", "content": "Original instructions"}, {"role": "user", "content": "caption"}]
    model.generate(prompt, text_format=NerRawOutput)
    messages = client.requests[0]["messages"]
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "Original instructions" in messages[0]["content"]
    assert prompt[0]["content"] == "Original instructions"


def test_hourly_reextraction_covers_entire_week_without_reusing_legacy(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
    from egolife_precompute import reextract_hourly
    entries = [SimpleNamespace(text=text, timestamp_int=(0,time)) for text,time in [("past",1),("future",200)]]
    memory = SimpleNamespace(captions={"1h": entries}, llm_model=SimpleNamespace(model_name="llm"),
                             base_granularity="30sec", _snapshot_identity=lambda t: {"base_openie": "hash"})
    output=tmp_path / "openie"; output.mkdir()
    legacy={"provenance": {"llm":"llm","base_openie":"hash"},"docs":[{"idx":"old","passage":"past"}]}
    (output / "1h.json").write_text(json.dumps(legacy))
    (output / "3min.json").write_text("keep this unchanged")
    calls=[]
    class Extractor:
        def ner(self, key, text):
            calls.append(text)
            return SimpleNamespace(unique_entities=["I"], metadata={})
        def triple_extraction(self, key, text, entities):
            return SimpleNamespace(triples=[["I","hold","cup"]],metadata={})
    reextract_hourly(memory,output,extractor=Extractor())
    docs=json.loads((output / "1h.json").read_text())["docs"]
    assert {d["passage"] for d in docs} == {"past","future"}
    assert all(d["extraction"]["policy"] == "bounded-openie-v1" for d in docs)
    assert calls == ["past","future"] or calls == ["future","past"]
    assert (output / "3min.json").read_text() == "keep this unchanged"
    backup=tmp_path / "hourly_reextract" / "legacy-1h.json"
    assert json.loads(backup.read_text()) == legacy
    reextract_hourly(memory,output,extractor=Extractor())
    assert len(calls)==2


def test_incomplete_hourly_reextraction_keeps_original_active(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
    from egolife_precompute import reextract_hourly
    memory = SimpleNamespace(captions={"1h":[SimpleNamespace(text="bad",timestamp_int=(0,1))]},
                             llm_model=SimpleNamespace(model_name="llm"),base_granularity="30sec",_snapshot_identity=lambda t:{"base_openie":"hash"})
    output=tmp_path / "openie";output.mkdir();(output/"1h.json").write_text("original")
    class Extractor:
        def ner(self,key,text):return SimpleNamespace(unique_entities=[],metadata={"error":"failed"})
    with pytest.raises(RuntimeError):reextract_hourly(memory,output,extractor=Extractor())
    assert (output/"1h.json").read_text()=="original"
