import json
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from worldmm.memory.episodic.memory import EpisodicMemory  # noqa: E402
from worldmm.memory.episodic.utils import compute_mdhash_id  # noqa: E402


def test_seed_precomputed_openie_reuses_ten_second_entities_and_triples(tmp_path: Path) -> None:
    passage = "A surgeon grasps the forceps."
    chunk_id = compute_mdhash_id(passage, prefix="chunk-")
    source = tmp_path / "offline.json"
    source.write_text(
        json.dumps(
            {
                "ner_results": {chunk_id: ["surgeon", "forceps"]},
                "triple_results": {chunk_id: [["surgeon", "grasps", "forceps"]]},
            }
        ),
        encoding="utf-8",
    )
    memory = EpisodicMemory.__new__(EpisodicMemory)
    memory.load_precomputed_openie(str(source))
    target = tmp_path / "cache" / "openie.json"
    hipporag = SimpleNamespace(openie_results_path=str(target))

    memory._seed_precomputed_openie(hipporag, [passage], "10sec")

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["docs"] == [
        {
            "idx": chunk_id,
            "passage": passage,
            "extracted_entities": ["surgeon", "forceps"],
            "extracted_triples": [["surgeon", "grasps", "forceps"]],
        }
    ]


def test_seed_precomputed_openie_marks_coarse_captions_as_already_extracted(tmp_path: Path) -> None:
    source = tmp_path / "offline.json"
    source.write_text(json.dumps({"ner_results": {}, "triple_results": {}}), encoding="utf-8")
    memory = EpisodicMemory.__new__(EpisodicMemory)
    memory.load_precomputed_openie(str(source))
    target = tmp_path / "cache" / "openie.json"
    hipporag = SimpleNamespace(openie_results_path=str(target))

    memory._seed_precomputed_openie(hipporag, ["coarse caption one", "coarse caption two"], "30sec")

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert [row["passage"] for row in payload["docs"]] == ["coarse caption one", "coarse caption two"]
    assert all(row["extracted_entities"] == [] for row in payload["docs"])
    assert all(row["extracted_triples"] == [] for row in payload["docs"])


def test_evaluators_load_persisted_openie_before_indexing() -> None:
    lvbench = (ROOT / "eval" / "eval.py").read_text(encoding="utf-8")
    medvidbench = (ROOT / "eval" / "eval_medvidbench.py").read_text(encoding="utf-8")

    assert "load_episodic_openie" in lvbench
    assert "load_episodic_openie" in medvidbench
