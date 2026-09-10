from pathlib import Path


def test_vlm2vec_qwen2_loader_defaults_to_flash_attention_2() -> None:
    source = (Path(__file__).resolve().parents[1] / "src" / "worldmm" / "embedding" / "VLM2Vec" / "src" / "model" / "model.py").read_text()
    assert "WORLDMM_VLM_ATTENTION" in source
    assert 'os.environ.get("WORLDMM_VLM_ATTENTION", "flash_attention_2")' in source
    assert 'config._attn_implementation = attention_implementation' in source
