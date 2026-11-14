import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.skipif(os.name != "posix" or not shutil.which("flock"), reason="Linux pipeline locking")
def test_duplicate_launch_preserves_active_run_provenance(tmp_path):
    import fcntl

    run_root = tmp_path / "active"
    run_root.mkdir()
    expected = {"code_commit": "active-commit\n", "working_tree_status": "active-status\n"}
    for name, content in expected.items():
        (run_root / name).write_text(content)
    with (run_root / "pipeline.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(
            ["bash", "script/run_variant_full.sh", "active"],
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, "WORLDMM_PIPELINE_ROOT": str(tmp_path)},
            capture_output=True, text=True, timeout=10,
        )
    assert result.returncode == 2
    assert "already active" in result.stderr
    assert {name: (run_root / name).read_text() for name in expected} == expected
    assert not (run_root / "pipeline.log").exists()
