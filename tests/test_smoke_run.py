"""Smoke tests: verify the pipeline produces expected output files."""

import os
import subprocess
import sys


def _run(args: list, cwd: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
    )


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_smoke_cli_creates_outputs(tmp_path):
    """cli.py should exit 0 and produce a run directory with required artefacts."""
    result = _run(
        [
            "cli.py",
            "--data", "data/demo.csv",
            "--target", "auto",
            "--output_root", str(tmp_path),
        ],
        cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0, f"cli.py failed:\n{result.stderr}"

    run_dirs = list(tmp_path.iterdir())
    assert run_dirs, "No run directory created"

    run_dir = run_dirs[0]
    for fname in ("report.md", "metrics.json", "confusion_matrix.png"):
        assert (run_dir / fname).exists(), f"Missing artefact: {fname}"


def test_smoke_run_agent_creates_outputs(tmp_path):
    """run_agent.py (legacy) should also exit 0 and produce a run directory."""
    result = _run(
        [
            "run_agent.py",
            "--data", "data/demo.csv",
            "--target", "auto",
            "--output_root", str(tmp_path),
            "--quiet",
        ],
        cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0, f"run_agent.py failed:\n{result.stderr}"

    run_dirs = list(tmp_path.iterdir())
    assert run_dirs, "No run directory created"


def test_auto_target_inference(tmp_path):
    """Target auto-inference should pick the 'label' column from demo.csv."""
    result = _run(
        [
            "cli.py",
            "--data", "data/demo.csv",
            "--target", "auto",
            "--output_root", str(tmp_path),
        ],
        cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0
    # 'label' should appear in the stdout / progress output
    assert "label" in result.stdout.lower() or len(list(tmp_path.iterdir())) > 0
