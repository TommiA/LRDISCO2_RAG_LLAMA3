import os
import shutil
import subprocess
import sys

import pytest


def test_query_script_help_runs():
    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    result = subprocess.run(
        [sys.executable, "query_chroma_db_and_llama.py", "--help"],
        cwd=repo_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0
    assert "usage" in result.stdout.lower()


def test_docker_build_and_run_help():
    if not shutil.which("docker"):
        pytest.skip("Docker is not installed")

    image_name = "lrdisco2-rag-llama3-test"
    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    build = subprocess.run(
        ["docker", "build", "-t", image_name, "."],
        cwd=repo_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=300,
    )
    assert build.returncode == 0, build.stderr

    run = subprocess.run(
        ["docker", "run", "--rm", image_name, "--help"],
        cwd=repo_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0, run.stderr
    assert "usage" in run.stdout.lower() or "usage" in run.stderr.lower()
