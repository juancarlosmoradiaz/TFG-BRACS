from __future__ import annotations
from pathlib import Path
import os
import mlflow

from . import paths

def tracking_uri() -> str:
    # Guardamos el tracking en outputs/runs/mlruns
    return f"file:{paths.outputs_root() / 'runs' / 'mlruns'}"

def set_experiment(experiment_name: str = "tfg-bracs") -> None:
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment(experiment_name)

def start_run(run_name: str | None = None, tags: dict | None = None):
    # Context manager: with start_run:
    set_experiment()
    return mlflow.start_run(run_name=run_name, tags=tags)

def log_common_tags() -> None:
    # Tags útiles para reproducibilidad
    try:
        import subprocess
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        mlflow.set_tag("git_commit", sha)
    except Exception:
        pass
    mlflow.set_tag("project_root", str(paths.project_root()))
    mlflow.set_tag("data_root", str(paths.data_root()))
