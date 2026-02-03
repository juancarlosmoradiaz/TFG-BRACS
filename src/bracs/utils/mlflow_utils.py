import mlflow
from pathlib import Path
import subprocess

def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"

def setup_mlflow(experiment_name: str, tracking_dir: Path) -> None:
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file:{tracking_dir.as_posix()}")
    mlflow.set_experiment(experiment_name)