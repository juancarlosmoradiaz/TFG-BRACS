from pathlib import Path

def project_root() -> Path:
    return Path(__file__).resolve().parents[3]

def outputs_dir() -> Path:
    p = project_root() / "outputs"
    p.mkdir(exist_ok=True)
    return p