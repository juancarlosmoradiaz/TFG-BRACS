from __future__ import annotations
from pathlib import Path

def project_root() -> Path:
    # .../tfg-bracs
    return Path(__file__).resolve().parents[3]

def data_root() -> Path:
    # .../tfg-bracs/data/histoimage  (symlink al dataset real del servidor)
    return project_root() / "data" / "histoimage"

def bracs_xlsx() -> Path:
    return data_root() / "BRACS.xlsx"

def roi_root() -> Path:
    # .../BRACS_RoI/latest_version/{train,val,test}
    return data_root() / "BRACS_RoI" / "latest_version"

def outputs_root() -> Path:
    return project_root() / "outputs"

def ensure_dirs() -> None:
    (outputs_root() / "runs").mkdir(parents=True, exist_ok=True)
    (outputs_root() / "models").mkdir(parents=True, exist_ok=True)
    (outputs_root() / "logs").mkdir(parents=True, exist_ok=True)
    (outputs_root() / "tmp").mkdir(parents=True, exist_ok=True)
    (outputs_root() / "wsi" / "patches").mkdir(parents=True, exist_ok=True)
    (outputs_root() / "wsi" / "masks").mkdir(parents=True, exist_ok=True)
    (outputs_root() / "wsi" / "features").mkdir(parents=True, exist_ok=True)
    (outputs_root() / "wsi" / "features_datasets").mkdir(parents=True, exist_ok=True)
