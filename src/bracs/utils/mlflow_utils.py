# ---------------------------------------------
# Pequeña capa de utilidad sobre MLflow.

# El objetivo es que todos los scripts de entrenamiento usen SIEMPRE las mismas convenciones de:
#    tracking_uri
#    nombre de experimento
#    tags comunes (dataset, tipo de modelo, etc.)
# ---------------------------------------------

from contextlib import contextmanager
import os
import socket
import subprocess
from datetime import datetime
from typing import Dict, Optional

import mlflow

from . import paths

# Nombre por defecto del experimento de MLflow
DEFAULT_EXPERIMENT_NAME = "bracs-roi-patches"


def _configure_mlflow() -> None:
    """
    Configuramos MLflow para que use nuestro FileStore local en:
        <outputs>/mlruns
    Llamamos a esta función desde start_run().
    """
    tracking_dir = paths.mlflow_root()
    tracking_dir.mkdir(parents=True, exist_ok=True)

    # Usamos una URI de tipo "file://..."
    mlflow.set_tracking_uri(tracking_dir.as_uri())
    
def _get_git_commit() -> Optional[str]:
    """
    Intentamos obtener el hash del commit actual de git, si el repo
    está versionado. Si falla, devolvemos None.
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=paths.project_root(),
        )
        return out.decode("utf-8").strip()
    except Exception:
        return None
    

@contextmanager
def start_run(run_name: Optional[str] = None, experiment_name: str = DEFAULT_EXPERIMENT_NAME, tags: Optional[Dict[str, str]] = None,):
    """
    Context manager para arrancar un run de MLflow.

    Parámetros
    ----------
    run_name : str, opcional
        Nombre legible del run (aparece en MLflow UI).
    experiment_name : str
        Nombre del experimento de MLflow (se crea si no existe).
    tags : dict, opcional
        Tags iniciales a añadir al run.
    """
    _configure_mlflow()

    # Nos aseguramos de que el experimento existe
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        if tags:
            mlflow.set_tags(tags)
        yield run
        

def log_common_tags(extra: Optional[Dict[str, str]] = None) -> None:
    """
    Registramos un conjunto de tags comunes para todos los runs:
    """
    base_tags = {
        "host": socket.gethostname(),
        "user": os.environ.get("USER", "unknown"),
        "dataset": "BRACS_RoI_patches_512_overlap_full",
        "repo_root": str(paths.project_root()),
        "datetime": datetime.now().isoformat(timespec="seconds"),
    }

    git_commit = _get_git_commit()
    if git_commit is not None:
        base_tags["git_commit"] = git_commit

    if extra:
        base_tags.update(extra)

    mlflow.set_tags(base_tags)


def log_params_from_dict(params: Dict) -> None:
    """
    Helper para registrar un diccionario de hiper-parámetros.
    """
    mlflow.log_params(params)


def log_metrics_from_dict(metrics: Dict, step: Optional[int] = None) -> None:
    """
    Helper para registrar un diccionario de métricas.
    """
    mlflow.log_metrics(metrics, step=step)


def log_figure(fig, artifact_name: str) -> None:
    """
    Registramos una figura de matplotlib en MLflow.

    Parámetros
    ----------
    fig : matplotlib.figure.Figure
        La figura ya construida.
    artifact_name : str
        Nombre con el que queremos guardar la figura dentro del run.

    """
    # Nos aseguramos de que en el filesystem también tengamos carpeta de figuras
    paths.figures_root().mkdir(parents=True, exist_ok=True)

    # mlflow.log_figure guarda y sube la figura directamente
    mlflow.log_figure(fig, artifact_file=artifact_name)
