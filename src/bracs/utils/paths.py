# ---------------------------------------------
# ARCHIVO DE RUTAS PARA LOS DATOS Y LOS OUTPUTS
# ---------------------------------------------

from __future__ import annotations
from pathlib import Path

# ----------------------
# RUTAS BASE 
# ----------------------

def project_root() -> Path:
    """
    Raíz del proyecto.

    Este archivo está en:
        <root>/src/bracs/utils/paths.py

    Por tanto, la raíz del proyecto es el padre nº 3:
        paths.py      -> parents[0] = .../src/bracs/utils
        utils/        -> parents[1] = .../src/bracs
        bracs/        -> parents[2] = .../src
        src/          -> parents[3] = .../<root>
    """
    return Path(__file__).resolve().parents[3]


def data_root() -> Path:
    """
    Carpeta donde guardamos todos los datasets.

    Todos los datos vivirán en:
        <root>/data/histoimage
    """    
    return project_root() / "data" / "histoimage"


def outputs_root() -> Path:
    """
    Carpeta donde guardamos todas las salidas del proyecto:
    - modelos entrenados
    - logs
    - métricas exportadas
    - figuras, ...
    """
    return project_root() / "outputs"


def results_root() -> Path:
    """
    Carpeta para resultados "procesados":
        tablas, informes intermedios, etc.
    """
    return project_root() / "results"


def runs_root() -> Path:
    """
    Carpeta base para ejecuciones, si queremos guardar cosas
    específicas de cada run fuera de MLflow.
    """
    return project_root() / "runs"

def roi_datasets_root() -> Path:
    """
    Carpeta donde guardamos los datasets(los .pkl y .npy de patches RoI).
    """
    return project_root() / "data" / "datasets" / "roi"


# ----------------------
# RUTAS de BRACS 
# ----------------------
def bracs_roi_root() -> Path:
    """
    Carpeta con las RoIs originales del dataset BRACS.
    """
    return data_root() / "BRACS_RoI"


def bracs_roi_patches_root() -> Path:
    """
    Carpeta con TODOS los patches generados a partir de las RoIs.
    """
    return data_root() / "BRACS_RoI_patches_512_overlap_full"


def bracs_xlsx() -> Path:
    """
    Ruta del Excel con la anotación del dataset BRACS.
    """
    return data_root() / "BRACS.xlsx"


# ----------------------
# RUTAS DE MLFlow 
# ----------------------
def mlflow_root() -> Path:
    """
    Carpeta raíz donde queremos que MLflow guarde los experimentos.
    """
    return outputs_root() / "mlruns"


def models_root() -> Path:
    """
    Carpeta donde guardaremos los pesos de los modelos entrenados.
    """
    return outputs_root() / "models"


def figures_root() -> Path:
    """
    Carpeta para guardar figuras (curvas de entrenamiento, matrices de
    confusión, etc.).
    """
    return outputs_root() / "figures"



# ----------------------
# Para asegurarnos de que existen las carpetas
# ----------------------
def ensure_dirs() -> None:
    """
    Si no existen, creamos las carpetas base del proyecto.
    Esta función la llamaremos al inicio de cualquier script "gordo"
    para asegurarnos de que el entorno de carpetas está preparado.
    """
    for p in [
        data_root(),
        outputs_root(),
        results_root(),
        runs_root(),
        mlflow_root(),
        models_root(),
        figures_root(),
        roi_datasets_root(),
    ]:
        p.mkdir(parents=True, exist_ok=True)