# ---------------------------------------------
# Script para controlar la semilla aleatoria del proyecto y asegurar reproducibilidad (en la medida de lo posible).
# ---------------------------------------------

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Fijamos la semilla global del experimento.

    Con esto intentamos que la ejecución sea lo más reproducible posible,
    controlando:
        - random de Python
        - NumPy
        - PyTorch CPU
        - PyTorch CUDA

    Además, activamos configuraciones de PyTorch orientadas a reproducibilidad.
    """
    # Python
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Variable de entorno para hashing reproducible en Python
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Configuración de cudnn para reproducibilidad
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """
    Inicializamos la semilla de cada worker del DataLoader.

    Esto ayuda a que, cuando usamos varios workers, las operaciones aleatorias
    sigan una pauta reproducible derivada de la semilla principal.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)