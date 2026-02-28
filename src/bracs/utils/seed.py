# ---------------------------------------------
# Utilidades para controlar la semilla aleatoria del proyecto y asegurar reproducibilidad (en la medida de lo posible).
# ---------------------------------------------

import os
import random
import numpy as np
import torch
from typing import Optional

def set_global_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Fijamos la semilla en todos los sitios relevantes:
        - random 
        - numpy
        - torch 

    Si deterministic=True y torch está disponible, también activamos
    los flags de comportamiento determinista cuando usemos GPU.
    """
    # Para algunas libs que miran esta variable de entorno
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Módulo random estándar
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch (CPU y GPU)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            # Forzamos a cuDNN a usar implementaciones deterministas
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            # Permitimos que cuDNN busque kernels más rápidos (pero menos reproducibles)
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
            

def seed_from_str(text: str, max_value: int = 2**31 - 1) -> int:
    """
    Utilidad opcional: generamos una semilla entera a partir de un string.
    Esto nos permite derivar semillas de nombres de experimentos.
    """
    return abs(hash(text)) % max_value