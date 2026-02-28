# ---------------------------------------------
# DEFINIMOS UN DATASET DE PYTORCH PARA CONVERTIR LOS .pkl Y SUS RUTAS EN TENSORES PARA ENTRENAR LOS MODELOS DE CLASIFICACIÓN DE PATCHES  
# ---------------------------------------------
from __future__ import annotations

from curses import raw
from email.mime import image
from email.mime import image
from pathlib import Path
from typing import Literal, Optional, Dict, Any, Tuple, List

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

from bracs.utils import paths

Split = Literal["train", "val", "test"]

CLASSES_7 = ["N", "PB", "UDH", "FEA", "ADH", "DCIS", "IC"]

CLASSES_3 = ["BT", "AT", "MT"]


def get_class_names(n_clases: int) -> List[str]:
    """
    Devolvemos los nombres de las clases según el nº de clases.
    """
    if n_clases == 3:
        return CLASSES_3
    elif n_clases == 7:
        return CLASSES_7
    else:
        raise ValueError(f"n_clases={n_clases} INCORRECTO, se espera 3 o 7.")
    
    
# ----------------------------------------------
# Carga de los ficheros .pkl con los datasets de ROIs
# ----------------------------------------------

def default_roi_pkl_name(n_clases: int) -> str:
    """
    Elegimos el nombre del .pkl por defecto para un nº de clases dado.
    Si queremos otras variantes (con otra resolución o subset), podremos cambiarlo fácilmente.
    """
    if n_clases == 3:
        return "data_roi_3cls_full.pkl"
    elif n_clases == 7:
        return "data_roi_7cls_full.pkl"
    else:
        raise ValueError(f"n_clases={n_clases} INCORRECTO, se espera 3 o 7.")

def load_roi_pkl(n_clases: int, dataset_name: Optional[str] = None,) -> Dict[str, Dict[str, Any]]:
    """
    Cargamos el diccionario de datos ROI desde un .pkl.

    Estructura esperada del .pkl:
        "train": {"x": np.array([paths...]), "y": np.array(one_hot)},
        "val":   {"x": ..., "y": ...},
        "test":  {"x": ..., "y": ...},
    
    Args:
        n_clases: 3 o 7
        dataset_name: nombre del .pkl dentro de data/datasets/roi.
                      Si es None, usamos el nombre por defecto según n_clases.

    Returns:
        data: diccionario con las claves "train", "val", "test".
    """
    if dataset_name is None:
        dataset_name = default_roi_pkl_name(n_clases)

    # Localizamos la carpeta
    roi_root: Path = paths.roi_datasets_root()
    pkl_path = roi_root / dataset_name

    if not pkl_path.is_file():
        raise FileNotFoundError(f"No se ha encontrado el dataset ROI: {pkl_path}")

    import pickle

    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)

    data = raw["splits"]

    # Comprobamos que no falta ningún split ni clave
    for split in ["train", "val", "test"]:
        if split not in data:
            raise ValueError(f"El .pkl no contiene el split '{split}': {pkl_path}")
        if "x" not in data[split] or "y" not in data[split]:
            raise ValueError(f"El split '{split}' debe tener claves 'x' y 'y'")

    return data


# ----------------------------------------------
# Dataset de Pytorch para parches de ROI
# ----------------------------------------------
class ROIPatchesDataset(Dataset):
    """
    Dataset de Pytorch para los parches de ROI.

    Trabajamos a partir del .pkl que tiene:
        - data[split]["x"]: rutas de imagen 
        - data[split]["y"]: etiquetas one-hot (np.array de shape [N, n_clases])

    Convertimos las etiquetas a:
        - self.labels: vector de enteros (0..n_clases-1)
        - self.paths: lista de Path (para depuración y loggeo)

    Devolvemos en __getitem__:(imagen_tensor, label_int, path_str)
        - imagen_tensor se usa para el modelo
        - label_int para la pérdida
        - path_str para depuración, análisis de errores,...
    """
    

    def __init__(self, split: Split, n_clases: int, transform=None,dataset_name: Optional[str] = None) -> None:
        super().__init__()
        
        if split not in ["train", "val", "test"]:
            raise ValueError(f"split debe ser 'train', 'val' o 'test', recibido: {split}")
        
        # Cargamos el .pkl con los datos de este split
        data = load_roi_pkl(n_clases, dataset_name)
        
        x = data[split]["x"]  # array de rutas (str)
        y = data[split]["y"]  # array de one-hot (np.array [N, n_clases])
        
        y = np.asarray(y)  # por si acaso no es np.array
        
        # Convertimos las etiquetas one-hot a índices
        if y.ndim == 2 and y.shape[1] == n_clases:
            labels = np.argmax(y, axis=1)  # vector de enteros (0..n_clases-1)
        elif y.ndim == 1:
            labels = y # NO NCECESARIO: Es por si queremos guardar labels enteros
        else:
            raise ValueError(f"Forma de y inesperada para split '{split}': {y.shape}")

        # Guardamos las rutas y etiquetas en el dataset
        self.paths: List[Path] = [Path(p) for p in x]
        self.labels: np.ndarray = labels.astype(int)
        self.transform = transform
        self.n_clases = n_clases
        
        # Por si queremos consultar los nombres de clase desde el dataset
        self.class_names = get_class_names(n_clases)
        
    def __len__(self) -> int:
        return len(self.paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """ 
        Devolvemos (imagen_tensor, label_int, path_str) para el índice dado.
        """
        # Cargamos la imagen desde la ruta
        img_path = self.paths[idx]
        label = int(self.labels[idx]) 

        # Abrimos la imagen como RGB con PIL
        imagen = Image.open(img_path).convert("RGB")
        
        # Aplicamos las transformaciones (si hay)
        if self.transform is not None:
            imagen = self.transform(imagen)
        
        # Si no hay transformaciones, convertimos a tensor float32 [C,H,W]
        if not isinstance(imagen, torch.Tensor):
            imagen = torch.from_numpy(np.array(imagen)).permute(2, 0, 1).float() / 255.0
            
        return imagen, label, str(img_path)


