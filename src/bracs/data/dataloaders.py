# ---------------------------------------------
# DATA LOADERS PARA PARCHES ROI
# ---------------------------------------------

from __future__ import annotations

from typing import Literal, Tuple, Optional

from torch.utils.data import DataLoader

from bracs.data.roi_dataset import ROIPatchesDataset
from bracs.data.transforms import (
    transformaciones_roi,
    NivelAugmentation,
    TipoNormalizacion,
)

Split = Literal["train", "val", "test"]

def get_roi_dataloaders(n_clases: int,
                       batch_size: int = 32,
                       tam_imagen: int = 512,
                       nivel_augmentation: NivelAugmentation = "light",
                       tipo_normalizacion: TipoNormalizacion = "imagenet",
                       num_workers: int = 4,
                       dataset_name: Optional[str] = None,
                       ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Construimos un DataLoader para los parches ROI a partir de los parámetros:
    Args:
        - n_clases: número de clases (3 o 7)
        - batch_size: tamaño del batch
        - nivel_augmentation: nivel de data augmentation ("none", "light", "heavy")
        - tipo_normalizacion: tipo de normalización ("none", "imagenet")
        - num_workers: número de workers para cargar los datos
        - dataset_name: nombre del .pkl 
        
    Returns:
        (train_loader, val_loader, test_loader)
        Cargamos tambien el test, aunque lo usaremos SOLO para el script final
    """
    
    # Construimos las transformaciones para cada split
    transfromaciones = transformaciones_roi(tam_imagen, nivel_augmentation, tipo_normalizacion)
    
    # Construimos los datasets para cada split
    ds_train = ROIPatchesDataset(
        split="train", 
        n_clases=n_clases, 
        transform=transfromaciones["train"], 
        dataset_name=dataset_name
    )
    
    ds_val = ROIPatchesDataset(
        split="val", 
        n_clases=n_clases, 
        transform=transfromaciones["val"], 
        dataset_name=dataset_name
    )
    
    ds_test = ROIPatchesDataset(
        split="test", 
        n_clases=n_clases, 
        transform=transfromaciones["test"], 
        dataset_name=dataset_name
    )
    
    
    # Construimos los DataLoaders para cada split
    train_loader = DataLoader(
        ds_train, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True, # Para acelerar la transferencia de la RAM a GPU
    )
    
    
    val_loader = DataLoader(
        ds_val, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        ds_test, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader


def get_roi_train_val_dataloaders(
    n_clases: int,
    batch_size: int,
    tam_imagen: int = 512,
    nivel_augmentation: str = "light",
    tipo_normalizacion: str = "imagenet",
    num_workers: int = 4,
    dataset_name: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Solo construye y devuelve DataLoaders de:
        - train
        - val
    El split de test NI SIQUIERA se carga aquí, para evitar
    tentaciones de usarlo antes de tiempo. El script de entrenamiento
    trabajará SIEMPRE con esta función.
    """
    tfms = transformaciones_roi(
        tam_imagen=tam_imagen,
        nivel_augmentation=nivel_augmentation,
        tipo_normalizacion=tipo_normalizacion,
    )

    ds_train = ROIPatchesDataset(
        split="train",
        n_clases=n_clases,
        transform=tfms["train"],
        dataset_name=dataset_name,
    )

    ds_val = ROIPatchesDataset(
        split="val",
        n_clases=n_clases,
        transform=tfms["val"],
        dataset_name=dataset_name,
    )

    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader