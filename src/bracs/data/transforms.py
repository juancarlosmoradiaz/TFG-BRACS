# ---------------------------------------------
# TRANSFORMACIONES PARA PARCHES ROI
# ---------------------------------------------

from __future__ import annotations

from typing import Dict, Literal

from torchvision import transforms

NivelAugmentation = Literal["none", "light", "heavy"]
TipoNormalizacion = Literal["none", "imagenet"]

def imagenet_estadisticas():
    """
    Devolvemos las estadísticas de media y desviación estándar de ImageNet para normalizar los parches.
    Esto será util para usar modelos preentrenados en ImageNet, que esperan esta normalización.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return {"mean": mean, "std": std}

def transformaciones_roi(tam_imagen:int = 512, 
                        nivel_augmentation:NivelAugmentation = "light", 
                        tipo_normalizacion:TipoNormalizacion = "imagenet") -> Dict[str, transforms.Compose]:
    """  
    Construimos un diccionario con las transformaciones para cada split (train, val, test) a partir de los parámetros:
    Args:    
        - tam_imagen: tamaño al que redimensionar los parches (ejemplo: 512)
        - nivel_augmentation: 
            "none": sin data augmentation, solo resize y normalización
            "light": augmaentation suave
            "heavy": augmentation más agresiva 
        - tipo_normalizacion: 
            "none": no se normaliza, solo se pasa a tensor
            "imagenet": se normaliza con las estadísticas de ImageNet
    Returns:
        Un diccionario con las transformaciones para cada split:
            "train": transforms.Compose([...]),
            "val": transforms.Compose([...]),
            "test": transforms.Compose([...])    
    """
    # Transformación base: to tensor
    # AÑDIDO RESIZE PARA VITS, YA QUE ESTOS MODELOS ESPERAN 512x512

    operaciones_base = [
        transforms.Resize((tam_imagen, tam_imagen)),
        transforms.ToTensor(),
    ]    
    
    if tipo_normalizacion == "imagenet":
        media, std = imagenet_estadisticas().values()
        operaciones_base.append(transforms.Normalize(media, std))
    elif tipo_normalizacion == "none":
        pass
    else:
        raise ValueError(f"Tipo de normalización no reconocido: {tipo_normalizacion}")
    
    transformaciones_base = transforms.Compose(operaciones_base)
    
    # AUGMENTATION PARA TRAIN
    if nivel_augmentation == "none":
        train_aug_ops = []
    elif nivel_augmentation == "light":
        train_aug_ops = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
        ]
    elif nivel_augmentation == "heavy":
        train_aug_ops = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.2, 
                                   contrast=0.2, 
                                   saturation=0.2, 
                                   hue=0.1),
        ]
    else:
        raise ValueError(f"Nivel de augmentation no reconocido: {nivel_augmentation}")
    
    # COMPOSE PARA CADA SPLIT
    
    transformaciones_train = transforms.Compose(train_aug_ops + [transformaciones_base])
        
    transformaciones_val = transformaciones_base
    
    transformaciones_test = transformaciones_base
    
    return {
        "train": transformaciones_train,
        "val": transformaciones_val,
        "test": transformaciones_test,
    }
    