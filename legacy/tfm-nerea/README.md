# TFG BRACS
Repositorio del Trabajo de Fin de Grado orientado a la comparación sistemática de técnicas de preprocesamiento en modelos de Deep Learning para la clasificación de imágenes histopatológicas de mama a partir del dataset **BRACS**.


## Estructura del proyecto (inicial, irá cambiando poco a poco)

```text
tfg-bracs/
├── data/.      # Contiene todos los datos del proyecto
│   ├── histoimage/
│   │   ├── BRACS.xlsx
│   │   ├── BRACS_RoI/
│   │   │   └── latest_version/ # Contiene las imágenes RoI en sus splits oficiales:
│   │   │       ├── train/
│   │   │       ├── val/
│   │   │       └── test/
│   │   └── BRACS_RoI_patches_512_overlap_full/ # Contiene todos los patches generados a partir de las RoI, también organizados por split.
│   │       ├── train/
│   │       ├── val/
│   │       └── test/
│   └── datasets/ # Contiene los datasets construidos a partir de los patches, en formato .pkl y .npy.
│       └── roi/
│           ├── data_roi_3cls_full.pkl
│           ├── data_roi_3cls_full.npy
│           ├── data_roi_7cls_full.pkl
│           └── data_roi_7cls_full.npy
│
├── outputs/
│   ├── mlruns/
│   ├── models/
│   └── figures/
│
├── results/
│
├── runs/
│
├── src/
│   └── bracs/
│       ├── data/
│       │   ├── __init__.py
│       │   ├── inspector.py. # Script de inspección del dataset para contar muestras por split y por clase.
│       │   ├── make_datasets.py. # Script para construir los datasets .pkl/.npy a partir de la estructura de patches.
│       │   ├── roi_dataset.py. # Define el dataset de PyTorch para cargar patches RoI desde los .pkl.
│       │   ├── dataloaders.py # Construye los DataLoader de train/val para PyTorch.
│       │   └── transforms.py  # Define las transformaciones de entrada.
│       │
│       ├── experiments/
│       │   └── train_cnn_roi.py
│       │
│       └── utils/
│           ├── __init__.py
│           ├── paths.py. # Centraliza todas las rutas del proyecto.
│           └── seed.py. # Fija la semilla global del experimento para reproducibilidad.
└── .gitignore 
```

## FASE 1: Benchmark de modelos para clasificación de patches histopatológicos

En esta fase del proyecto trabajamos a nivel de **patches** extraídos de las **Regions of Interest (RoI)**, con el objetivo de construir un **ranking inicial de modelos** que sirva como base para seleccionar el mejor candidato y, posteriormente, optimizarlo y extender el estudio a niveles superiores de análisis.

---

### Objetivo de esta fase

La fase actual del proyecto consiste en construir un **benchmark inicial, limpio y reproducible**, comparando distintas familias de modelos sobre el problema de clasificación de patches RoI de BRACS.

### Decisiones metodológicas fijadas en esta fase

- Trabajamos **solo con patches**
- Usamos únicamente los splits **train** y **val**
- El split **test no se toca**
- El benchmark inicial se hace en **7 clases**
- El baseline inicial se realiza:
  - **sin data augmentation**
  - **sin normalización adicional**
- Repetimos cada experimento con **5 semillas distintas**
- Las métricas principales son:
  - `val_f1_macro`
  - `val_accuracy`

### Semillas fijadas

Las semillas del benchmark son:

- `13`
- `29`
- `47`
- `71`
- `101`

---
