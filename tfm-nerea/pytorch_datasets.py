import torch
from torch.utils import data
import numpy as np
from torchvision import transforms
from cv2 import imread
import pandas as pd

# Slideflow es opcional (solo para normalización de staining)
try:
    import slideflow as sf
    SLIDEFLOW_AVAILABLE = True
except ImportError:
    SLIDEFLOW_AVAILABLE = False
    print("⚠️  Slideflow no disponible - normalización de staining desactivada")

class Dataset(data.Dataset):
    def __init__(self, inputs, labels, transform=None, normalization=None):
        """
        Inicialización del conjunto de datos.

        Args:
            inputs (list): Lista de rutas de archivos de entrada.
            labels (list): Lista de etiquetas correspondientes a los datos de entrada.
            transform (callable, optional): Transformaciones a aplicar a los datos. Por defecto es None.
            normalization (str, optional): Método de normalización de tinción a utilizar ('reinhard', 'macenko' o None).
                                           Por defecto es None.
        """
        self.labels = labels
        self.inputs = inputs
        self.transform = transform
        self.normalization = normalization

        if normalization and not SLIDEFLOW_AVAILABLE:
            print("⚠️  Normalización solicitada pero slideflow no está instalado. Continuando sin normalización.")
            self.normalization = None
        
        if normalization == 'reinhard' and SLIDEFLOW_AVAILABLE:
            self.normalization_method = sf.norm.autoselect('reinhard_fast')
        elif normalization == 'macenko' and SLIDEFLOW_AVAILABLE:
            self.normalization_method = sf.norm.autoselect('macenko')

    def __len__(self):
        """
        Devuelve la longitud del conjunto de datos.

        Returns:
            int: Número total de muestras en el conjunto de datos.
        """
        return len(self.inputs)

    def __getitem__(self, index):
        """
        Genera una muestra de datos en función del índice.

        Args:
            index (int): Índice de la muestra.

        Returns:
            tuple: Tupla que contiene la imagen, la etiqueta y el archivo.
        """
        file = self.inputs[index]
        x = imread(file).astype(np.uint8)

        if self.normalization is not None:
            # Aplicar normalización de tinción
            x = self.normalization_method.transform(x)

        if self.transform:
            x = self.transform(transforms.ToPILImage()(x))

        y = self.labels[index]
        y = torch.from_numpy(np.asarray(y)).float()

        return x, y, file



from PIL import Image

class Dataset_full(data.Dataset):
    def __init__(self, inputs, labels, transform=None,  normalization=None, resize=None):
        """
        Inicialización del conjunto de datos.

        Args:
            inputs (list): Lista de rutas de archivos de entrada.
            labels (list): Lista de etiquetas correspondientes a los datos de entrada.
            transform (callable, optional): Transformaciones a aplicar a los datos. Por defecto es None.
            resize (tuple, optional): Tamaño de redimensionamiento de la imagen. Por defecto es None.
        """
        self.labels = labels
        self.inputs = inputs
        self.transform = transform
        self.resize = resize
        self.normalization = normalization
        
        if normalization and not SLIDEFLOW_AVAILABLE:
            print("⚠️  Normalización solicitada pero slideflow no está instalado. Continuando sin normalización.")
            self.normalization = None
        
        if normalization == 'reinhard' and SLIDEFLOW_AVAILABLE:
            self.normalization_method = sf.norm.autoselect('reinhard_fast')
        elif normalization == 'macenko' and SLIDEFLOW_AVAILABLE:
            self.normalization_method = sf.norm.autoselect('macenko')

    def __len__(self):
        """
        Devuelve la longitud del conjunto de datos.

        Returns:
            int: Número total de muestras en el conjunto de datos.
        """
        return len(self.inputs)

    def __getitem__(self, index):
        """
        Genera una muestra de datos en función del índice.

        Args:
            index (int): Índice de la muestra.

        Returns:
            tuple: Tupla que contiene la imagen, la etiqueta y el archivo.
        """
        file = self.inputs[index]
        x = imread(file).astype(np.uint8)

        # Redimensionar la imagen si se especifica el tamaño de redimensionamiento
        if self.resize:
            x = Image.fromarray(x)
            x = x.resize(self.resize)
            x = np.array(x)

        if self.normalization is not None:
            # Aplicar normalización de tinción
            x = self.normalization_method.transform(x)

        # Aplicar transformaciones adicionales si se especifica
        if self.transform:
            x = Image.fromarray(x)
            x = self.transform(x)

        y = self.labels[index]
        y = torch.from_numpy(np.asarray(y)).float()

        return x, y, file
