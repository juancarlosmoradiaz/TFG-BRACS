import torch
from torch.utils import data
import numpy as np
from torchvision import transforms
from cv2 import imread
import pandas as pd
from typing import Tuple
import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available()
                                   else "cpu") 

class Dataset(data.Dataset):
    def __init__(self, features, labels):
        self.labels = labels
        self.features = features
       
    def __len__(self):
        return len(self.features)

    def __getitem__(self, item: np.int64) -> Tuple[torch.Tensor, torch.Tensor]:
        print(type(self.features[item]))
        features_array = np.asarray(self.features[item], dtype=np.float32)
        features_tensor = torch.tensor(features_array, dtype=torch.float32)

        # Crear un tensor unidimensional para las etiquetas
        labels_tensor = torch.tensor(self.labels[item], dtype=torch.float32)

        # Verifica las dimensiones de los tensores
        print("Dimensiones de features:", features_tensor.size())
        print("Dimensiones de labels:", labels_tensor.size())

        return (features_tensor, labels_tensor)