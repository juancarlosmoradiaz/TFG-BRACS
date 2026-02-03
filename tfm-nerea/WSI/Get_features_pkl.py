

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from ctran import ctranspath
from torch.utils import data
import numpy as np
from cv2 import imread
import slideflow as sf
import argparse
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
from torch.utils import data
from cv2 import imread
import argparse
import os

path_dir='./'
# Crear el objeto ArgumentParser y definir los argumentos
parser = argparse.ArgumentParser(description='Configuración para el entrenamiento del modelo')
parser.add_argument('--data_RoI', type=str, default='data_RoI_256.pkl',
                    help='pkl de datasets')
parser.add_argument('--normalization', type=str, default=None,
                    help='pkl de datasets')
parser.add_argument('--batch_size', type=int, default=None,
                    help='tamaño del batch')
parser.add_argument('--folder_name', type=str, default=None,
                    help='nombre de la carpeta donde guardar las caracteristicas')
# Parsear los argumentos
args = parser.parse_args()
data_RoI_pkl=args.data_RoI
norm=args.normalization
batch_size=args.batch_size
folder_name=args.folder_name

os.makedirs(path_dir+'features/'+folder_name, exist_ok=True)

class Dataset(data.Dataset):
    def __init__(self, inputs,  transform=None, normalization=None):
        """
        Inicialización del conjunto de datos.

        Args:
            inputs (list): Lista de rutas de archivos de entrada.
            labels (list): Lista de etiquetas correspondientes a los datos de entrada.
            transform (callable, optional): Transformaciones a aplicar a los datos. Por defecto es None.
            normalization (str, optional): Método de normalización de tinción a utilizar ('reinhard', 'macenko' o None).
                                           Por defecto es None.
        """
        
        self.inputs = inputs
        self.transform = transform
        self.normalization = normalization

        if normalization == 'reinhard':
            self.normalization_method = sf.norm.autoselect('reinhard_fast')
        elif normalization == 'macenko':
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

        try:
            if self.normalization is not None:
                # Aplicar normalización de tinción
                x = self.normalization_method.transform(x)
        except Exception as e:
            pass
        
        if self.transform:
            x = self.transform(transforms.ToPILImage()(x))
            
        return x

device = torch.device("cuda:0" if torch.cuda.is_available()
                                   else "cpu") 
torch.cuda.empty_cache()

 
""" Crear readers """
dataReaders = {}

with open(data_RoI_pkl, 'rb') as fp:
    data_RoI = pickle.load(fp)
dataReaders['CNN'] = data_RoI

# Datasets
datasets = ['train', 'val','test']

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
trnsfrms = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean, std = std)
    ]
)

#suffle train set
index_train = np.random.randint(0, len(dataReaders['CNN']['train']['x']), len(dataReaders['CNN']['train']['x']))
dataReaders['CNN']['train']['x'] = dataReaders['CNN']['train']['x'][index_train]
dataReaders['CNN']['train']['y'] = dataReaders['CNN']['train']['y'][index_train]

dataset_train = Dataset(dataReaders['CNN']['train']['x'], 
                         trnsfrms, norm)

#crear val dataset
dataset_val = Dataset(dataReaders['CNN']['val']['x'], 
                 trnsfrms, norm)


#crear test dataset
dataset_test = Dataset(dataReaders['CNN']['test']['x'],
                     trnsfrms, norm)


#Crear dataloaders

dataloader_train = DataLoader(dataset_train, batch_size=batch_size,
                                    shuffle=False, num_workers=8, 
                                    pin_memory=True)

dataloader_val = DataLoader(dataset_val, batch_size=batch_size,
                                shuffle=False, num_workers=8, 
                                pin_memory=True)

dataloader_test = DataLoader(dataset_test, batch_size=batch_size,
                                    shuffle=False, num_workers=8,
                                    pin_memory=True)

dataloaders = {'train': dataloader_train, 'val': dataloader_val, 'test': dataloader_test}

model = ctranspath()
model.head = nn.Identity()
td = torch.load(r'./ctranspath.pth')
model.load_state_dict(td['model'], strict=True)
model = model.to(device)
all_features = {}
all_features['train'] = {'Case_Ids': [], 'features': []}
all_features['val'] = {'Case_Ids': [], 'features': []}
all_features['test'] = {'Case_Ids': [], 'features': []}
all_final_datasets = {}

model.eval()
save_path = path_dir+'results/'
with torch.no_grad():
    for phase, dataloader in dataloaders.items():
        for inputs in tqdm(dataloader):
            inputs = inputs.to(device)
            features = model(inputs)
            features = features.cpu().numpy()
            all_features[phase]['features'].append(features)
        # Concatena las características de todos los lotes en un solo array NumPy
        all_features[phase]['features'] = np.concatenate(all_features[phase]['features'], axis=0)
        all_features[phase]['Case_Ids'] = dataReaders['CNN'][phase]['x']
        # Imprime las dimensiones de features y all_features
        print("Dimensiones de all_features:", all_features[phase]['features'].shape)


        ids=[]
        data = pd.DataFrame()
        data['Case_Ids'] = dataReaders['CNN'][phase]['x']
        data['features']= all_features[phase]['features'].tolist()
        data['labels'] = np.argmax(dataReaders['CNN'][phase]['y'], axis=1)
        for j in data['Case_Ids']:
            aux='_'.join(j.split('_')[0:-1])
            ids.append(aux)
        ids=pd.unique(ids)

        final=pd.DataFrame(columns=['Case_id','features']) 
        for k in ids:
            p=data[data['Case_Ids'].str.contains(k)]
            f=np.vstack(p['features'])
            f = f.tolist()
            l=p['labels'].value_counts().idxmax()
            final=final.append({'Case_id':k,'features':f, 'labels':l}, ignore_index=True)
        nombre='features_'+phase+'.parquet'
        final.to_parquet(nombre)