import argparse
import json
import os
import pickle
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import random
import torchvision.transforms as transforms
from pytorch_datasets import Dataset, Dataset_full
from train_pred import  train_model
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
from torch.utils import data
from cv2 import imread
import random
import argparse
import pytorch_warmup as warmup
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

# Crear el objeto ArgumentParser y definir los argumentos
parser = argparse.ArgumentParser(description='Configuración para el entrenamiento del modelo')

parser.add_argument('--lr', type=float, default=3e-3,
                    help='Tasa de aprendizaje (lr)')
parser.add_argument('--epochs', type=int, default=50,
                    help='Número de épocas')
parser.add_argument('--bool_lr_scheduler', type=int, default=1,
                    help='Indicador booleano para habilitar o deshabilitar el ajuste de tasa de aprendizaje')
parser.add_argument('--results_folder_name', type=str, default='resultados_v2',
                    help='Nombre de la carpeta para los resultados')
parser.add_argument('--max_patches', type=int, default=None,
                    help='Número máximo de patches por imágenes')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Tamaño del batch')
parser.add_argument('--data_RoI', type=str, default='data_RoI_256.pkl',
                    help='pkl de datasets')
parser.add_argument('--data_augmentation', type=int, default=0,
                    help='Indicador booleano para habilitar o deshabilitar el data augmentation')
parser.add_argument('--weightsbyclass', type=int, default=0,
                    help='Indicador booleano para habilitar o deshabilitar los pesos segun el tamaño de la clase')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout')
parser.add_argument('--patch_size', type=int, default=512,
                    help='Tamaño del patch')
parser.add_argument('--warmup', type=str, default='linear',
                        choices=['linear', 'exponential', 'radam', 'none'],
                        help='warmup schedule')
parser.add_argument('--lr_min', type=float, default=1e-6,
                    help='lr mínimo')
parser.add_argument('--model', type=str, default='resnet18',
                    help='Nombre de la red preentrenada')
parser.add_argument('--normalization', type=str, default=None,
                    help='pkl de datasets')
parser.add_argument('--im_size', type=int, default=512,
                    help='Tamaño de la imagen')
parser.add_argument('--full', type=int, default=0,
                    help='Indicador booleano para habilitar o deshabilitar tratar con full imagenes')
parser.add_argument('--n_clases', type=int, default=3,
                    help='Número de clases')
parser.add_argument('--optimizer', type=str, default='AdamW',
                        choices=['AdamW', 'RAdam', 'NAdam', 'SGD'],
                        help='Optimizador')
# Parsear los argumentos
args = parser.parse_args()

# Acceder a los valores de los argumentos
batch_size = args.batch_size
patch_size = args.patch_size
lr = args.lr
epochs = args.epochs
bool_lr_scheduler = bool(args.bool_lr_scheduler)
results_folder_name = args.results_folder_name
max_patches=args.max_patches
path_dir='./'
data_RoI_pkl=args.data_RoI
data_augmentation=bool(args.data_augmentation)
full=bool(args.full)
weightsbyclass=bool(args.weightsbyclass)
model_name='modelo'+data_RoI_pkl+'_'+str(max_patches)
dropout=args.dropout
model_cnn = args.model
norm=args.normalization
n = args.im_size
n_clases=args.n_clases
optimizer_name=args.optimizer

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

device = torch.device("cuda:0" if torch.cuda.is_available()
                                   else "cpu") 
torch.cuda.empty_cache()


# Crear carpeta para guardar los pesos
os.makedirs(path_dir+'weights/'+results_folder_name, exist_ok=True)
os.makedirs(path_dir+'results/'+results_folder_name, exist_ok=True)
    
""" Crear readers """
dataReaders = {}

with open(data_RoI_pkl, 'rb') as fp:
    data_RoI = pickle.load(fp)
dataReaders['CNN'] = data_RoI

# Datasets
datasets = ['train', 'val','test']

#suffle train set
index_train = np.random.randint(0, len(dataReaders['CNN']['train']['x']), len(dataReaders['CNN']['train']['x']))
dataReaders['CNN']['train']['x'] = dataReaders['CNN']['train']['x'][index_train]
dataReaders['CNN']['train']['y'] = dataReaders['CNN']['train']['y'][index_train]

data = pd.DataFrame()
data['Case_Ids'] = dataReaders['CNN']['train']['x']

ids = []

#Seleccionar un maximo de n patches por cada foto
if max_patches!=None:
    for j in data['Case_Ids']:
        aux = '_'.join(j.split('_')[0:-1])
        ids.append(aux)

    ids = pd.unique(ids)

    selected_indices = []

    for k in ids:
        p = data[data['Case_Ids'].str.contains(k)]
        indices_aleatorios = random.sample(range(len(p)), min(len(p), max_patches))
        filas_aleatorias = p.iloc[indices_aleatorios]
        selected_indices.extend(filas_aleatorias.index.tolist())


    dataReaders['CNN']['train']['x'] = dataReaders['CNN']['train']['x'][selected_indices]
    dataReaders['CNN']['train']['y'] = dataReaders['CNN']['train']['y'][selected_indices]


train_transform = transforms.Compose([
    # Aplicar la personalización de la imagen
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

if data_augmentation and not full:
    n=patch_size-112
    train_transform = transforms.Compose([
    # Aplicar la personalización de la imagen
    transforms.ToTensor(),
    transforms.RandomCrop((n, n)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation((0, 180)),
    transforms.ColorJitter(brightness= 0.5, contrast= 0.5, saturation=  0.5, hue= 0.3),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

elif data_augmentation and full:
    train_transform = transforms.Compose([
    # Aplicar la personalización de la imagen
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation((0, 180)),
    transforms.ColorJitter(brightness= 0.5, contrast= 0.5, saturation=  0.5, hue= 0.3),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


if full:
    dataset_train = Dataset_full(dataReaders['CNN']['train']['x'], 
                            dataReaders['CNN']['train']['y'], train_transform, norm, (n,n))

    #crear val dataset
    dataset_val = Dataset_full(dataReaders['CNN']['val']['x'], 
                    dataReaders['CNN']['val']['y'], val_transform, norm, (n,n))


    #crear test dataset
    dataset_test = Dataset_full(dataReaders['CNN']['test']['x'],
                        dataReaders['CNN']['test']['y'], val_transform, norm, (n,n))

else:
    dataset_train = Dataset(dataReaders['CNN']['train']['x'], 
                            dataReaders['CNN']['train']['y'], train_transform, norm)
   
    #crear val dataset
    dataset_val = Dataset(dataReaders['CNN']['val']['x'], 
                    dataReaders['CNN']['val']['y'], val_transform, norm)

    
    #crear test dataset
    dataset_test = Dataset(dataReaders['CNN']['test']['x'],
                        dataReaders['CNN']['test']['y'], val_transform, norm)
    

#Crear dataloaders

dataloader_train = DataLoader(dataset_train, batch_size=batch_size,
                                    shuffle=False, num_workers=8, 
                                    pin_memory=True)

dataloader_val = DataLoader(dataset_val, batch_size=8,
                                shuffle=False, num_workers=4, 
                                pin_memory=True)

dataloader_test = DataLoader(dataset_test, batch_size=1,
                                    shuffle=False, num_workers=1,
                                    pin_memory=True)


dataloaders = {'train': dataloader_train, 'val': dataloader_val, 'test': dataloader_test}

#obtener tamaños de los datasets
dataset_sizes = {x: len(dataReaders['CNN'][x]['x']) for x in ['train', 'val', 'test']}

'''
ResNet
Una red residual, o ResNet para abreviar, es una red neuronal artificial que ayuda a construir redes neuronales más profundas utilizando conexiones de salto o atajos para saltarse algunas capas. Verá cómo los saltos ayudan a construir capas de red más profundas sin caer en el problema de la desaparición de gradientes.

Existen diferentes versiones de ResNet, entre ellas ResNet-18, ResNet-34, ResNet-50, etcétera. Los números denotan capas, aunque la arquitectura es la misma.
'''
# Crear un modelo a partir de resnet18 o resnet50

if model_cnn=='resnet18':
    model = torchvision.models.resnet18(weights='DEFAULT' , progress=True) 
elif model_cnn=='resnet50':
    model = torchvision.models.resnet50(weights='DEFAULT' , progress=True)

if n_clases==3: 
    clases=['AT', 'BT', 'MT']
elif n_clases==7: 
    clases=['N', 'PB', 'UDH', 'FEA', 'ADH', 'DCIS', 'IC']
else:
    clases=['PB', 'UDH', 'FEA', 'ADH', 'DCIS', 'IC']

for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features

#cambiar la última capa con nuestras clases
model.fc =  nn.Sequential(
                nn.Dropout(p=dropout),  
                nn.Linear(num_ftrs, len(clases))
            )

# iniciación de pesos con xavier uniform
model.fc.apply(init_weights)

#registrar operaciones en este tensor
for param in model.fc.parameters():
    param.requires_grad = True
for param in model.layer3.parameters():
    param.requires_grad = True

#pasar el modelo a la GPU
model = model.to(device)


class_counts = torch.sum(torch.from_numpy(dataset_train.labels), dim=0)

# Calcular los pesos inversos proporcionales al tamaño de las clases
class_weights = 1.0 / class_counts.float()
class_weights = class_weights / torch.sum(class_weights)

# Crear un tensor de pesos
weight_tensor = torch.zeros(len(class_counts))
for i, class_index in enumerate(class_counts.nonzero()):
    weight_tensor[class_index] = class_weights[i]


if weightsbyclass:
   # Ejemplo de uso de los pesos en la función de pérdida
    criterion = nn.CrossEntropyLoss(weight=weight_tensor.to(device))
else:
    criterion = nn.CrossEntropyLoss()


# guardar resultados en path
save_path = path_dir+'results/'+results_folder_name+'/'

params = [p for p in model.parameters() if p.requires_grad]

if optimizer_name=='AdamW': 
    # Configura el optimizador con weight decay
    optimizer = optim.AdamW(params, lr=lr)
elif optimizer_name=='RAdam':
    optimizer = optim.RAdam(params, lr=lr)
elif optimizer_name=='NAdam':
    optimizer = optim.NAdam(params, lr=lr)
elif optimizer_name=='SGD':
    optimizer = optim.SGD(params, lr=lr)
    
#Disminuye la tasa de aprendizaje de cada grupo de parámetros por gamma cada step_size epochs
if bool_lr_scheduler:
    num_steps = len(dataloader_train) * args.epochs
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_steps, eta_min=args.lr_min)
    if args.warmup == 'linear':
        warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
    elif args.warmup == 'exponential':
        warmup_scheduler = warmup.UntunedExponentialWarmup(optimizer)
    elif args.warmup == 'radam':
        warmup_scheduler = warmup.RAdamWarmup(optimizer)
    elif args.warmup == 'none':
        warmup_scheduler = warmup.LinearWarmup(optimizer, 1)
#entrenamiento del modelo
results = train_model(model=model, criterion=criterion, optimizer=optimizer, dataloaders=dataloaders, 
            dataset_sizes=dataset_sizes, lr_scheduler=lr_scheduler,warmup_scheduler=warmup_scheduler, save_path=save_path, num_epochs=epochs, verbose=True)
writer.flush()
writer.close()
print('Best acc : ', results['best_acc'])

print("Saving model's weights to folder")
torch.save(results['model'].state_dict(), 'weights/'+results_folder_name+'/final_weights.pkl')

# escribir los resultados del diccionario en un archivo
a_file = open(save_path+'results_Epoch_'+str(results['best_epoch'])+'.pkl', "wb")
pickle.dump(results, a_file)
a_file.close()

model = results['model']

model.eval()

# guardar predicciones de val y train

for i in ['val', 'train']:
  data = pd.DataFrame()
  data['Case_Ids'] = dataReaders['CNN'][i]['x']
  data['Preds'] = results['_'.join([i,'preds'])]
  data['Real'] = val_labels = np.argmax(dataReaders['CNN'][i]['y'], axis=1)
  data.to_excel(save_path+i+'_Epoch'+str(results['best_epoch'])+'.xlsx')

#Prediccion de las imagenes completas

from sklearn.metrics import accuracy_score
for i in ['val', 'train']:
  data = pd.DataFrame()
  data['Case_Ids'] = dataReaders['CNN'][i]['x']
  if i=="train":
    data['Preds'] = results['train_preds']
    data['Real'] =  results['train_labels']
  else:
    data['Preds'] = results['val_preds']
    data['Real'] =  results['val_labels']
  ids=[]
  for j in data['Case_Ids']:
    aux='_'.join(j.split('_')[0:-1])
    ids.append(aux)
  ids=pd.unique(ids)
  final=pd.DataFrame(columns=['Case_id','preds','real'])
  for k in ids:
    p=data[data['Case_Ids'].str.contains(k)]
    pred=p['Preds'].value_counts().idxmax()
    real=p['Real'].value_counts().idxmax()
    if n_clases==3:
        labels = str(np.where(real == 0, 'AT', np.where(real == 1, 'BT', 'MT')))
        preds= str(np.where(pred == 0, 'AT', np.where(pred == 1, 'BT', 'MT')).astype(str))
    elif n_clases==7:
        label_mapping = {
        0: 'N',
        1: 'PB',
        2: 'UDH',
        3: 'FEA',
        4: 'ADH',
        5: 'DCIS',
        6: 'IC'
        }

        # Mapear los valores reales a etiquetas
        labels = str(label_mapping[real] )
        preds= str(label_mapping[ pred])
    else:
        label_mapping = {
        0: 'PB',
        1: 'UDH',
        2: 'FEA',
        3: 'ADH',
        4: 'DCIS',
        5: 'IC'
        }

        # Mapear los valores reales a etiquetas
        labels = str(label_mapping[real] )
        preds= str(label_mapping[ pred])
    final=final.append({'Case_id':k,'preds':preds,'real':labels}, ignore_index=True)
  final.to_excel(save_path+i+'_results'+'.xlsx')
  accuracy = accuracy_score( np.array(final['real']), np.array(final['preds']))
  text=i+' accuracy:'+ str(accuracy)
  print(text)