import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import h5py
import torch.nn as nn
from ctran import ctranspath
import pandas as pd
from torchvision import transforms
import torchvision.transforms as transforms
import warnings
from utils import *
warnings.filterwarnings("ignore")
from sklearn import preprocessing

parser = argparse.ArgumentParser(description='Generate features from a given patches')
parser.add_argument('--use_ext', default=0, type=int,
                    help='Directorio externo')
parser.add_argument('--ext_dir', default='', type=str,
                    help='Nombre del disco duro externo')   
parser.add_argument('--n_clases', default=3, type=int,
                    help='Número de clases')
parser.add_argument('--parquet_name', default='', type=str,
                    help='Nombre del archivo parquet')   
parser.add_argument('--patch_size', default=768, type=int, help='patch size, '
                                                                'default 768')
parser.add_argument('--max_patches_per_slide', default=None, type=int)


args = parser.parse_args()
use_ext=bool(args.use_ext)
ext_dir=args.ext_dir
n_clases=args.n_clases
parquet_name=args.parquet_name
patch_size=args.patch_size
max_patches_per_slide=args.max_patches_per_slide
if use_ext:
    dir=ext_dir
else:
    dir='.'
#el código dentro de este bloque solo se ejecutará si el script es ejecutado directamente y no si es importado como un módulo en otro script
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    transforms_val = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
    )
    model = ctranspath()
    model.head = nn.Identity()
    td = torch.load(r'./ctranspath.pth')
    model.load_state_dict(td['model'], strict=True)
    model = model.to(device)
    model.eval()
    print('Loading dataset...')

    
    # Ruta del archivo Excel
    excel_file = "BRACS.xlsx"
    patch_folder_WSI='_patches_max'+str(max_patches_per_slide)+'_size'+str(patch_size)
    # Leer el archivo Excel
    df = pd.read_excel(excel_file)
    AT = ['FEA', 'ADH']
    BT = ['N', 'PB', 'UDH']
    MT = ['DCIS', 'IC']
    label_mapping = {'AT': AT, 'BT': BT, 'MT': MT}
    df['group'] = [next(key for key, value in label_mapping.items() if elemento in value) for elemento in df['WSI label']]
    clases3 = ['AT', 'BT', 'MT']
    clases7 = ['N', 'PB', 'UDH', 'FEA', 'ADH', 'DCIS', 'IC']
    # Inicializar el codificador
    encoder3 = preprocessing.LabelEncoder()
    encoder7 = preprocessing.LabelEncoder()

    # Ajustar y transformar las etiquetas
    encoder3.fit(clases3)
    encoder7.fit(clases7)

    #Codificacion ohe para las etiquetas
    ohe = preprocessing.OneHotEncoder(sparse_output=False)

    if n_clases==3:
        classes = np.array(clases3)
        ohe.fit(classes.reshape(-1, 1))
    else:
        classes = np.array(clases7)
        ohe.fit(classes.reshape(-1, 1))

    # Imprimir la correspondencia completa
    print("Correspondencia completa:")
    print(dict(zip(encoder3.classes_, range(len(encoder3.classes_)))))
    print(dict(zip(encoder7.classes_, range(len(encoder7.classes_)))))
    
    datasets=['train']
    # Aplicar la función a las filas del DataFrame y crear una nueva columna 'Nombre completo'
    for i in datasets:
        if i=='train':
            set='Training'
        elif i=='test':
            set='Testing'
        else:
            set='Validation'

        df_aux=pd.DataFrame()
        df_aux=df[df['Set']==set]
        # Eliminar la columna temporal utilizada para verificar la existencia
        
        df_aux['path'] = df_aux.apply(lambda row: build_path(row, prefix='', mode=i, extension='svs', wsi=True), axis=1)
        # Agregar una nueva columna al DataFrame para verificar la existencia
        df_aux['Archivo_Existe'] = df_aux['path'].apply(lambda x: os.path.exists(os.path.join('./', x)))
        df_aux = df_aux[df_aux['Archivo_Existe']]
        df_aux = df_aux.drop(columns=['Archivo_Existe'])
        df_aux['path_patch'] = df_aux.apply(lambda row: build_path(row, prefix=patch_folder_WSI, mode=i, extension=None, dir=dir), axis=1)
        df_aux['path_features'] = df_aux.apply(lambda row: build_path(row, prefix='_features', mode=i, extension=None, dir=dir), axis=1)
        df_aux['path_hdf5'] =  df_aux.apply(lambda row: build_path(row, prefix=patch_folder_WSI, mode=i, extension='hdf5', dir=dir), axis=1)
        print(f'Number of slides = {df_aux.shape[0]}')
        
        print(f'Number of slides = {df_aux.shape[0]}')
        # Crear un DataFrame vacío
        columns = ['WSI Filename', 'Features', 'label','true_label']
        df_dataset = pd.DataFrame(columns=columns)
        parquet_dataset='./features_datasets/'+i+'/'+patch_folder_WSI+'_features_'+i+'.parquet'
        print(df_aux['path'])
        print('Existen', sum([os.path.exists(ruta) for ruta in df_aux['path']]), 'imagenes')
        for j, row in tqdm(df_aux.iterrows()):
            WSI = row['WSI Filename']
            path_patch=row['path_patch']
            path_h5=row['path_hdf5']
            path_features=row['path_features']
            if n_clases==3:
                label=row['group']
                label_encoder=encoder3.transform([label])[0]
            else:
                label=row['WSI label']
                label_encoder=encoder7.transform([label])[0]
            if not os.path.exists(path_features):
                os.makedirs(path_features)

            if os.path.exists(os.path.join(path_features, "complete_features.txt")):
                print(f'{WSI}: Features already obtained')
                with h5py.File(os.path.join(path_features,  WSI+'.h5'), 'r') as ftrs:
                    features = ftrs['features'][:]
                    new_row = pd.DataFrame([[WSI, features.tolist(),label_encoder, label]], columns=columns)
                    df_dataset = pd.concat([df_dataset, new_row], ignore_index=True)

                continue

            if os.path.exists(path_h5):
                print(f'{path_h5}: H5 exist')

            try:
                with h5py.File(path_h5, 'r') as f_read:
                    keys = list(f_read.keys())

                    features_tiles = []
                    for key in tqdm(keys):
                        image = f_read[key][:]
                        image = apply_transformations(image,transforms_val)
                        with torch.no_grad():
                            features = model(image[None,:])
                            features_tiles.append(features[0].detach().cpu().numpy())

                features_tiles = np.asarray(features_tiles)
                n_tiles = len(features_tiles)

                f_write = h5py.File(os.path.join(path_features, WSI+'.h5'), "w")
                dset = f_write.create_dataset("features", data=features_tiles)
                f_write.close()

                with open(os.path.join(path_features, "complete_features.txt"), 'w') as f_sum:
                    f_sum.write(f"Total n patch = {n_tiles}")
                new_row = pd.DataFrame([[WSI, features_tiles.tolist(),label_encoder, label]], columns=columns)
                df_dataset = pd.concat([df_dataset, new_row], ignore_index=True)
                

            except Exception as e:
                print('Exception in:', WSI)
                continue
        print(df_dataset['true_label'])
        df_dataset['label_ohe'] = ohe.transform(np.asarray(df_dataset['true_label']).reshape(-1, 1)).tolist()
        df_dataset.to_parquet(parquet_dataset)