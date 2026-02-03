from cv2 import imread, imwrite
from pathlib import Path
from tqdm import tqdm
import numpy as np 
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import glob
import numpy as np
from PIL import Image
import argparse

# Crear el objeto ArgumentParser y definir los argumentos
parser = argparse.ArgumentParser(description='Configuración para la creación de patches')

parser.add_argument('--patch_size', type=int, default=512,
                    help='Tamaño del patch')
parser.add_argument('--overlap', type=float, default=0,
                    help='Porcentaje de superposición')

# Parsear los argumentos
args = parser.parse_args()

# Acceder a los valores de los argumentos
patch_size = args.patch_size
overlap = args.overlap
folder_name='_RoI_patches_overlapping_'+str(patch_size)


datasets = ['test', 'val']
clases_roi = pd.Series(['0_N', '1_PB', '2_UDH', '3_FEA', '4_ADH', '5_DCIS', '6_IC'])

files_RoI_train = []
files_RoI=[]
for i in datasets:
    paths_RoI = './BRACS_RoI/latest_version/' + i + '/' + clases_roi + '/'
    for j in range(7):
        aux = glob.glob(paths_RoI[j] + '*.png')
        files_RoI += aux

paths_RoI = './BRACS_RoI/latest_version/train/' + clases_roi + '/'
for j in range(7):
    aux = glob.glob(paths_RoI[j] + '*.png')
    files_RoI_train += aux

patch_size = patch_size
thr = 270
overlap_size = int(patch_size * overlap)  # Tamaño de superposición

for filename in tqdm(files_RoI_train):
    f = str(filename)
    save_name = f.split('/')[-1].split('.')[0]
    save_path = f.split('_')[0] + folder_name + '/' + '/'.join(f.split('/')[3:-1])

    # Verificar si el directorio ya existe
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    im = Image.open(f)
    nobgr_img_blocks = []
    # Redimensionar la imagen si no cumple con el tamaño mínimo del parche
    if im.width < patch_size or im.height < patch_size:
        im = im.resize((patch_size, patch_size))

    # Iterar para superponer los parches
    for j in range(0, im.width - patch_size + 1, patch_size - overlap_size):
        for i in range(0, im.height - patch_size + 1, patch_size - overlap_size):
            block = np.array(im.crop((j, i, j + patch_size, i + patch_size)))
            if block.shape == (patch_size, patch_size, 3):
                gray_block = np.array(Image.fromarray(block).convert('L'))  # Convertir a escala de grises
                # Aplicar nuestro detector de desenfoque utilizando FFT
                if (
                    np.mean(block[:, :, 0]) < 220.0
                    and np.mean(block[:, :, 1]) < 220.0
                    and np.mean(block[:, :, 2]) < 220.0
                    and np.min(gray_block) > 20
                ):
                    nobgr_img_blocks.append(block)

    if len(nobgr_img_blocks) < 1:
        block = np.array(im.resize((patch_size, patch_size)))
        nobgr_img_blocks.append(block)
    for index, block in enumerate(nobgr_img_blocks):
        save_filename = f'{save_path}/{save_name}_{index}.jpeg'
        if not os.path.isfile(save_filename):
            Image.fromarray(block).save(save_filename)
        else:
            print('already converted')

for filename in tqdm(files_RoI):
    f = str(filename)
    save_name = f.split('/')[-1].split('.')[0]
    save_path = f.split('_')[0] + folder_name + '/' + '/'.join(f.split('/')[3:-1])

    # Verificar si el directorio ya existe
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    im = Image.open(f)
    nobgr_img_blocks = []
    # Redimensionar la imagen si no cumple con el tamaño mínimo del parche
    if im.width < patch_size or im.height < patch_size:
        im = im.resize((patch_size, patch_size))
    for j in range(0, im.width, patch_size):
        for i in range(0, im.height, patch_size):
            block = np.array(im.crop((j, i, j + patch_size, i + patch_size)))
            if block.shape == (patch_size, patch_size, 3):
                gray_block = np.array(Image.fromarray(block).convert('L'))  # Convertir a escala de grises
                # Aplicar nuestro detector de desenfoque utilizando FFT
                if (
                    np.mean(block[:, :, 0]) < 220.0
                    and np.mean(block[:, :, 1]) < 220.0
                    and np.mean(block[:, :, 2]) < 220.0
                    and np.min(gray_block) > 20
                ):
                    nobgr_img_blocks.append(block)

   
    if len(nobgr_img_blocks) < 1:
        block = np.array(im.resize((patch_size, patch_size)))
        nobgr_img_blocks.append(block)
    for index, block in enumerate(nobgr_img_blocks):
        save_filename = f'{save_path}/{save_name}_{index}.jpeg'
        if not os.path.isfile(save_filename):
            Image.fromarray(block).save(save_filename)
        else:
            print('already converted')