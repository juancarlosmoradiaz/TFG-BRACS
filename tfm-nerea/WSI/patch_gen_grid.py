import pandas as pd
import numpy as np
from openslide import OpenSlide
from multiprocessing import Pool, Value, Lock
import os
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from skimage.io import imsave, imread
from skimage.exposure.exposure import is_low_contrast
from utils.functions import *
from skimage.transform import resize
from scipy.ndimage import binary_dilation, binary_erosion
import argparse
import logging
import pickle
import warnings
warnings.filterwarnings("ignore")

from itertools import tee
from PIL import Image

import h5py
from tqdm import tqdm

import pickle
import re
import pdb
import pandas as pd

def get_mask_image(img_RGB, RGB_min=50):
    img_HSV = rgb2hsv(img_RGB)

    background_R = img_RGB[:, :, 0] > threshold_otsu(img_RGB[:, :, 0])
    background_G = img_RGB[:, :, 1] > threshold_otsu(img_RGB[:, :, 1])
    background_B = img_RGB[:, :, 2] > threshold_otsu(img_RGB[:, :, 2])
    tissue_RGB = np.logical_not(background_R & background_G & background_B)
    tissue_S = img_HSV[:, :, 1] > threshold_otsu(img_HSV[:, :, 1])
    min_R = img_RGB[:, :, 0] > RGB_min
    min_G = img_RGB[:, :, 1] > RGB_min
    min_B = img_RGB[:, :, 2] > RGB_min

    mask = tissue_S & tissue_RGB & min_R & min_G & min_B
    return mask

def get_mask(slide, level='max', RGB_min=50):
    #read svs image at a certain level  and compute the otsu mask
    if level == 'max':
        level = len(slide.level_dimensions) - 1
    # note the shape of img_RGB is the transpose of slide.level_dimensions
    img_RGB = np.transpose(np.array(slide.read_region((0, 0),level,slide.level_dimensions[level]).convert('RGB')),
                           axes=[1, 0, 2])

    tissue_mask = get_mask_image(img_RGB, RGB_min)
    return tissue_mask, level


def extract_patches(slide_path, mask_path, patch_size, patches_output_dir, slide_id, max_patches_per_slide=2000):
    patch_folder = os.path.join(patches_output_dir, slide_id)
    
    if not os.path.isdir(patch_folder):
        os.makedirs(patch_folder)
    else:
        # Check if the directory is empty
        if not os.listdir(patch_folder):
            # The directory exists, but it's empty. Continue with the code.
            pass
        else:
            # The directory exists and contains files. Return from the function.
            return
        
    slide = OpenSlide(slide_path)

    path_hdf5 = os.path.join(patch_folder, f"{slide_id}.hdf5")
    hdf = h5py.File(path_hdf5, 'w')

    patch_folder_mask = os.path.join(mask_path, slide_id)
    if not os.path.isdir(patch_folder_mask):
        os.makedirs(patch_folder_mask)
        mask, mask_level = get_mask(slide)
        mask = binary_dilation(mask, iterations=3)
        mask = binary_erosion(mask, iterations=3)
        np.save(os.path.join(patch_folder_mask, "mask.npy"), mask) 
    else:
        mask = np.load(os.path.join(mask_path, slide_id, 'mask.npy'))
        
    mask_level = len(slide.level_dimensions) - 1
    

    PATCH_LEVEL = 0
    BACKGROUND_THRESHOLD = .2

    try:
        #with open(os.path.join(patch_folder, 'loc.txt'), 'w') as loc:
        #loc.write("slide_id {0}\n".format(slide_id))
        #loc.write("id x y patch_level patch_size_read patch_size_output\n")

        ratio_x = slide.level_dimensions[PATCH_LEVEL][0] / slide.level_dimensions[mask_level][0]
        ratio_y = slide.level_dimensions[PATCH_LEVEL][1] / slide.level_dimensions[mask_level][1]

        xmax, ymax = slide.level_dimensions[PATCH_LEVEL]
        #patch_size_resized = patch_size
        # handle slides with 40 magnification at base level
        resize_factor = float(slide.properties.get('aperio.AppMag', 20)) / 20.0
        if not slide.properties.get('aperio.AppMag', 20): print(f"magnifications for {slide_id} is not found, using default magnificantion 20X")
        
        resize_factor = resize_factor * args.dezoom_factor
        patch_size_resized = (int(resize_factor * patch_size[0]), int(resize_factor * patch_size[1]))
        print(f"patch size for {slide_id}: {patch_size_resized}")
        i = 0

        indices = [(x, y) for x in range(0, xmax, patch_size_resized[0]) for y in
                       range(0, ymax, patch_size_resized[0])]
        
        # here, we generate all the pathes with valid mask
        if max_patches_per_slide is None:
            max_patches_per_slide = len(indices)
        np.random.seed(5)
        np.random.shuffle(indices)

        for x, y in indices:
            # check if in background mask
            x_mask = int(x / ratio_x)
            y_mask = int(y / ratio_y)
            if mask[x_mask, y_mask] == 1:
                patch = slide.read_region((x, y), PATCH_LEVEL, patch_size_resized).convert('RGB')
                try:
                    mask_patch = get_mask_image(np.array(patch))
                    mask_patch = binary_dilation(mask_patch, iterations=3)
                except Exception as e:
                    print("error with slide id {} patch {}".format(slide_id, i))
                    print(e)
                    '''
                if (mask_patch.sum() > BACKGROUND_THRESHOLD * mask_patch.size) and not (is_low_contrast(patch)):
                    #loc.write("{0} {1} {2} {3} {4} {5}\n".format(i, x, y, PATCH_LEVEL, patch_size_resized[0],
                    #                                                 patch_size_resized[1]))
                    imsave(os.path.join(patch_folder, "{0}_patch_{1}.jpeg".format(slide_id, i)), np.array(patch))
                    i += 1
                    '''
                if (mask_patch.sum() > BACKGROUND_THRESHOLD * mask_patch.size) and not (is_low_contrast(patch)):
                    if resize_factor != 1.0:
                        patch = patch.resize(patch_size)
                    patch = np.array(patch)
                    tile_name = f"{slide_id}_patch_{i}_{x}_{y}"
                    hdf.create_dataset(tile_name, data=patch)
                    i = i + 1
                    
            if i >= max_patches_per_slide:
                break
        hdf.close()
        if i == 0:
            print("no patch extracted for slide {}".format(slide_id))
        else:
            with open(os.path.join(patch_folder, "complete.txt"), 'w') as f:
                f.write('Process complete!\n')
                f.write(f"Total n patch = {i}")
                print(f"{slide_id} complete, total n patch = {i}")


    except Exception as e:
        print("error with slide id {} patch {}".format(slide_id, i))
        print(e)

def get_slide_id(slide_name):
    return slide_name.split('.')[0]+'.'+slide_name.split('.')[1]


def process(opts):
    # global lock
    slide_path, patch_size, patches_output_dir, mask_path, slide_id, max_patches_per_slide = opts
   # for slide_path, patches_output_dir, mask_path, slide_id in zip(slide_list, patches_output_dir_list, mask_path_list, slide_id_list):
    extract_patches(slide_path, mask_path, patch_size,
                    patches_output_dir, slide_id, max_patches_per_slide)


parser = argparse.ArgumentParser(description='Generate patches from a given folder of images')
parser.add_argument('--patch_size', default=768, type=int, help='patch size, '
                                                                'default 768')
parser.add_argument('--max_patches_per_slide', default=None, type=int)
parser.add_argument('--num_process', default=10, type=int,
                    help='number of mutli-process, default 10')         

parser.add_argument('--dezoom_factor', default=2.0, type=float,
                    help='dezoom  factor, 1.0 means the images are taken at 20x magnification, 2.0 means the images are taken at 40x magnification')
parser.add_argument('--debug', default=0, type=int,
                    help='whether to use debug mode')
parser.add_argument('--parallel', default=1, type=int,
                    help='whether to use parallel computation')
parser.add_argument('--use_ext', default=0, type=int,
                    help='Directorio externo')
parser.add_argument('--ext_dir', default='/media/pc-i1-2/EXTERNAL_USB', type=str,
                    help='Nombre del disco duro externo')   



if __name__ == '__main__':
    # count = Value('i', 0)
    # lock = Lock()

    args = parser.parse_args()
    use_ext=bool(args.use_ext)
    ext_dir=args.ext_dir
    ##DEBUG
    import pandas as pd
    # Ruta del archivo Excel
    excel_file = "BRACS.xlsx"
    patch_folder_WSI='_patches_max'+str(args.max_patches_per_slide)+'_size'+str(args.patch_size)
    # Leer el archivo Excel
    df = pd.read_excel(excel_file)
    AT = ['FEA', 'ADH']
    BT = ['N', 'PB', 'UDH']
    MT = ['DCIS', 'IC']
    label_mapping = {'AT': AT, 'BT': BT, 'MT': MT}
    df['group'] = [next(key for key, value in label_mapping.items() if elemento in value) for elemento in df['WSI label']]
    #print("DEBUGING SMALL SLIDE LIST")
    #slide_list = ['GTEX-14A5I-0925.svs','GTEX-14A6H-0525.svs'
    #          ]
    # Aplicar la función a las filas del DataFrame y crear una nueva columna 'Nombre completo'
    datasets=['train','test','val']
    if use_ext:
        dir=ext_dir
    else:
        dir='.'
    for i in datasets:
        if i=='train':
            set='Training'
        elif i=='test':
            set='Testing'
        else:
            set='Validation'
        print(i, set)
        df_aux=pd.DataFrame()
        df_aux=df[df['Set']==set]
        df_aux['path'] = df_aux.apply(lambda row: build_path(row, prefix='', mode=i, extension='svs', wsi=True), axis=1)
        df_aux['path_patch'] = df_aux.apply(lambda row: build_path(row, prefix=patch_folder_WSI, mode=i, extension=None, dir=dir), axis=1)
        df_aux['mask_patch'] = df_aux.apply(lambda row: build_path(row, prefix='_mask', mode=i, extension=None, dir=dir), axis=1)
        print(df_aux.head)
        slide_list=list(df_aux['path'])
        slide_id=list(df_aux['WSI Filename'])
        patch_path=list(df_aux['path_patch'])
        mask_path=list(df_aux['mask_patch'])
        opts = [
            (slide, (args.patch_size, args.patch_size), patch, mask, slide_id, args.max_patches_per_slide)
            for slide, patch, mask, slide_id in zip(slide_list, patch_path, mask_path, slide_id)
        ]
        #pool = Pool(processes=args.num_process)
        #pool.map(process, opts)

    
        for opt in tqdm(opts):
            process(opt)
    '''
        from tqdm import tqdm
    for opt in tqdm(opts):
        process(opt)
    '''
