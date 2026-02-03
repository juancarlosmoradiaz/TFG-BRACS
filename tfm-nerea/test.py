import os
import pickle
import argparse
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pytorch_datasets import Dataset, Dataset_full
from train_pred import predict_WSI
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms as transforms
import seaborn as sns

import argparse
# Crear el objeto ArgumentParser y definir los argumentos
parser = argparse.ArgumentParser(description='Configuración para la visualización de resultados')

parser.add_argument('--results_folder_name', type=str, default='resultados_v2',
                    help='Nombre de la carpeta para los resultados')
parser.add_argument('--data_RoI', type=str, default='data_RoI_256.pkl',
                    help='pkl de datasets')
parser.add_argument('--Prob', type=int, default=1,
                    help='Indicador booleano para habilitar o deshabilitar los pesos segun el tamaño de la clase')
parser.add_argument('--normalization', type=str, default=None,
                    help='pkl de datasets')
parser.add_argument('--wsi_level', type=int, default=0,
                    help='Indicador booleano para habilitar o deshabilitar hacer la predicciónd de la imagen completa')
parser.add_argument('--im_size', type=int, default=512,
                    help='Tamaño de la imagen')
parser.add_argument('--n_clases', type=int, default=3,
                    help='Número de clases')

# Parsear los argumentos
args = parser.parse_args()
results_folder_name = args.results_folder_name
data_RoI_pkl=args.data_RoI
Prob=bool(args.Prob)
norm=args.normalization
path_dir='./'
save_path = path_dir+'results/'+results_folder_name+'/'
directorio_actual = os.getcwd()
wsi_level=bool(args.wsi_level)

n = args.im_size
n_clases=args.n_clases

val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
""" Crear readers """
dataReaders = {}

with open(data_RoI_pkl, 'rb') as fp:
    data_RoI = pickle.load(fp)
dataReaders['CNN'] = data_RoI


import warnings
warnings.filterwarnings("ignore")
#crear test dataloader


dataset_test = Dataset(dataReaders['CNN']['test']['x'],
                        dataReaders['CNN']['test']['y'], val_transform, normalization=norm)
    
dataloader_test = DataLoader(dataset_test, batch_size=1,
                                shuffle=False, num_workers=1,
                                pin_memory=True)

dataloaders = { 'test': dataloader_test}
#obtener los tamaños de los datasets
dataset_sizes = {x: len(dataReaders['CNN'][x]['x']) for x in [ 'train', 'val','test']}

os.chdir(save_path) 
# Obtener la lista de archivos en la carpeta actual
archivos = os.listdir()

# Buscar el archivo que cumpla con el criterio
archivo_deseado = None
for archivo in archivos:
    if archivo.startswith("results_Epoch_") and archivo.endswith(".pkl"):
        archivo_deseado = archivo
        break

# Verificar si se encontró el archivo
if archivo_deseado is not None:
    # Abrir el archivo
    with open(archivo_deseado, 'rb') as file:
        results = pickle.load(file)

    # Realizar las operaciones necesarias con los datos del archivo
    # ...
else:
    print("No se encontró ningún archivo que cumpla con el criterio.")

model = results['model']

model.eval()

os.chdir(directorio_actual)
# predecir en el conjunto de pruebas 
test_results = predict_WSI(model, dataloader_test, dataset_sizes['test'])
print('Test acc: {:.4f}\n'.format(test_results['acc']))

# escribir los resultados del dict en un archivo
a_file = open(save_path+'test_results.pkl', "wb")
pickle.dump(test_results, a_file)
a_file.close()

test_labels = np.argmax(dataReaders['CNN']['test']['y'], axis=1)
case_ids_test = dataReaders['CNN']['test']['x']

data = pd.DataFrame()
data['Case_Ids'] = case_ids_test
data['Preds'] = test_results['preds']
data['Real'] = test_labels

data.to_excel(save_path+'test.xlsx')

#Prediccion de las imagenes completas

data = pd.DataFrame()
data['Case_Ids'] = dataReaders['CNN']['test']['x']
data['Preds'] = test_results['preds']
data['Real'] =  test_results['labels']
probs=test_results['probs']
i='test'

if n_clases==3:
            clases=['AT', 'BT', 'MT']
elif n_clases==7:
     clases=['ADH',  'DCIS', 'FEA',  'IC', 'N', 'PB', 'UDH']
else:
     clases=['ADH',  'DCIS', 'FEA',  'IC', 'PB', 'UDH']


if wsi_level:
    excel_file = "BRACS.xlsx"
    df = pd.read_excel(excel_file)
    ids = list(df[df['Set'] == 'Testing']['WSI Filename'])

else:
    ids=[]
    for j in data['Case_Ids']:
        aux='_'.join(j.split('_')[0:-1])
        aux=aux+'_'
        ids.append(aux)
    ids=pd.unique(ids)

final = pd.DataFrame(columns=['Case_id', 'preds', 'real', 'n pred', 'n real'])

# Definir el número de clases (n_clases) y la lista de clases (clases) antes de este punto

prob = np.empty((0, len(clases)))

for k in ids:
    print(k)
    p = data[data['Case_Ids'].str.contains(k)]

    if not p.empty:  # Verificar si p no está vacío
        print(p)
        prob = np.concatenate([prob, probs[p.index].sum(axis=0)], axis=0)

        if Prob:
            m_train = probs[p.index]
            pred = int(np.argmax(m_train.sum(axis=0)))
        else:
            pred = p['Preds'].value_counts().idxmax()
        
        real = int(p['Real'].value_counts().idxmax())
        label_mapping = dict(zip(range(n_clases), clases))
        # Mapear los valores reales a etiquetas
        labels = str(label_mapping[real])
        preds = str(label_mapping[pred])

        final = final.append({'Case_id': k, 'preds': preds, 'real': labels, 'n pred': pred, 'n real': real},
                            ignore_index=True)
    else:
        # Caso en el que p está vacío
        print(f"No se encontraron datos para Case_id: {k}")
final.to_excel(save_path+i+'_results'+'.xlsx')
y_real= np.array(final['real'])
y_pred= np.array(final['preds'])

# Calcular la suma de cada fila
row_sums = np.sum(prob, axis=1)

#   Dividir cada elemento de la fila por la suma de la fila
normalized_matrix = prob / row_sums


y_probs=np.asarray(normalized_matrix)
y_real_num=np.array(final['n real'])


     
accuracy=accuracy_score(y_real, y_pred)
f1_W = f1_score(y_real, y_pred, average='weighted')
f1_micro = f1_score(y_real, y_pred, average='micro')
f1_macro = f1_score(y_real, y_pred, average='macro')
# Calcular el AUC
#auc = roc_auc_score(y_real_num, y_probs, multi_class = 'ovo')

cm=confusion_matrix(y_real, y_pred)
text_acc=i+' accuracy:'+ str(accuracy)
text_f1w=i+' f1 score weighted:'+ str(f1_W)
text_f1mi=i+' f1 score micro:'+ str(f1_micro)
text_f1ma=i+' f1 score macro:'+ str(f1_macro)
#text_auc=i+' AUC:'+ str(auc)
print(text_acc) 
print(text_f1w) 
print(text_f1mi) 
print(text_f1ma) 
#print(text_auc)

print('Matriz de confusión: ')
print(cm)
name='matriz_confusion_test_nclases'+str(n_clases)+'.png'
df_cm = pd.DataFrame(cm, columns=clases, index = clases)
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)#for label size
sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16}, cbar=False)# font size
plt.show()
# Guardar la visualización como un archivo PNG
os.chdir(save_path) 
plt.savefig(name)
