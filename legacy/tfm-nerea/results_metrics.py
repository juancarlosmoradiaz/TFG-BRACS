import os
import pickle
import argparse
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Crear el objeto ArgumentParser y definir los argumentos
parser = argparse.ArgumentParser(description='Configuración para la visualización de resultados')

parser.add_argument('--results_folder_name', type=str, default='resultados_v2',
                    help='Nombre de la carpeta para los resultados')
parser.add_argument('--Prob', type=int, default=1,
                    help='Indicador booleano para habilitar o deshabilitar los pesos segun el tamaño de la clase')
parser.add_argument('--data_RoI', type=str, default='data_RoI_512.pkl',
                    help='pkl de datasets')
parser.add_argument('--full', type=int, default=0,
                    help='Indicador booleano para habilitar o deshabilitar tratar con full imagenes')
parser.add_argument('--n_clases', type=int, default=3,
                    help='Número de clases')


# Parsear los argumentos
args = parser.parse_args()
results_folder_name = args.results_folder_name
Prob=bool(args.Prob)
data_RoI_pkl=args.data_RoI
full=bool(args.full)
n_clases=args.n_clases

import warnings
warnings.filterwarnings("ignore")

""" Create readers """
dataReaders = {}

with open(data_RoI_pkl, 'rb') as fp:
    data_RoI = pickle.load(fp)
dataReaders['CNN'] = data_RoI

path_dir='./'
save_path = path_dir+'results/'+results_folder_name+'/'
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

if n_clases==3:
            clases=['AT', 'BT', 'MT']
else:
            clases=['ADH',  'DCIS', 'FEA',  'IC', 'N', 'PB', 'UDH']

for i in ['val', 'train']:
  data = pd.DataFrame() 
  if i=="train":
    data['Case_Ids'] = results['train_case_ids']
    data['Preds'] = results['train_preds']
    data['Real'] =  results['train_labels']
    probs=results['train_probs']

  else:
    data['Case_Ids'] = dataReaders['CNN']['val']['x']
    data['Preds'] = results['val_preds']
    data['Real'] =  results['val_labels']
    probs=results['val_probs']
  
  if full:
    y_real=np.array(data['Real'])
    y_pred=np.array(data['Preds'])
    y_probs=np.asarray(probs)
    y_real_num=y_real
  else:
      ids=[]
      for j in data['Case_Ids']:
          aux='_'.join(j.split('_')[0:-1])
          aux=aux+'_'
          ids.append(aux)
      ids=pd.unique(ids)
      final=pd.DataFrame(columns=['Case_id','preds','real', 'n pred', 'n real'])
      
      prob=np.empty((0, len(clases)))

      for k in ids:
          p = data[data['Case_Ids'].str.contains(k)]
          prob = np.concatenate([prob, probs[p.index].sum(axis=0)/len(p)], axis=0)
          if Prob:
              m_train=probs[p.index]
              pred=np.argmax(m_train.sum(axis=0))

          else: 
              pred=p['Preds'].value_counts().idxmax()
          real=p['Real'].value_counts().idxmax()

          label_mapping = dict(zip(range(n_clases), clases))
          # Mapear los valores reales a etiquetas
          labels = str(label_mapping[real])
          preds= str(label_mapping[pred])

          final=final.append({'Case_id':k,'preds':preds,'real':labels, 'n pred':pred, 'n real':real}, ignore_index=True)
      final.to_excel(i+'_results'+'.xlsx')
      y_real= np.array(final['real'])
      y_pred= np.array(final['preds'])
      y_probs=np.asarray(prob)
      y_real_num=np.array(final['n real'])
  accuracy = accuracy_score(y_real, y_pred)

  f1_W = f1_score(y_real, y_pred, average='weighted')
  f1_micro = f1_score(y_real, y_pred, average='micro')
  f1_macro = f1_score(y_real, y_pred, average='macro')
  # Calcular el AUC
  #auc = roc_auc_score(y_real_num, y_probs)

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
  name='matriz_confusion_test_nclases_'+i+'.png'
  df_cm = pd.DataFrame(cm, columns=clases, index = clases)
  df_cm.index.name = 'Actual'
  df_cm.columns.name = 'Predicted'
  plt.figure(figsize = (10,7))
  sns.set(font_scale=1.4)#for label size
  sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16}, cbar=False)# font size
  plt.show()
  # Guardar la visualización como un archivo PNG
  plt.savefig(name)


