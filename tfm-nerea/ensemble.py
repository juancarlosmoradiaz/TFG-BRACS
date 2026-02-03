import os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# Directorio raíz
root_directory = './results'

# Patrón para buscar carpetas
folder_pattern = '7clases'

# Nombre del archivo a buscar
file_name = 'test_results.xlsx'
save_path=root_directory+'/resultados_ensemble'
os.makedirs(save_path, exist_ok=True)
# DataFrame base
base_df = None

# Recorrer las carpetas que coinciden con el patrón
for folder_name in os.listdir(root_directory):
    folder_path = os.path.join(root_directory, folder_name)
    if os.path.isdir(folder_path) and folder_pattern in folder_name:
        file_path = os.path.join(folder_path, file_name)
        if os.path.exists(file_path):
            df = pd.read_excel(file_path)
            text=file_path.split('/')[-2].split('_')[-1]
            if base_df is None:
                base_df = df[['Case_id', 'real','preds']]
            else:
                base_df = pd.merge(base_df, df[['Case_id','preds']], on='Case_id', suffixes=('', text))



# Obtener las columnas que contienen 'preds' en su nombre
preds_columns = [col for col in base_df.columns if 'preds' in col]

# Obtener el valor más repetido de las columnas 'preds'
base_df['preds_ensemble'] = base_df[preds_columns].mode(axis=1)[0]

os.chdir(save_path) 
base_df.to_excel('test_result_ensemble'+'.xlsx')

y_real=np.array(base_df['real'])
y_pred=np.array(base_df['preds_ensemble'])

i='test'
accuracy=accuracy_score(y_real, y_pred)
f1_W = f1_score(y_real, y_pred, average='weighted')
f1_micro = f1_score(y_real, y_pred, average='micro')
f1_macro = f1_score(y_real, y_pred, average='macro')
# Calcular el AUC
#auc = roc_auc_score(y_real_num, y_probs, multi_class = 'ovo')
clases=['ADH',  'DCIS', 'FEA',  'IC', 'N', 'PB', 'UDH']

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
name='matriz_confusion_test_nclases'+'.png'
df_cm = pd.DataFrame(cm, columns=clases, index = clases)
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)#for label size
sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16}, cbar=False)# font size
plt.show()
# Guardar la visualización como un archivo PNG

plt.savefig(name)