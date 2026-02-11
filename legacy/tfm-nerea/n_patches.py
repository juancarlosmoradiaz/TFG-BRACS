
import pickle
import argparse
import pandas as pd


# Crear el objeto ArgumentParser y definir los argumentos
parser = argparse.ArgumentParser(description='Configuración para la visualización de resultados')

parser.add_argument('--data_RoI', type=str, default='data_RoI512.pkl',
                    help='pkl de datasets')
parser.add_argument('--patch_size', type=int, default=512,
                    help='Tamaño del patch')
# Parsear los argumentos
args = parser.parse_args()

data_RoI_pkl=args.data_RoI
patch_size=args.patch_size
import warnings
warnings.filterwarnings("ignore")

""" Create readers """
dataReaders = {}

with open(data_RoI_pkl, 'rb') as fp:
    data_RoI = pickle.load(fp)
dataReaders['CNN'] = data_RoI


for i in ['val', 'train', 'test']:
    data = pd.DataFrame() 
    if i=="train":
        data['Case_Ids'] = dataReaders['CNN']['train']['x']
    elif  i=="val":
        data['Case_Ids'] = dataReaders['CNN']['val']['x']
    else: 
        data['Case_Ids'] = dataReaders['CNN']['test']['x']
    ids=[]
    for j in data['Case_Ids']:
        aux='_'.join(j.split('_')[0:-1])
        aux=aux+'_'
        ids.append(aux)
    ids=pd.unique(ids)
    final=pd.DataFrame(columns=['Case_id','# patches'])

    for k in ids:
        p = data[data['Case_Ids'].str.contains(k)]
        n=len(p)
        final=final.append({'Case_id':k,'# patches':n}, ignore_index=True)
    final.to_excel('./'+i+str(patch_size)+'n_patches'+'.xlsx')
    value_counts = final['# patches'].value_counts()
    print(i,' ',str(patch_size),' ','n_patches: ',value_counts)