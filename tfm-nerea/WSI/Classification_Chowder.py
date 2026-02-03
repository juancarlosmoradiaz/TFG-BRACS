from chowder import Chowder
import torch
import argparse
import warnings
from copy import deepcopy
import multiprocessing
from datetime import datetime
from utils.pytorch_dataset import Dataset
from IPython.display import clear_output
import numpy as np
from sklearn.model_selection import StratifiedKFold
from utils.trainer import TorchTrainer, slide_level_train_step, slide_level_val_step
import pandas as pd
from utils.functions import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score



parser = argparse.ArgumentParser(description='WSI clasification from a given features')
parser.add_argument('--n_clases', default=3, type=int,
                    help='Número de clases')  
parser.add_argument('--parquet_name', default=None, type=str,
                    help='Nombre del archivo parquet')   
parser.add_argument('--patch_size', default=256, type=int, help='patch size, '
                                                                'default 256')
parser.add_argument('--max_patches_per_slide', default=None, type=int)
parser.add_argument('--lr', type=float, default=3e-3,
                    help='Tasa de aprendizaje (lr)')
parser.add_argument('--epochs', type=int, default=50,
                    help='Número de épocas')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Tamaño del batch')
parser.add_argument('--weight_decay', type=float, default=0.0,
                    help='weight_decay')
parser.add_argument('--mlp_hidden', type=list, default=[200, 100],
                    help='mlp_hidden')
parser.add_argument('--n_top', default=5, type=int,
                    help='Número de instancias top')  
parser.add_argument('--n_bottom', default=5, type=int,
                    help='Número de intancias de evidencia negativa')  

args = parser.parse_args()
n_clases=args.n_clases
parquet_name=args.parquet_name
patch_size=args.patch_size
max_patches_per_slide=args.max_patches_per_slide
batch_size=args.batch_size                        
epochs=args.epochs                         
lr=args.lr                    
weight_decay=args.weight_decay
mlp_hidden=args.mlp_hidden
n_top=args.n_top,                             # number of top scores in Chowder (in the image, N is 2)
n_bottom=args.n_bottom

datasets_names=['train','val','test']

patch_folder_WSI='_patches_max'+str(max_patches_per_slide)+'_size'+str(patch_size)
dataReaders={}
datasets={}
# Leer datasets guardados en .parcket
for i in datasets_names:
    # Ruta al archivo Parquet
    if parquet_name is not None:
        ruta_archivo_parquet = './features_datasets/'+i+'/'+parquet_name
    else:
        ruta_archivo_parquet = './features_datasets/'+i+'/'+patch_folder_WSI+'_features_'+i+'.parquet'
    # Leer el archivo Parquet como un DataFrame
    dataReaders[i] = pd.read_parquet(ruta_archivo_parquet)
    datasets[i]=Dataset(dataReaders[i]['Features'].apply(np.vstack), dataReaders[i]['label_ohe'])


chowder = Chowder(
    in_features=768,                     # output dimension of Phikon
    out_features=n_clases,                      # dimension of predictions (a probability for class "1")
    n_top=n_top,                             # number of top scores in Chowder (in the image, N is 2)
    n_bottom=n_bottom,                          # number of bottom scores in Chowder
    mlp_hidden=mlp_hidden,               # MLP hidden layers after the max-min layer
    mlp_activation=torch.nn.Softmax(dim=1),   # MLP activation
    bias=True                            # bias for first 1D convolution which computes scores
)



# We define the loss function, optimizer and metrics for the training
criterion = torch.nn.CrossEntropyLoss()  # Binary Cross-Entropy Loss
optimizer = torch.optim.AdamW              # Adam optimizer
metrics = {"auc": auc}                    # AUC will be the tracking metric


# ``collator`` is a function that apply a deterministic
# transformation to a batch of samples before being processed
# by the GPU. Here, this function is ``pad_collate_fn``. The
# goal of this function is align matrices of features (the inputs)
# in terms of shape. Indeed, some WSI may have 200 features (very
# small piece of tissues) or 1,000 (the maximum we set). In that case,
# all matrices will have a shape of at most the bigger matrices in the
# batch. Our (200, 768) input matrix will become a (1000, 768) matrix
# with 800 ``inf`` values. A boolean mask is stored so that to tell
# torch not to process these 800 values but only focus on the 200 real ones

collator = pad_collate_fn

train_metrics, val_metrics = [], []
test_logits = []



batch_size=3                      
epochs=20                      
lr=3e-3                    
weight_decay=0.0
trainer = TorchTrainer(
            model=deepcopy(chowder),
            criterion=criterion,
            metrics=metrics,
            batch_size=batch_size,                           # you can tweak this
            num_epochs=epochs,                           # you can tweak this
            learning_rate=lr,                      # you can tweak this
            weight_decay=weight_decay,                        # you can tweak this
            device="cuda:0",
            num_workers=multiprocessing.cpu_count(), # you can tweak this
            optimizer=deepcopy(optimizer),
            train_step=slide_level_train_step,
            val_step=slide_level_val_step,
            collator=pad_collate_fn,
        )

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    # Training step for the given number of epochs
    train_metrics, val_metrics = trainer.train(
        datasets['train'], datasets['val']
    )
    # Predictions on test (logits, sigmoid(logits) = probability)
    test_logits = trainer.predict(datasets['test'])[1]
    print()

    # Aplicar softmax a los logits
    labels_test=torch.tensor(datasets['test'].labels).numpy()
    probs_test = np.exp(test_logits) / np.sum(np.exp(test_logits), axis=1, keepdims=True)

    test_auc = roc_auc_score(labels_test, probs_test, multi_class='ovr')

    test_auc




    # Obtener las etiquetas binarias
    labels_bin = np.argmax(labels_test, axis=1)

    # Obtener las predicciones a partir de las probabilidades
    predictions = np.argmax(probs_test, axis=1)

    # Calcular Accuracy
    accuracy = accuracy_score(labels_bin, predictions)

    # Calcular Precision
    precision = precision_score(labels_bin, predictions, average='weighted')

    # Calcular Recall
    recall = recall_score(labels_bin, predictions, average='weighted')

    # Calcular F1 Score
    f1 = f1_score(labels_bin, predictions, average='weighted')

    # Calcular AUC ROC
    auc_roc = roc_auc_score(labels_test, probs_test, multi_class='ovr')

    # Calcular Confusion Matrix
    conf_matrix = confusion_matrix(labels_bin, predictions)

    # Imprimir los resultados
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("AUC ROC:", auc_roc)
    print("Confusion Matrix:\n", conf_matrix)
