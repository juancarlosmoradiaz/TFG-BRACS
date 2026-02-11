import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from sklearn import preprocessing
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms
from cv2 import imread
import pandas as pd
import slideflow as sf
import os
from typing import Optional
import random
from transformers import set_seed as set_seed_hf
from transformers import AutoImageProcessor
import torchvision.transforms as transforms
from torchvision.models import resnet50,resnet18, vgg16, ResNet50_Weights, ResNet18_Weights,VGG16_Weights
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
import os
from typing import Optional
from typing import Any
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from transformers import AutoModelForImageClassification
import warnings
from copy import deepcopy
from peft import LoraConfig, get_peft_model
import evaluate
from transformers import TrainingArguments, Trainer
from transformers.utils import logging
from pathlib import Path
working_directory = Path(".").resolve()
cache_dir = working_directory / "cache"

# Crear el objeto ArgumentParser y definir los argumentos
parser = argparse.ArgumentParser(description='Configuración para el entrenamiento del modelo')

parser.add_argument('--lr', type=float, default=3e-3,
                    help='Tasa de aprendizaje (lr)')
parser.add_argument('--epochs', type=int, default=50,
                    help='Número de épocas')
parser.add_argument('--results_folder_name', type=str, default='resultados_v2',
                    help='Nombre de la carpeta para los resultados')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Tamaño del batch')
parser.add_argument('--data_RoI', type=str, default='data_RoI_256.pkl',
                    help='pkl de datasets')
parser.add_argument('--normalization', type=str, default=None,
                    help='tipo de normalización')
parser.add_argument('--n_clases', type=int, default=3,
                    help='Número de clases')
parser.add_argument('--lora', type=int, default=1,
                    help='Indicador booleano para habilitar o deshabilitar el fine tunning con LoRA')


# Parsear los argumentos
args = parser.parse_args()

# Acceder a los valores de los argumentos
batch_size = args.batch_size
lr = args.lr
num_epochs = args.epochs
results_folder_name = args.results_folder_name
path_dir='./'
data_RoI=args.data_RoI
model_name='modelo_TCGA'+data_RoI
norm=args.normalization
n_clases=args.n_clases
lora=bool(args.lora)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

device = torch.device("cuda:0" if torch.cuda.is_available()
                                   else "cpu") 
torch.cuda.empty_cache()



class Dataset(data.Dataset):
    def __init__(self, inputs, labels, transform=None, normalization=None):
        """
        Inicialización del conjunto de datos.

        Args:
            inputs (list): Lista de rutas de archivos de entrada.
            labels (list): Lista de etiquetas correspondientes a los datos de entrada.
            transform (callable, optional): Transformaciones a aplicar a los datos. Por defecto es None.
            normalization (str, optional): Método de normalización de tinción a utilizar ('reinhard', 'macenko' o None).
                                           Por defecto es None.
        """
        self.data = pd.DataFrame({'image': inputs, 'label': labels})
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
        return len(self.data)

    def __getitem__(self, index):
        """
        Genera una muestra de datos en función del índice.

        Args:
            index (int): Índice de la muestra.

        Returns:
            dict: Un diccionario que contiene la imagen, la etiqueta y el archivo.
        """
        row = self.data.iloc[index]
        file = row['image']
        x = imread(file).astype(np.uint8)

        try:
            if self.normalization is not None:
                # Aplicar normalización de tinción
                x = self.normalization_method.transform(x)
        except Exception as e:
            pass

        if self.transform:
            pil=transforms.ToPILImage()(x)
            x = self.transform(transforms.ToPILImage()(x))

        y = row['label']
        #y = torch.from_numpy(np.asarray(y)).float()

        return {'image': pil, 'label': y, 'pixel_values':x, 'file': file}
    
image_processor = AutoImageProcessor.from_pretrained("owkin/phikon")

# ImageNet normalization
normalize = Normalize(
    mean=image_processor.image_mean,
    std=image_processor.image_std
)




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

with open(data_RoI, 'rb') as fp:
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




train_transform = transforms.Compose(
    [
        RandomResizedCrop(image_processor.size["height"]),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ]
)

val_transform = transforms.Compose(
        [
        Resize(image_processor.size["height"]),
        CenterCrop(image_processor.size["height"]),
        ToTensor(),
        normalize,
    ]

    )

# Supongamos que dataReaders['CNN']['train']['y'] contiene tus datos one-hot encoded
y_label= {}
for i in datasets:
  y_label[i] = [np.argmax(item) for item in dataReaders['CNN'][i]['y']]

train_dataset = Dataset(dataReaders['CNN']['train']['x'],
                        y_label['train'], train_transform, norm)

#crear val dataset
val_dataset = Dataset(dataReaders['CNN']['val']['x'],
                y_label['val'], val_transform, norm)


#crear test dataset
test_dataset = Dataset(dataReaders['CNN']['test']['x'],
                    y_label['test'], val_transform, norm)


dataset_sizes = {x: len(dataReaders['CNN'][x]['x']) for x in ['train', 'val', 'test']}


# default: we set the working directory to the temporary directory
# created by Colab







def print_trainable_parameters(model: torch.nn) -> None:
    """Print number of trainable parameters."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param}"
        f" || trainable%: {100 * trainable_params / all_param:.2f}"
    )


# Labels from our dataset
label2id = {
    '0': "AT",
    '1': "BT",
    '2': "MT"
}
id2label = {v: k for (k, v) in label2id.items()}

# Load the model
model = AutoModelForImageClassification.from_pretrained(
    "owkin/phikon",
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=False,
    cache_dir=cache_dir,
)
print_trainable_parameters(model)



frozen_model = deepcopy(model)

for name, param in frozen_model.named_parameters():
     if not name.startswith("classifier."):
        param.requires_grad = False
print_trainable_parameters(frozen_model)


# load and configure LoRA from Hugging Face peft library
config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["classifier"],
)
lora_model = get_peft_model(model, config)
print_trainable_parameters(lora_model)

# Check if CUDA is available
if torch.cuda.is_available():
    # Get the current device
    device = torch.cuda.current_device()
    print(f"Using CUDA device {device}")
else:
    print("CUDA is not available. Using CPU.")




SEED = 123

# LoRA configuration



args = TrainingArguments(
    "phikon-finetuned-nct-1k",
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=lr,
    gradient_accumulation_steps=1,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    fp16=True,
    seed=SEED,
    num_train_epochs=num_epochs,
    logging_steps=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",  # dataset is roughly balanced
    push_to_hub=False,
    label_names=["labels"],
)

# Metric configuration

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred: np.ndarray) -> float:
    """Computes accuracy on a batch of predictions."""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

# Inputs generation for training

def collate_fn(examples) -> dict[str, torch.Tensor]:
    """Create the inputs for LoRA from an example in the dataset."""
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])

    return {"pixel_values": pixel_values, "labels": labels}





if lora:
    # Here is the final trainer
    trainer_lora = Trainer(
        model=lora_model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )
    # We display the accuracy on the test set at the end
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        train_results_lora = trainer_lora.train()
        metrics_lora = trainer_lora.evaluate(test_dataset)
        trainer_lora.log_metrics("Fine-tuned model: VAL-CRC-7K", metrics_lora)

else:
    trainer_frozen = Trainer(
    frozen_model,
    args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        train_results_frozen = trainer_frozen.train()
        metrics_frozen = trainer_frozen.evaluate(test_dataset)
        trainer_frozen.log_metrics("Frozen model: VAL-CRC-7K", metrics_frozen)