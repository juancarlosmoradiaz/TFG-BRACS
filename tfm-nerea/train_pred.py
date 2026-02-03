import time
import torch
import copy
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from collections import Counter 
import logging
from sklearn.metrics import roc_auc_score, confusion_matrix
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans


device = torch.device("cuda:0" if torch.cuda.is_available()
                                   else "cpu")

import logging
import time
import copy
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

def train_model(model, criterion, optimizer, dataloaders, dataset_sizes,
                lr_scheduler,  warmup_scheduler, save_path, num_epochs=25, verbose=True):
    LOG = save_path + "/execution.log"
    logging.basicConfig(filename=LOG, filemode="w", level=logging.DEBUG)

    # controlador de consola
    console = logging.StreamHandler()
    console.setLevel(logging.ERROR)
    logging.getLogger("").addHandler(console)

    logger = logging.getLogger(__name__)
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    best_auc = 0

    acc_array = {'train': [], 'val': []}
    loss_array = {'train': [], 'val': []}
    auc_array = {'train': [], 'val': []}
    best_val_preds = []
    best_train_preds = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        logger.debug('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logger.debug('-' * 10)
        val_preds = []
        train_preds = []
        train_case_ids = []
        train_labels = []
        val_labels = []
        train_labels_auc = []
        train_preds_auc = []
        val_labels_auc = []
        val_preds_auc = []
        train_probs = []
        val_probs = []
        sizes = {'train': 0, 'val': 0}
        # Cada epoch tiene una fase de entrenamiento y otra de validación
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Poner el modelo en modo de entrenamiento
            else:
                model.eval()   # Poner el modelo en modo evaluación

            running_loss = 0.0
            running_corrects = 0

            # Iterar sobre los datos.
            for inputs, labels, case_ids in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                inputs.requires_grad = True
                labels = labels.to(device)
                # poner a cero los gradientes de los parámetros
                optimizer.zero_grad()

                # hacia delante
                # seguimiento del historial si sólo en entrenamiento
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    _, mlabels = torch.max(labels, 1)

                    loss = criterion(outputs, mlabels)
                    if phase == 'val':
                        val_preds += list(preds.cpu().numpy())
                        val_labels += list(mlabels.cpu().numpy())
                        val_probs.extend(list(outputs.cpu().detach().numpy()))

                    # hacia atrás + optimizar sólo si está en fase de entrenamiento

                    if phase == 'train':
                        train_preds.extend(preds.cpu().numpy())
                        train_case_ids.extend(case_ids)
                        train_labels.extend(mlabels.cpu().numpy())
                        train_probs.extend(list(outputs.cpu().detach().numpy()))
                        loss.backward()
                        optimizer.step()
                        with warmup_scheduler.dampening():
                             lr_scheduler.step()
                # estadicticas

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == mlabels)
                sizes[phase] += inputs.size(0)

            
            epoch_loss = running_loss / sizes[phase]
            epoch_acc = running_corrects.item() / sizes[phase]
            if phase == 'train':
                 writer.add_scalar("Loss/train", epoch_loss, epoch)
                 writer.add_scalar("Acc/train", epoch_acc, epoch)
            if phase == 'val':
                writer.add_scalar("Loss/val", epoch_loss, epoch)
                writer.add_scalar("Acc/val", epoch_acc, epoch)
            loss_array[phase].append(epoch_loss)
            acc_array[phase].append(epoch_acc)

            if verbose:
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
                logger.debug('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_epoch = epoch
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_val_preds = val_preds[:]
                best_train_preds = train_preds[:]
                best_train_probs= np.matrix(train_probs[:])
                best_val_probs= np.matrix(val_probs[:])
        print()

    time_elapsed = time.time() - since
    if verbose:
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        logger.debug('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        logger.debug('Best val Acc: {:4f}'.format(best_acc))

    # cargar las pesos del mejor modelo
    model.load_state_dict(best_model_wts)
    if best_val_preds == []:
        best_val_preds = val_preds[:]
    results = {
        'model': model,
        'best_acc': best_acc,
       'best_auc': best_auc,
        'best_epoch': best_epoch,
        'acc_array': acc_array,
        'loss_array': loss_array,
        'auc_array': auc_array,
        'val_preds': best_val_preds,
        'val_labels': val_labels,
        'train_preds': best_train_preds,
        'train_case_ids': train_case_ids,
        'train_labels': train_labels,
        'train_probs': best_train_probs,
        'val_probs': best_val_probs
    }

    return results

import torch.nn.functional as F

def predict_WSI(model, dataloader, dataset_size, verbose=True):
    """Predecir patches/imagenes"""
    activation = {}
    def _get_features(name):
        def hook(model, input, output):
            activation[name] = input[0].detach()
        return hook

    model.eval()
    since = time.time()
    corrects = 0
    # variables
    test_preds = []
    probs = []
    model.fc.register_forward_hook(_get_features('fc'))
    features = []
    case_ids = []
    test_labels = []

    for inputs, labels, cids in tqdm(dataloader):
        inputs = inputs.to(device)
        inputs.requires_grad = True
        labels = labels.to(device)
        outputs = model(inputs)
        _, mlabel = torch.max(labels, 1)
        _, preds = torch.max(outputs, 1)

        test_preds += list(preds.cpu().numpy())
        test_labels += list(mlabel.cpu().numpy())
        probabilities = F.softmax(outputs, dim=1)  # Aplicar softmax a las salidas del modelo
        probs.extend(list(probabilities.cpu().detach().numpy()))
        features.append(activation['fc'].cpu().numpy())
        case_ids.append(cids)
        # Calcular precisión
        corrects += torch.sum(preds == mlabel)

    acc = corrects.item() / dataset_size

    probs = np.matrix(probs[:])

    time_elapsed = time.time() - since
    if verbose:
        print('Test complete in {:.0f}m {:.0f}s'.format(
              time_elapsed // 60, time_elapsed % 60))

    all_results = {
        'acc': acc,
        'preds': test_preds,
        'labels': test_labels,
        'probs': probs,
        'patch_case_ids': case_ids,
        'features': features
    }

    return all_results
