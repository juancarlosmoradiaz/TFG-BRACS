import torch
import numpy as np
from torchvision import transforms
from torch_staintools.normalizer import NormalizerBuilder
from typing import Any, List, Optional, Tuple
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data.dataloader import default_collate


def apply_transformations(file, transform=None, normalization=None, device='cuda'):
    x = file
  
    if normalization != None:
          normalization_method = NormalizerBuilder.build(normalization, use_cache=True,
                                              # use least square solver, along with cache, to perform
                                              # normalization on-the-fly
                                              concentration_method='ls')
          normalization_method = normalization_method.to(device)
    try:
        if normalization is not None:
            # Aplicar normalización de tinción
            x = normalization_method(x)
    except Exception as e:
        pass

    if transform is not None:
        x_transformed = transform(transforms.ToPILImage()(x)).to(device)
    return x_transformed

def build_path(row, prefix='', mode='train', extension=None, dir='.', wsi=False):
    group_path = f'Group_{row["group"]}'
    type_path = f'Type_{row["WSI label"]}'
    filename = row["WSI Filename"]

    if extension is not None:
        if not wsi:
            filename = f'{filename}/{filename}.{extension}'
        else:
            filename = f'{filename}.{extension}'
            dir='.'

    return f'{dir}/BRACS_WSI{prefix}/{mode}/{group_path}/{type_path}/{filename}'


def pad_collate_fn(
    batch: List[Tuple[torch.Tensor, Any]],
    batch_first: bool = True,
    max_len: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.BoolTensor, Any]:
    """Pad together sequences of arbitrary lengths.
    Add a mask of the padding to the samples that can later be used
    to ignore padding in activation functions.
    Expected to be used in combination of a torch.utils.datasets.DataLoader.
    Expect the sequences to be padded to be the first one in the sample tuples.
    Others members will be batched using ``torch.utils.data.dataloader.default_collate``.
    Parameters
    ----------
    batch: List[Tuple[torch.Tensor, Any]]
        List of tuples (features, Any). Features have shape (N_slides_tiles, F)
        with ``N_slides_tiles`` being specific to each slide depending on the
        number of extractable tiles in the tissue matter. ``F`` is the feature
        extractor output dimension.
    batch_first: bool = True
        Either return (B, N_TILES, F) or (N_TILES, B, F)
    max_len: Optional[int] = None
        Pre-defined maximum length for elements inside a batch.
    Returns
    -------
    padded_sequences, masks, Any: Tuple[torch.Tensor, torch.BoolTensor, Any]
        - if batch_first: Tuple[(B, N_TILES, F), (B, N_TILES, 1), ...]
        - else: Tuple[(N_TILES, B, F), (N_TILES, B, 1), ...]
        with N_TILES = max_len if max_len is not None
        or N_TILES = max length of the training samples.
    """
    # Expect the sequences to be the first one in the sample tuples
    sequences = []
    others = []
    for sample in batch:
        sequences.append(sample[0])
        others.append(sample[1:])

    if max_len is None:
        max_len = max([s.size(0) for s in sequences])

    trailing_dims = sequences[0].size()[1:]

    if batch_first:
        padded_dims = (len(sequences), max_len) + trailing_dims
        masks_dims = (len(sequences), max_len, 1)
    else:
        padded_dims = (max_len, len(sequences)) + trailing_dims
        masks_dims = (max_len, len(sequences), 1)

    padded_sequences = sequences[0].data.new(*padded_dims).fill_(0.0)
    masks = torch.ones(*masks_dims, dtype=torch.bool)

    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            padded_sequences[i, :length, ...] = tensor[:max_len, ...]
            masks[i, :length, ...] = False
        else:
            padded_sequences[:length, i, ...] = tensor[:max_len, ...]
            masks[:length, i, ...] = False

    # Batching other members of the tuple using default_collate
    others = default_collate(others)

    return (padded_sequences, masks, *others)
    

def auc(labels: np.array, logits: np.array) -> float:
    """ROC AUC score for binary classification.
    Parameters
    ----------
    labels: np.array
        Labels of the outcome.
    logits: np.array
        Probabilities.
    """
    return roc_auc_score(labels, logits, multi_class='ovr')
    