# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pickle
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset


def slide_level_train_step(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Training step for slide-level experiments. This will serve as the
    ``train_step`` in ``TorchTrainer``printclass.
    Parameters
    ----------
    model: nn.Module
        The PyTorch model to be trained.
    train_dataloader: torch.utils.data.DataLoader
        Training data loader.
    criterion: nn.Module
        The loss criterion used for training.
    optimizer: Callable = Adam
        The optimizer class to use.
    device : str = "cpu"
        The device to use for training and evaluation.
    """
    model.train()

    _epoch_loss, _epoch_logits, _epoch_labels = [], [], []

    for batch in train_dataloader:
        # Get data.
        features, mask, labels = batch

        # Put on device.
        features = features.to(device)
        mask = mask.to(device)
        labels = labels.to(device)

        # Compute logits and loss.
        logits = model(features, mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Stack logits & labels to compute epoch metrics.
        _epoch_loss.append(loss.detach().cpu().numpy())
        _epoch_logits.append(logits.detach())
        _epoch_labels.append(labels.detach())

    _epoch_loss = np.mean(_epoch_loss)
    _epoch_logits = torch.cat(_epoch_logits, dim=0).cpu().numpy()
    _epoch_labels = torch.cat(_epoch_labels, dim=0).cpu().numpy()

    return _epoch_loss, _epoch_logits, _epoch_labels


def slide_level_val_step(
    model: torch.nn.Module,
    val_dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inference step for slide-level experiments. This will serve as the
    ``val_step`` in ``TorchTrainer``class.
    Parameters
    ----------
    model: nn.Module
        The PyTorch model to be trained.
    val_dataloader: torch.utils.data.DataLoader
        Inference data loader.
    criterion: nn.Module
        The loss criterion used for training.
    device : str = "cpu"
        The device to use for training and evaluation.
    """
    model.eval()

    with torch.no_grad():
        _epoch_loss, _epoch_logits, _epoch_labels = [], [], []

        for batch in val_dataloader:
            # Get data.
            features, mask, labels = batch

            # Put on device.
            features = features.to(device)
            mask = mask.to(device)
            labels = labels.to(device)

            # Compute logits and loss.
            logits = model(features, mask)
            loss = criterion(logits, labels)

            # Stack logits & labels to compute epoch metrics.
            _epoch_loss.append(loss.detach().cpu().numpy())
            _epoch_logits.append(logits.detach())
            _epoch_labels.append(labels.detach())

    _epoch_loss = np.mean(_epoch_loss)
    _epoch_logits = torch.cat(_epoch_logits, dim=0).cpu().numpy()
    _epoch_labels = torch.cat(_epoch_labels, dim=0).cpu().numpy()

    return _epoch_loss, _epoch_logits, _epoch_labels

class TorchTrainer:
    """Trainer class for training and evaluating PyTorch models.
    Parameters
    ----------
    model: nn.Module
        The PyTorch model to be trained.
    criterion: nn.Module
        The loss criterion used for training.
    metrics: Dict[str, Callable]
        Dictionary of metrics functions to evaluate the model's performance.
    batch_size: int = 16
        The batch size for training and evaluation
    num_epochs : int = 10
        The number of training epochs.
    learning_rate: float = 1.0e-3
        The learning rate for the optimizer.
    weight_decay: float = 0.0
        The weight decay for the optimizer.
    device : str = "cpu"
        The device to use for training and evaluation.
    num_workers: int = 8
        Number of workers.
    optimizer: Callable = Adam
        The optimizer class to use.
    train_step: Callable = slide_level_train_step
        The function for training step.
    val_step: Callable = slide_level_val_step
        The function for validation step.
    collator: Optional[Callable] = None
        The collator function for data preprocessing.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        metrics: Dict[str, Callable],
        batch_size: int = 16,
        num_epochs: int = 10,
        learning_rate: float = 1.0e-3,
        weight_decay: float = 0.0,
        device: str = "cpu",
        num_workers: int = 8,
        optimizer: Callable = Adam,
        train_step: Callable = slide_level_train_step,
        val_step: Callable = slide_level_val_step,
        collator: Optional[Callable] = None,
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.metrics = metrics

        self.train_step = train_step
        self.val_step = val_step

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.collator = collator
        self.device = device
        self.num_workers = num_workers

        self.train_losses: List[float]
        self.val_losses: List[float]
        self.train_metrics: Dict[str, List[float]]
        self.val_metrics: Dict[str, List[float]]

    def train(
        self,
        train_set: Subset,
        val_set: Subset,
    ) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
        """
        Train the model using the provided training and validation datasets.
        Parameters
        ----------
        train_set: Subset
            The training dataset.
        val_set: Subset
            The validation dataset.
        Returns
        -------
        Tuple[Dict[str, List[float]], Dict[str, List[float]]]
            2 dictionaries containing the training and validation metrics for each epoch.
        """
        # Dataloaders.
        train_dataloader = DataLoader(
            dataset=train_set,
            shuffle=True,
            batch_size=self.batch_size,
            pin_memory=True,
            collate_fn=self.collator,
            drop_last=True,
            num_workers=self.num_workers,
        )
        val_dataloader = DataLoader(
            dataset=val_set,
            shuffle=False,
            batch_size=self.batch_size,
            pin_memory=True,
            collate_fn=self.collator,
            drop_last=False,
            num_workers=self.num_workers,
        )

        # Prepare modules.
        model = self.model.to(self.device)
        criterion = self.criterion.to(self.device)
        optimizer = self.optimizer(
            params=model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Training.
        train_losses, val_losses = [], []
        train_metrics: Dict[str, List[float]] = {
            k: [] for k in self.metrics.keys()
        }
        val_metrics: Dict[str, List[float]] = {
            k: [] for k in self.metrics.keys()
        }
        for ep in range(self.num_epochs):
            # Train step.
            (
                train_epoch_loss,
                train_epoch_logits,
                train_epoch_labels,
            ) = self.train_step(
                model=model,
                train_dataloader=train_dataloader,
                criterion=criterion,
                optimizer=optimizer,
                device=self.device,
            )

            # Inference step.
            val_epoch_loss, val_epoch_logits, val_epoch_labels = self.val_step(
                model=model,
                val_dataloader=val_dataloader,
                criterion=criterion,
                device=self.device,
            )
            
            # Compute metrics.
            for k, m in self.metrics.items():
                train_metric = m(train_epoch_labels, train_epoch_logits)
                val_metric = m(val_epoch_labels, val_epoch_logits)

                train_metrics[k].append(train_metric)
                val_metrics[k].append(val_metric)

                print(
                    f"Epoch {ep+1}: train_loss={train_epoch_loss:.5f}, train_{k}={train_metric:.4f}, val_loss={val_epoch_loss:.5f}, val_{k}={val_metric:.4f}"
                )

            train_losses.append(train_epoch_loss)
            val_losses.append(val_epoch_loss)

        self.train_losses = train_losses
        self.val_losses = val_losses
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics

        return train_metrics, val_metrics

    def evaluate(
        self,
        test_set: Subset,
    ) -> Dict[str, float]:
        """Evaluate the model using the provided test dataset.
        Parameters
        ----------
        test_set: Subset
            The test dataset.
        Returns
        -------
        Dict[str, float]
            A dictionary containing the test metrics.
        """
        # Dataloader.
        test_dataloader = DataLoader(
            dataset=test_set,
            shuffle=False,
            batch_size=self.batch_size,
            pin_memory=True,
            collate_fn=self.collator,
            drop_last=False,
            num_workers=self.num_workers,
        )

        # Prepare modules.
        model = self.model.to(self.device)
        criterion = self.criterion.to(self.device)

        # Inference step.
        _, test_epoch_logits, test_epoch_labels = self.val_step(
            model=model,
            val_dataloader=test_dataloader,
            criterion=criterion,
            device=self.device,
        )

        # Compute metrics.
        test_metrics = {
            k: m(test_epoch_labels, test_epoch_logits)
            for k, m in self.metrics.items()
        }

        return test_metrics

    def predict(
        self,
        test_set: Subset,
    ) -> Tuple[np.array, np.array]:
        """Make predictions using the provided test dataset.
        Parameters
        ----------
        test_set: Subset
            The test dataset.
        Returns
        --------
        Tuple[np.array, np.array]
            A tuple containing the test labels and logits.
        """
        # Dataloader
        test_dataloader = DataLoader(
            dataset=test_set,
            shuffle=False,
            batch_size=self.batch_size,
            pin_memory=True,
            collate_fn=self.collator,
            drop_last=False,
            num_workers=self.num_workers,
        )

        # Prepare modules
        model = self.model.to(self.device)
        criterion = self.criterion.to(self.device)

        # Val step
        _, test_epoch_logits, test_epoch_labels = self.val_step(
            model=model,
            val_dataloader=test_dataloader,
            criterion=criterion,
            device=self.device,
        )

        return test_epoch_labels, test_epoch_logits