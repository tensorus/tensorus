"""Utilities for loading data and storing predictions in TensorStorage.

This module contains convenience helpers for common workflows when training
models using data stored in :class:`~tensorus.tensor_storage.TensorStorage`.

Example
-------
>>> from tensorus.tensor_storage import TensorStorage
>>> from tensorus.models.linear_regression import LinearRegressionModel
>>> from tensorus.models.utils import load_xy_from_storage, store_predictions
>>>
>>> storage = TensorStorage()
>>> # assume 'train_ds' contains feature tensors with metadata {'label': int}
>>> X, y = load_xy_from_storage(storage, 'train_ds', target_field='label')
>>> model = LinearRegressionModel()
>>> model.fit(X, y)
>>> preds = model.predict(X)
>>> store_predictions(storage, 'train_predictions', preds,
...                   model_name='LinearRegressionModel')
"""

from typing import Tuple, Any, Dict, Optional
import logging
import torch

from ..tensor_storage import TensorStorage

logger = logging.getLogger(__name__)


def load_xy_from_storage(
    storage: TensorStorage,
    dataset_name: str,
    target_field: str = "y",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load feature and target tensors from a dataset.

    The dataset should store one tensor per sample. Each tensor's metadata must
    contain the target value under ``target_field``. All tensors are stacked into
    a single feature matrix ``X`` and the targets are returned as a 1-D tensor
    ``y``.

    Parameters
    ----------
    storage:
        The :class:`TensorStorage` instance to load from.
    dataset_name:
        Name of the dataset containing the records.
    target_field:
        Metadata key that holds the target value for each record.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        ``(X, y)`` where ``X`` has shape ``(n_samples, ...)`` and ``y`` has shape
        ``(n_samples,)``.
    """
    records = storage.get_dataset_with_metadata(dataset_name)
    features = []
    targets = []
    for rec in records:
        tensor = rec["tensor"]
        meta = rec["metadata"]
        if target_field not in meta:
            raise ValueError(f"Record missing target field '{target_field}'")
        features.append(tensor)
        targets.append(meta[target_field])

    X = torch.stack([t.float() for t in features]) if features else torch.empty(0)
    y = torch.tensor(targets, dtype=torch.float32) if targets else torch.empty(0)
    logger.info(
        "Loaded dataset '%s' with %d samples from TensorStorage", dataset_name, len(features)
    )
    return X, y


def store_predictions(
    storage: TensorStorage,
    dataset_name: str,
    predictions: torch.Tensor,
    model_name: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Store prediction tensor into ``TensorStorage``.

    Parameters
    ----------
    storage:
        Destination :class:`TensorStorage`.
    dataset_name:
        Dataset name to insert into (created if missing).
    predictions:
        Tensor of model predictions.
    model_name:
        Identifier of the model that produced the predictions.
    metadata:
        Optional additional metadata to store alongside the tensor.

    Returns
    -------
    str
        Record ID of the stored prediction tensor.
    """
    if dataset_name not in storage.list_datasets():
        storage.create_dataset(dataset_name)
    meta = metadata.copy() if metadata else {}
    meta.update({"model_name": model_name, "created_by": "store_predictions"})
    record_id = storage.insert(dataset_name, predictions.cpu(), meta)
    logger.info(
        "Stored predictions in dataset '%s' with record_id '%s'", dataset_name, record_id
    )
    return record_id
