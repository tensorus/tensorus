"""Shared utilities for Tensorus tutorial notebooks.

These helpers centralize repeated setup steps so each notebook can focus on
showcasing specific features. All functions gracefully fall back to demo mode
when the Tensorus API is unavailable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import requests

DEFAULT_API_URL = "http://127.0.0.1:7860"


@dataclass
class DatasetResult:
    name: str
    created: bool
    payload: Dict[str, Any]


def ping_server(api_url: str = DEFAULT_API_URL, timeout: float = 5.0) -> bool:
    """Return True if the Tensorus API responds to `/health` within *timeout*."""
    try:
        response = requests.get(f"{api_url}/health", timeout=timeout)
        return response.status_code == 200
    except requests.RequestException:
        return False


def ensure_dataset(name: str, api_url: str = DEFAULT_API_URL) -> DatasetResult:
    """Create a dataset if it does not already exist."""
    try:
        payload = {"name": name}
        response = requests.post(f"{api_url}/datasets/create", json=payload)
        created = response.status_code == 200
        return DatasetResult(name=name, created=created, payload=response.json())
    except requests.RequestException as exc:  # pragma: no cover
        return DatasetResult(name=name, created=False, payload={"error": str(exc)})


def ingest_tensor(
    dataset_name: str,
    tensor: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
    api_url: str = DEFAULT_API_URL,
) -> Dict[str, Any]:
    """Serialize *tensor* and send it to `/datasets/{dataset_name}/ingest`."""
    metadata = metadata or {}
    payload = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "data": tensor.tolist(),
        "metadata": metadata,
    }
    try:
        response = requests.post(
            f"{api_url}/datasets/{dataset_name}/ingest",
            json=payload,
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:  # pragma: no cover
        return {"error": str(exc)}


def fetch_dataset(dataset_name: str, api_url: str = DEFAULT_API_URL) -> Dict[str, Any]:
    """Fetch dataset contents with defensive error handling."""
    try:
        response = requests.get(f"{api_url}/datasets/{dataset_name}/fetch", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:  # pragma: no cover
        return {"error": str(exc)}


def summarize_records(records: Iterable[Dict[str, Any]]) -> List[str]:
    """Return a human-readable summary for dataset records."""
    summaries: List[str] = []
    for idx, record in enumerate(records):
        metadata = record.get("metadata", {}) if isinstance(record, dict) else {}
        record_id = record.get("record_id", "unknown")
        name = metadata.get("name", f"tensor_{idx}")
        shape = metadata.get("shape") or metadata.get("tensor_shape")
        summaries.append(
            f"{name} (record_id={record_id}, shape={shape or 'unknown'})"
        )
    return summaries


def tensor_addition(
    dataset_name: str,
    tensor_ids: Iterable[str],
    api_url: str = DEFAULT_API_URL,
) -> Dict[str, Any]:
    """Add two tensors by record ID using `/ops/add`."""
    ids = list(tensor_ids)
    if len(ids) != 2:
        raise ValueError("tensor_addition requires exactly two tensor IDs")
    payload = {
        "input1": {"dataset_name": dataset_name, "record_id": ids[0]},
        "input2": {"dataset_name": dataset_name, "record_id": ids[1]},
    }
    try:
        response = requests.post(f"{api_url}/ops/add", json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:  # pragma: no cover
        return {"error": str(exc)}


def pretty_json(data: Dict[str, Any]) -> str:
    """Return formatted JSON for quick printing inside notebooks."""
    return json.dumps(data, indent=2, sort_keys=True)
