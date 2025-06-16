import logging
import time
from typing import Dict, TYPE_CHECKING

from fastapi import HTTPException, status

from tensorus.tensor_storage import (
    TensorStorage,
    DatasetNotFoundError,
    TensorNotFoundError,
)
from tensorus.nql_agent import NQLAgent

if TYPE_CHECKING:  # pragma: no cover - for type hints only
    from tensorus.ingestion_agent import DataIngestionAgent

logger = logging.getLogger(__name__)

# Initialize core components
try:
    tensor_storage_instance = TensorStorage()
    nql_agent_instance = NQLAgent(tensor_storage_instance)
except Exception as e:  # pragma: no cover - fail fast at import
    logger.exception("Failed initializing Tensorus components: %s", e)
    raise

# Agent registry and live agent tracking
agent_registry = {
    "ingestion": {
        "name": "Data Ingestion",
        "description": "Monitors sources and ingests data into TensorStorage.",
        "config": {
            "source_directory": "./temp_ingestion_source_api",
            "polling_interval_sec": 15,
            "dataset_name": "ingested_data_api",
        },
    },
    "rl_trainer": {
        "name": "RL Trainer",
        "description": "Trains reinforcement learning models using stored experiences.",
        "status": "stopped",
        "config": {
            "experience_dataset": "rl_experiences",
            "batch_size": 128,
            "target_update_freq": 500,
        },
        "last_log_timestamp": None,
    },
    "automl_search": {
        "name": "AutoML Search",
        "description": "Performs hyperparameter optimization.",
        "status": "stopped",
        "config": {
            "trials": 50,
            "results_dataset": "automl_results",
            "task_type": "regression",
        },
        "last_log_timestamp": None,
    },
    "nql_query": {
        "name": "NQL Query Service",
        "description": "Processes natural language queries.",
        "status": "running",
        "config": {"parser_type": "regex"},
        "last_log_timestamp": None,
    },
}

live_agents: Dict[str, "DataIngestionAgent"] = {}


def _get_or_create_ingestion_agent() -> "DataIngestionAgent":
    from tensorus.ingestion_agent import DataIngestionAgent

    if "ingestion" not in live_agents:
        config = agent_registry["ingestion"]["config"]
        source_dir = config["source_directory"]
        dataset_name = config["dataset_name"]
        polling_interval = config["polling_interval_sec"]
        live_agents["ingestion"] = DataIngestionAgent(
            source_directory=source_dir,
            dataset_name=dataset_name,
            polling_interval_sec=polling_interval,
            storage=tensor_storage_instance,
        )
    return live_agents["ingestion"]


async def get_tensor_storage() -> TensorStorage:
    return tensor_storage_instance


async def get_nql_agent() -> NQLAgent:
    return nql_agent_instance
