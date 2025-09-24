# Tensorus Tutorials

Welcome to the Tensorus tutorial series. These notebooks provide hands-on examples for working with tensors, from basics to production deployment. Each tutorial is self-contained, includes a lightweight dependency installer, and works in demo mode if the Tensorus API is unavailable.

## Table of Contents

| Tutorial | Description |
|----------|-------------|
| [01_getting_started.ipynb](01_getting_started.ipynb) | Introduction to Tensorus basics |
| [02_tensor_basics.ipynb](02_tensor_basics.ipynb) | Tensor creation, manipulations, devices, and dtypes |
| [03_hybrid_search.ipynb](03_hybrid_search.ipynb) | Hybrid search with semantic, math, and metadata filters |
| [04_tensor_decompositions.ipynb](04_tensor_decompositions.ipynb) | Advanced tensor decompositions with trade-off analysis |
| [05_automl_agent.ipynb](05_automl_agent.ipynb) | AutoML agent for classification and regression tasks |
| [06_rl_agent.ipynb](06_rl_agent.ipynb) | Reinforcement learning with DQN and prioritized replay |
| [07_performance.ipynb](07_performance.ipynb) | Performance benchmarking for latency and throughput |
| [08_scientific_features.ipynb](08_scientific_features.ipynb) | Scientific features with lineage tracking and reproducibility |
| [09_multi_modal.ipynb](09_multi_modal.ipynb) | Multi-modal data handling with cross-modal search |
| [10_production.ipynb](10_production.ipynb) | Production deployment, monitoring, HA, and DR simulation |
| [11_vector_workflows.ipynb](11_vector_workflows.ipynb) | Embeddings, FAISS indexing, similarity & hybrid search |
| [12_schema_metadata.ipynb](12_schema_metadata.ipynb) | Dataset schemas, validation, and metadata queries |
| [13_ingestion_agent.ipynb](13_ingestion_agent.ipynb) | Configure and monitor ingestion agents |
| [14_operations_lineage.ipynb](14_operations_lineage.ipynb) | Operation history and computational lineage |
| [15_storage_compression.ipynb](15_storage_compression.ipynb) | Storage backends, persistence, and compression |

## Prerequisites
- Python 3.9+
- Recommended: GPU with CUDA for performance demos, but CPU-only is supported.

Install dependencies (either run the first cell in any notebook or use the requirements file):

```bash
pip install -r tutorials/requirements.txt
```

## Running Tips
- For best results, start a Tensorus server locally (see repository docs) to unlock live API examples.
- GPU-accelerated sections will detect CUDA automatically; otherwise, they run on CPU.
- If a cell references an endpoint your server doesn’t expose yet, the notebook will continue in demo mode.
- Tutorials `11`–`15` import helpers from `tutorial_utils.py`; install requirements once for shared dependencies.

## Getting Started
Start with `01_getting_started.ipynb` to set up your environment and explore the core concepts. Each tutorial builds on the previous ones, covering progressively advanced topics.

## Tutorial Order

1. 01_getting_started.ipynb — Getting Started (intro and quickstart)
2. 02_tensor_basics.ipynb — Tutorial 2: Tensor Basics
3. 03_hybrid_search.ipynb — Tutorial 3: Hybrid Search
4. 04_tensor_decompositions.ipynb — Tutorial 4: Tensor Decompositions
5. 05_automl_agent.ipynb — Tutorial 5: AutoML Agent
6. 06_rl_agent.ipynb — Tutorial 6: RL Agent
7. 07_performance.ipynb — Tutorial 7: Performance
8. 08_scientific_features.ipynb — Tutorial 8: Scientific Features
9. 09_multi_modal.ipynb — Tutorial 9: Multi-Modal
10. 10_production.ipynb — Tutorial 10: Production

## Notes on Consistency
- File names and titles are being unified. “AGENT” is the correct naming for agent-related tutorials.
- All notebooks include server connectivity checks and will run in demo mode if the API is not available.

## Support
If you encounter any issues running a tutorial, please open an issue with the notebook name, cell output, and your environment details.

Happy tensor computing!
