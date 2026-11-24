<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/tensorus/.github/main/assets/hero-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/tensorus/.github/main/assets/hero-light.svg">
  <img alt="Tensorus — The Agentic Tensor Database and Data Lakehouse" src="https://raw.githubusercontent.com/tensorus/.github/main/assets/hero-dark.svg" width="880">
</picture>

<br/>

<p>
  <a href="https://tensorus.com"><img src="https://img.shields.io/badge/Platform-tensorus.com-173c2c?style=flat-square" alt="Tensorus Platform"></a>
  <a href="https://tensorus.com/handbook"><img src="https://img.shields.io/badge/Docs-Developer_Handbook-245c44?style=flat-square" alt="Developer Handbook"></a>
  <a href="https://pypi.org/project/tensorus/"><img src="https://img.shields.io/badge/PyPI-tensorus_0.2.0-0f766e?style=flat-square" alt="PyPI tensorus 0.2.0"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.12%2B-2563eb?style=flat-square" alt="Python 3.12+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-475569?style=flat-square" alt="MIT License"></a>
</p>

**[Website](https://tensorus.com)** · **[Developer Handbook](https://tensorus.com/handbook)** · **[API Reference](https://tensorus.com/handbook#ch-api-reference)** · **[Request Early Access](https://tensorus.com/#connect)** · **[Contact (`ai@tensorus.com`)](mailto:ai@tensorus.com)**

</div>

---

## A First-Class Home for Multidimensional Data & AI Agents

Tables have data warehouses. 1D embeddings have vector databases. Yet the **multidimensional numerical arrays** that power modern AI, robotics, scientific modeling, and quantitative analytics—feature matrices, attention maps, sensor streams, and physical grids—still end up scattered across raw `.npy` and `.zarr` buckets with brittle scripts and no safe way for AI agents to collaborate.

**Tensorus** is the **Agentic Tensor Database and Data Lakehouse**. This repository provides the official **`tensorus` Python SDK**, **Xarray** & **MLflow** integrations, and **Model Context Protocol (MCP)** client bridge.

| What You Get | How It Helps Your Team |
| :--- | :--- |
| **Native N-Dimensional Storage** | Store, slice, and stream arrays up to 16 dimensions with exact shapes, dtypes, physical units, coordinates, and lossless `SHA-256` integrity—never flattened into rows. |
| **Git-Like Versions & Snapshots** | Every update creates an immutable version. Freeze whole datasets into **Snapshots** and **Data Products** so any experiment or release is 100% reproducible months later. |
| **Safe AI Agent Collaboration** | Connect **Claude, Cursor, or custom MCP agents** with scoped, expiring permissions. Agents explore data and **propose** structured plans; execution happens only after **explicit approval**. |
| **Leak-Free Quality & Evaluation** | Pair tensors with boolean **validity masks** to keep missing or noisy values out of scaling and statistics, and gate new embeddings or outputs against golden references. |

<br/>

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/tensorus/.github/main/assets/workflow-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/tensorus/.github/main/assets/workflow-light.svg">
  <img alt="How Tensorus Works — Store & Version, Inspect & Propose, Approve & Reproduce" src="https://raw.githubusercontent.com/tensorus/.github/main/assets/workflow-dark.svg" width="880">
</picture>
</div>

---

## Where Teams Use Tensorus

- **Multimodal AI & Vector Pipeline Staging:** Version multimodal feature tensors, projection scalers, and masks; validate drift against golden baselines before pushing 1D embeddings to serving vector indexes.
- **Robotics, Autonomous Systems & Industrial Sensors:** Govern multi-channel sensor and trajectory arrays with explicit validity masks so faulty sensor windows never pollute downstream models.
- **Climate, Energy, Bio & Scientific AI:** Preserve `Xarray` coordinates (`lat`, `lon`, `time`) and physical units (`K`, `m/s`) natively, reading lightweight spatial/temporal slices without downloading full archives.
- **Regulated Enterprise & Quantitative Research:** Delegate routine data preparation to AI assistants under strict dataset access controls, outbound egress allowlists, and complete audit trails.

---

## Installation

Install the lightweight core client (4 pure dependencies: `numpy`, `httpx`, `pydantic`, `jsonschema`) or add optional ecosystem extras in Python **3.12+**:

```bash
# Core Python SDK & CLI
pip install tensorus

# With Model Context Protocol (MCP) bridge for Claude Desktop, Claude Code, Cursor, etc.
pip install "tensorus[mcp]"

# With Xarray labeled-array & coordinate interoperability
pip install "tensorus[xarray]"

# With MLflow evaluation, embedding refresh & experiment lineage
pip install "tensorus[mlflow]"

# Everything included
pip install "tensorus[all]"
```

---

## 30-Second Quickstart

Store an array, verify lossless readback, and run a reviewed numerical transformation:

```python
import numpy as np
from tensorus import Client
from tensorus.models import Operation, PlanDefinition, Step

with Client("https://api.tensorus.com", api_key="tns_...") as client:
    # 1. Create a dataset and store an N-dimensional tensor
    client.create_dataset("climate_grid", description="Global sea-surface temperature grids")

    grid = np.asarray([[289.15, 290.40], [288.90, 291.05]], dtype="float32")
    v1 = client.put(
        "climate_grid",
        "sst_kelvin",
        grid,
        expected_version=0,
        metadata={"source": "argo-float-network", "unit": "K"},
    )

    # 2. Read back losslessly (or preview a window slice without downloading everything)
    restored = client.get(v1["id"])
    assert np.array_equal(restored, grid)

    # 3. Preview (`explain`) and propose a structured workflow
    definition = PlanDefinition(
        dataset="climate_grid",
        title="Normalize sea-surface temperature",
        steps=[
            Step(
                id="norm",
                operation=Operation.normalize,
                inputs=[v1["id"]],
                output="sst_norm",
            )
        ],
    )
    preflight = client.explain(definition)
    plan = client.propose(definition)

    # 4. Approve the exact reviewed plan hash and wait for completion
    client.approve(plan["id"], plan["plan_hash"])
    finished = client.wait(plan["id"], timeout=60)
```

---

## Ecosystem Integrations

### 1. AI Agents via MCP (`tensorus[mcp]`)

Give **Claude Desktop, Claude Code, Cursor**, or custom MCP agents governed access to your workspace—without exposing unrestricted write keys or running arbitrary code:

- **Safe by default:** Read-only discovery, tensor metadata, window previews, and dry-run `explain` checks work out of the box.
- **Scoped delegation:** Plan proposals require an explicit opt-in (`TENSORUS_MCP_ENABLE_PROPOSALS=1`) and a dataset/budget-scoped grant key.
- **Human-in-the-loop:** Plan approval (`Client.approve`) and workspace administration are never exposed over MCP.

```json
{
  "mcpServers": {
    "tensorus": {
      "command": "tensorus-mcp",
      "env": {
        "TENSORUS_URL": "https://api.tensorus.com",
        "TENSORUS_API_KEY": "tns_your_scoped_agent_grant_key",
        "TENSORUS_MCP_ENABLE_PROPOSALS": "1"
      }
    }
  }
}
```

### 2. Labeled Coordinates with Xarray (`tensorus[xarray]`)

Round-trip `xarray.DataArray` dimensions, coordinates, and physical units directly to and from Tensorus:

```python
import xarray as xr
from tensorus.xarray import prepare_xarray, read_xarray

prepared = prepare_xarray(data_array)
version = client.put("ocean_obs", "surface_temp", prepared.values, expected_version=0, metadata=prepared.metadata)

restored = read_xarray(client, version["id"])  # preserves dims, coords & units
```

### 3. Evaluation & Experiment Lineage with MLflow (`tensorus[mlflow]`)

Connect MLflow model/agent evaluations, multimodal embedding refreshes, and calibration runs to exact SHA-256-verified tensors, scalers, and releases:

```python
import mlflow
from tensorus.mlflow import (
    as_mlflow_dataset,
    load_logged_tensor,
    log_fitted_state,
    log_product_release,
    log_reference_evaluation,
)

with mlflow.start_run(run_name="multimodal-embedding-refresh") as run:
    mlflow.log_input(as_mlflow_dataset(embedding_release), context="evaluation")
    log_product_release(embedding_release, label="candidate_embeddings", context="evaluation")
    log_fitted_state(projection_scaler, label="scaler")
    log_reference_evaluation(golden_drift_eval, label="drift_gate")

# Downstream serving or audit: verify hashes & reload exact arrays
embeddings = load_logged_tensor(client, run.info.run_id, label="candidate_embeddings", role="embeddings")
```

---

## Command-Line Utilities

```bash
tensorus --version
tensorus whoami --url https://api.tensorus.com --api-key "$TENSORUS_API_KEY"
tensorus datasets --url https://api.tensorus.com --api-key "$TENSORUS_API_KEY"
tensorus inspect-npy ./sample.npy
tensorus mcp
```

---

## Documentation & Resources

- **[Developer Handbook (`tensorus.com/handbook`)](https://tensorus.com/handbook):** Full 10-chapter guide covering quickstart, physical semantics, validity masks, all 65 operations, MCP setup, Data Products, governance, and interactive API explorer.
- **[Examples (`examples/`)](examples/):** Runnable Python examples for quickstart, Xarray, and MLflow workflows.
- **[Cloud Workspace & Early Access (`tensorus.com`)](https://tensorus.com):** Explore plans (`Starter $0`, `Pro`, `Scale`, `Enterprise`) or connect to your workspace.

---

## Heritage & Historical Archive

Initiated in **December 2024** and first published to GitHub and PyPI in **April 2025**, Tensorus pioneered the **Agentic Tensor Database & Data Lakehouse** category—combining native multidimensional tensor storage with structured AI-agent workflows.

- **`0.2.0+` (`main`):** Production-grade commercial client SDK, Xarray & MLflow integrations, and Model Context Protocol (MCP) bridge.
- **`0.0.x`–`0.1.0` (`legacy/v0.1-archive` / tag `archive/legacy-v0.1`):** Original open-source research prototype preserved permanently for historical reference and reproducibility.

---

## License

MIT License — Copyright (c) 2024–2026 Tensorus.
