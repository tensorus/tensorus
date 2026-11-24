"""MLflow evaluation & lineage example (`pip install "tensorus[mlflow]"`)."""

import os

import mlflow
import numpy as np

from tensorus import Client
from tensorus.mlflow import as_mlflow_dataset, load_logged_tensor, log_tensor_version


def main() -> None:
    url = os.environ.get("TENSORUS_URL", "https://api.tensorus.com")
    api_key = os.environ["TENSORUS_API_KEY"]

    with Client(url, api_key) as client:
        client.create_dataset("embedding_eval_demo", description="Multimodal embedding evaluation")
        embeddings = np.asarray([[0.15, -0.40, 0.85], [0.60, 0.25, -0.75]], dtype="float32")
        version = client.put(
            "embedding_eval_demo",
            "candidate_embeddings",
            embeddings,
            expected_version=0,
        )

        with mlflow.start_run(run_name="candidate-embedding-check") as run:
            mlflow.log_input(as_mlflow_dataset(version), context="evaluation")
            receipt = log_tensor_version(version, label="candidate_embeddings")

        verified = load_logged_tensor(client, run.info.run_id, label="candidate_embeddings")
        assert np.array_equal(verified.values, embeddings)
        print(f"Logged & SHA-256-verified MLflow manifest {receipt.manifest_sha256[:12]}...")


if __name__ == "__main__":
    main()
