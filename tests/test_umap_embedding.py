import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("umap")

from tensorus.models.umap_embedding import UMAPEmbeddingModel


def test_umap_embedding_dimensionality_reduction(tmp_path):
    X = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0]])
    model = UMAPEmbeddingModel(n_components=2, random_state=42)
    X_embedded = model.fit_transform(X)
    assert X_embedded.shape == (4, 2)

    save_path = tmp_path / "umap.joblib"
    model.save(str(save_path))
    model2 = UMAPEmbeddingModel(n_components=2)
    model2.load(str(save_path))
    assert np.allclose(model2.transform(X), model.transform(X))

