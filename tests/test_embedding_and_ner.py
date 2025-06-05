import numpy as np
import torch

from tensorus.models.word2vec_model import Word2VecModel
from tensorus.models.glove_model import GloVeModel
from tensorus.models.named_entity_recognition import NamedEntityRecognitionModel


def _get_corpus():
    return [["hello", "world"], ["hello", "tensorus"]]


def test_word2vec_training(tmp_path):
    corpus = _get_corpus()
    model = Word2VecModel(vector_size=10, window=2, min_count=1, epochs=20)
    model.fit(corpus)
    vec = model.predict(["hello"])
    assert vec.shape == (1, 10)

    save_path = tmp_path / "w2v.model"
    model.save(str(save_path))
    model2 = Word2VecModel()
    model2.load(str(save_path))
    assert np.allclose(model2.predict(["hello"]), vec)


def test_glove_training(tmp_path):
    corpus = _get_corpus()
    model = GloVeModel(vector_size=10, window=2, iterations=5)
    model.fit(corpus)
    vec = model.predict(["hello"])
    assert vec.shape == (1, 10)

    save_path = tmp_path / "glove.joblib"
    model.save(str(save_path))
    model2 = GloVeModel(vector_size=10)
    model2.load(str(save_path))
    assert np.allclose(model2.predict(["hello"]), vec)


def test_simple_ner_tagging():
    sentences = np.array([[0, 1, 2], [2, 1, 0]])
    tags = np.array([[1, 0, 1], [1, 0, 1]])
    model = NamedEntityRecognitionModel(
        vocab_size=3,
        tagset_size=2,
        embedding_dim=8,
        hidden_dim=8,
        lr=0.1,
        epochs=100,
    )
    model.fit(sentences, tags)
    preds = model.predict(sentences)
    assert torch.equal(preds, torch.tensor(tags))
