import numpy as np
from collections import defaultdict
from typing import Iterable, List, Optional

from gensim.corpora import Dictionary
import joblib

from .base import TensorusModel


class GloVeModel(TensorusModel):
    """Minimal GloVe embedding implementation using ``gensim`` utilities."""

    def __init__(
        self,
        vector_size: int = 50,
        window: int = 5,
        iterations: int = 10,
        x_max: int = 100,
        alpha: float = 0.75,
        lr: float = 0.05,
    ) -> None:
        self.vector_size = int(vector_size)
        self.window = int(window)
        self.iterations = int(iterations)
        self.x_max = int(x_max)
        self.alpha = float(alpha)
        self.lr = lr
        self.dictionary: Optional[Dictionary] = None
        self.wi: Optional[np.ndarray] = None
        self.wj: Optional[np.ndarray] = None
        self.bi: Optional[np.ndarray] = None
        self.bj: Optional[np.ndarray] = None
        self.embeddings: Optional[np.ndarray] = None

    def _build_cooc(self, corpus: List[List[str]]):
        self.dictionary = Dictionary(corpus)
        cooc = defaultdict(float)
        for sentence in corpus:
            ids = [self.dictionary.token2id[w] for w in sentence if w in self.dictionary.token2id]
            for idx, i in enumerate(ids):
                start = max(0, idx - self.window)
                end = min(len(ids), idx + self.window + 1)
                for jdx in range(start, end):
                    if jdx == idx:
                        continue
                    j = ids[jdx]
                    distance = abs(jdx - idx)
                    cooc[(i, j)] += 1.0 / distance
        return cooc

    def fit(self, sentences: Iterable[List[str]], y: any = None) -> None:
        corpus = list(sentences)
        cooc = self._build_cooc(corpus)
        vocab_size = len(self.dictionary)
        self.wi = np.random.randn(vocab_size, self.vector_size) / np.sqrt(self.vector_size)
        self.wj = np.random.randn(vocab_size, self.vector_size) / np.sqrt(self.vector_size)
        self.bi = np.zeros(vocab_size)
        self.bj = np.zeros(vocab_size)

        for _ in range(self.iterations):
            for (i, j), x in cooc.items():
                weight = (x / self.x_max) ** self.alpha if x < self.x_max else 1.0
                diff = np.dot(self.wi[i], self.wj[j]) + self.bi[i] + self.bj[j] - np.log(x)
                grad = weight * diff
                self.wi[i] -= self.lr * grad * self.wj[j]
                self.wj[j] -= self.lr * grad * self.wi[i]
                self.bi[i] -= self.lr * grad
                self.bj[j] -= self.lr * grad

        self.embeddings = self.wi + self.wj

    def predict(self, tokens: List[str]) -> np.ndarray:
        if self.embeddings is None or self.dictionary is None:
            raise ValueError("Model not trained. Call fit() first.")
        return np.vstack([self.embeddings[self.dictionary.token2id[t]] for t in tokens])

    def save(self, path: str) -> None:
        if self.embeddings is None or self.dictionary is None:
            raise ValueError("Model not trained. Call fit() before save().")
        joblib.dump(
            {
                "wi": self.wi,
                "wj": self.wj,
                "bi": self.bi,
                "bj": self.bj,
                "token2id": self.dictionary.token2id,
            },
            path,
        )

    def load(self, path: str) -> None:
        data = joblib.load(path)
        self.wi = data["wi"]
        self.wj = data["wj"]
        self.bi = data["bi"]
        self.bj = data["bj"]
        self.dictionary = Dictionary()
        self.dictionary.token2id = data["token2id"]
        self.dictionary.id2token = {v: k for k, v in data["token2id"].items()}
        self.embeddings = self.wi + self.wj
