import numpy as np
from typing import Iterable, List, Optional
from gensim.models import Word2Vec
import joblib

from .base import TensorusModel


class Word2VecModel(TensorusModel):
    """Word2Vec embeddings using ``gensim.models.Word2Vec``."""

    def __init__(
        self,
        vector_size: int = 50,
        window: int = 5,
        min_count: int = 1,
        epochs: int = 5,
        **kwargs: any,
    ) -> None:
        self.vector_size = int(vector_size)
        self.window = int(window)
        self.min_count = int(min_count)
        self.epochs = int(epochs)
        self.kwargs = kwargs
        self.model: Optional[Word2Vec] = None

    def fit(self, sentences: Iterable[List[str]], y: any = None) -> None:
        self.model = Word2Vec(
            sentences=list(sentences),
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs,
            **self.kwargs,
        )

    def predict(self, tokens: List[str]) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return np.vstack([self.model.wv[token] for token in tokens])

    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Model not trained. Call fit() before save().")
        self.model.save(path)

    def load(self, path: str) -> None:
        self.model = Word2Vec.load(path)
