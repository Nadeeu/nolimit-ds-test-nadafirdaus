import numpy as np
from sentence_transformers import SentenceTransformer


def load_embedder(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def generate_embeddings(texts: list, model: SentenceTransformer, batch_size: int = 64) -> np.ndarray:
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True)


def save_embeddings(embeddings: np.ndarray, path: str) -> None:
    np.save(path, embeddings)


def load_embeddings(path: str) -> np.ndarray:
    return np.load(path)