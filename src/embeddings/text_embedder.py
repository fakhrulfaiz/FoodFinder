from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class TextEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2", verbose: bool = True):
        self.model = SentenceTransformer(model_name)
        self.dimension = 768
        if verbose:
            print(f"Loaded text embedder: {model_name}")
    
    def embed_batch(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
    
    def embed(self, text: str) -> np.ndarray:
        return self.model.encode([text], convert_to_numpy=True)[0]
