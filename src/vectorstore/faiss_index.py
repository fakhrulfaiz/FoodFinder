import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Optional

class FAISSIndex:
    def __init__(self, dimension: int, index_type: str = "Flat"):
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.metadata = []
        
    def create_index(self):
        if self.index_type == "Flat":
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "IVF":
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
    
    def add(self, embeddings: np.ndarray, metadata: List[dict]):
        if self.index is None:
            self.create_index()
        
        # Validate embeddings shape
        if embeddings.ndim != 2:
            raise ValueError(f"Embeddings must be 2D array, got shape {embeddings.shape}")
        
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embeddings dimension {embeddings.shape[1]} doesn't match index dimension {self.dimension}")
        
        if len(embeddings) != len(metadata):
            raise ValueError(f"Number of embeddings ({len(embeddings)}) must match metadata ({len(metadata)})")
        
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            self.index.train(embeddings.astype('float32'))
        
        self.index.add(embeddings.astype('float32'))
        self.metadata.extend(metadata)
        
        print(f"Added {len(embeddings)} vectors. Total: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> Tuple[np.ndarray, List[dict]]:
        query = query_embedding.astype('float32').reshape(1, -1)
        distances, indices = self.index.search(query, k)
        
        results = [self.metadata[idx] for idx in indices[0] if idx < len(self.metadata)]
        return distances[0], results
    
    def save(self, index_path: str, metadata_path: str):
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.index, index_path)
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        print(f"Saved index: {index_path}")
    
    def load(self, index_path: str, metadata_path: str):
        self.index = faiss.read_index(index_path)
        
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        return self
