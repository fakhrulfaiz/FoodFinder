from pathlib import Path
from typing import Optional
import os
from .faiss_index import FAISSIndex
from ..embeddings.text_embedder import TextEmbedder
from ..embeddings.image_embedder import ImageEmbedder

class MultimodalRetriever:
    def __init__(self, index_dir: str = "indexes", verbose: bool = True):
        # Get project root (3 levels up from this file: src/vectorstore/retriever.py)
        project_root = Path(__file__).parent.parent.parent
        
        # Resolve index_dir relative to project root if it's not absolute
        if Path(index_dir).is_absolute():
            self.index_dir = Path(index_dir)
        else:
            self.index_dir = project_root / index_dir
        
        self.text_embedder = TextEmbedder(verbose=verbose)
        self.image_embedder = ImageEmbedder(verbose=verbose)
        
        self.text_index = FAISSIndex(dimension=768)
        self.image_index = FAISSIndex(dimension=512)
    
    def search_text(self, query: str, k: int = 10):
        embedding = self.text_embedder.embed(query)
        distances, results = self.text_index.search(embedding, k)
        return results
    
    def search_image(self, image_path: str, k: int = 10):
        embedding = self.image_embedder.embed(image_path)
        distances, results = self.image_index.search(embedding, k)
        
        # Add distance and similarity to each result
        for i, result in enumerate(results):
            if i < len(distances):
                result['distance'] = float(distances[i])
                result['similarity'] = float(1 / (1 + distances[i]))
        
        return results
    
    def save_indices(self):
        self.text_index.save(
            str(self.index_dir / "text_index.faiss"),
            str(self.index_dir / "text_metadata.pkl")
        )
        self.image_index.save(
            str(self.index_dir / "image_index.faiss"),
            str(self.index_dir / "image_metadata.pkl")
        )
    
    def load_indices(self):
        text_index_path = str(self.index_dir / "text_index.faiss")
        text_metadata_path = str(self.index_dir / "text_metadata.pkl")
        image_index_path = str(self.index_dir / "image_index.faiss")
        image_metadata_path = str(self.index_dir / "image_metadata.pkl")
        
        # Verify text index exists (required)
        if not Path(text_index_path).exists():
            raise FileNotFoundError(
                f"Text index not found at: {text_index_path}\n"
                f"Index directory: {self.index_dir}\n"
                f"Please run: python scripts/build_indices.py"
            )
        
        # Load text index (required)
        self.text_index.load(text_index_path, text_metadata_path)
        
        # Load image index (optional)
        if Path(image_index_path).exists():
            self.image_index.load(image_index_path, image_metadata_path)
        
        return self


