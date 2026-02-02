from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
from typing import List

class ImageEmbedder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", verbose: bool = True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.dimension = 512
        self.model.eval()
        if verbose:
            print(f"Loaded image embedder on {self.device}: {model_name}")
    
    def embed_batch(self, image_paths: List[str], batch_size: int = 32, return_indices: bool = False):
        from tqdm import tqdm
        embeddings = []
        skipped = 0
        all_valid_indices = []
        
        num_batches = (len(image_paths) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(image_paths), batch_size), 
                     total=num_batches,
                     desc="Embedding images",
                     unit="batch"):
            batch_paths = image_paths[i:i+batch_size]
            
            # Load images with error handling
            images = []
            valid_indices = []
            for idx, path in enumerate(batch_paths):
                try:
                    img = Image.open(path)
                    img.verify()  # Verify it's a valid image
                    img = Image.open(path)  # Reopen after verify
                    img = img.convert("RGB")
                    images.append(img)
                    valid_indices.append(i + idx)  # Global index
                except Exception as e:
                    skipped += 1
                    continue
            
            if len(images) == 0:
                continue
            
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            
            # CLIP returns BaseModelOutputWithPooling with pooler_output attribute
            # pooler_output has shape (batch_size, hidden_dim) which is what we need
            if hasattr(image_features, 'pooler_output'):
                embeddings.append(image_features.pooler_output.cpu().numpy())
            else:
                # Fallback for direct tensor output
                embeddings.append(image_features.cpu().numpy())
            
            all_valid_indices.extend(valid_indices)
            
            # Clear GPU cache periodically
            if (i // batch_size) % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if skipped > 0:
            print(f"\n⚠️  Skipped {skipped} corrupted/invalid images")
        
        if len(embeddings) == 0:
            # No valid images
            result = np.array([]).reshape(0, 512)  # Empty array with correct shape
        else:
            # Stack all embeddings into single array
            result = np.vstack(embeddings)
        
        if return_indices:
            return result, all_valid_indices
        return result

    
    def embed(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        
        # Handle BaseModelOutputWithPooling
        if hasattr(image_features, 'pooler_output'):
            return image_features.pooler_output.cpu().numpy()[0]
        else:
            return image_features.cpu().numpy()[0]




