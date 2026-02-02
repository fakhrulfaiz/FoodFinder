import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import pickle
from tqdm import tqdm
from metadata_utils import create_text_metadata, create_image_metadata

def rebuild_text_metadata():
    """Rebuild text metadata pickle file without regenerating embeddings."""
    print("\n=== Rebuilding Text Metadata ===")
    
    # Load processed restaurants
    with open("data/processed/restaurants.json", 'r', encoding='utf-8') as f:
        restaurants = json.load(f)
    
    print(f"Loaded {len(restaurants)} restaurants")
    
    # Create metadata
    metadata = []
    for restaurant in tqdm(restaurants, desc="Creating metadata"):
        metadata.append(create_text_metadata(restaurant))
    
    # Save metadata
    output_path = "indexes/text_metadata.pkl"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"✓ Saved {len(metadata)} metadata entries to {output_path}")
    return metadata

def rebuild_image_metadata():
    """Rebuild image metadata pickle file without regenerating embeddings."""
    print("\n=== Rebuilding Image Metadata ===")
    
    # Load processed restaurants
    with open("data/processed/restaurants.json", 'r', encoding='utf-8') as f:
        restaurants = json.load(f)
    
    restaurants_with_photos = [r for r in restaurants if r.get('photos')]
    print(f"Found {len(restaurants_with_photos)} restaurants with photos")
    
    # Create metadata
    metadata = []
    for restaurant in tqdm(restaurants_with_photos, desc="Creating metadata"):
        for photo in restaurant['photos']:
            photo_path = photo['path']
            if Path(photo_path).exists():
                metadata.append(create_image_metadata(restaurant, photo))
    
    # Save metadata
    output_path = "indexes/image_metadata.pkl"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"✓ Saved {len(metadata)} metadata entries to {output_path}")
    return metadata

def verify_metadata(metadata_path: str):
    """Verify metadata structure."""
    print(f"\n=== Verifying {metadata_path} ===")
    
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"Total entries: {len(metadata)}")
    print(f"\nSample entry:")
    print(json.dumps(metadata[0], indent=2, default=str))
    
    # Count fields
    print(f"\nFields in metadata: {len(metadata[0])}")
    print(f"Field names: {list(metadata[0].keys())}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Rebuild metadata pickle files')
    parser.add_argument('--text-only', action='store_true', help='Rebuild only text metadata')
    parser.add_argument('--image-only', action='store_true', help='Rebuild only image metadata')
    parser.add_argument('--verify', action='store_true', help='Verify metadata after rebuilding')
    args = parser.parse_args()
    
    print("="*60)
    print("Rebuilding Metadata Files")
    print("="*60)
    
    if not args.image_only:
        text_metadata = rebuild_text_metadata()
        if args.verify:
            verify_metadata("indexes/text_metadata.pkl")
    
    if not args.text_only:
        image_metadata = rebuild_image_metadata()
        if args.verify:
            verify_metadata("indexes/image_metadata.pkl")
    
    print("\n" + "="*60)
    print("Metadata Rebuild Complete!")
    print("="*60)
    print("\nNote: This only updates metadata (.pkl files)")
    print("The FAISS index files (.faiss) are unchanged")

if __name__ == "__main__":
    main()
