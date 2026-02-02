import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from dotenv import load_dotenv
import json
from tqdm import tqdm
from src.vectorstore.faiss_index import FAISSIndex
from src.embeddings.text_embedder import TextEmbedder
from src.embeddings.image_embedder import ImageEmbedder
from scripts.metadata_utils import create_text_metadata, create_image_metadata

def load_restaurants(data_path: str = "data/processed/restaurants.json"):
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def build_text_index(restaurants: list):
    print("\n=== Building Text Index ===")
    
    embedder = TextEmbedder()
    index = FAISSIndex(dimension=768, index_type="Flat")
    
    texts = []
    metadata = []
    
    for restaurant in tqdm(restaurants, desc="Preparing texts"):
        text = f"{restaurant['name']}. {restaurant.get('categories', '')}."
        texts.append(text)
        # Use comprehensive metadata from metadata_utils
        metadata.append(create_text_metadata(restaurant))
    
    print("Generating text embeddings...")
    embeddings = embedder.embed_batch(texts, batch_size=64)
    
    print("Adding to index...")
    index.add(embeddings, metadata)
    
    print("Saving index...")
    index.save("indexes/text_index.faiss", "indexes/text_metadata.pkl")
    
    print(f"✓ Text index built with {len(restaurants)} restaurants")
    return index

def build_image_index(restaurants: list, limit: int = None):
    print("\n=== Building Image Index ===")
    
    restaurants_with_photos = [r for r in restaurants if r.get('photos')]
    print(f"Found {len(restaurants_with_photos)} restaurants with photos")
    
    if len(restaurants_with_photos) == 0:
        print("⚠️  No photos found, skipping image index")
        return None
    
    embedder = ImageEmbedder()
    index = FAISSIndex(dimension=512, index_type="Flat")
    
    image_paths = []
    metadata = []
    
    for restaurant in tqdm(restaurants_with_photos, desc="Collecting images"):
        for photo in restaurant['photos']:
            photo_path = photo['path']
            if Path(photo_path).exists():
                image_paths.append(photo_path)
                metadata.append({
                    'id': restaurant['business_id'],
                    'name': restaurant['name'],
                    'photo_id': photo['photo_id'],
                    'label': photo.get('label', ''),
                    'categories': restaurant.get('categories', ''),
                    'rating': restaurant.get('stars', 0),
                    'latitude': restaurant.get('latitude'),
                    'longitude': restaurant.get('longitude'),
                    'address': restaurant.get('address', ''),
                    'city': restaurant.get('city', ''),
                    'state': restaurant.get('state', '')
                })
    
    print(f"Found {len(image_paths)} valid image files")
    
    # Apply limit if specified
    if limit and limit > 0:
        image_paths = image_paths[:limit]
        metadata = metadata[:limit]
        print(f"Limiting to {len(image_paths)} images for testing")
    
    if len(image_paths) == 0:
        print("⚠️  No valid image files found, skipping image index")
        return None
    
    print("Generating image embeddings (this may take a while)...")
    embeddings, valid_indices = embedder.embed_batch(image_paths, batch_size=32, return_indices=True)
    
    # Filter metadata to only include successfully embedded images
    valid_metadata = [metadata[i] for i in valid_indices]
    
    print("Adding to index...")
    index.add(embeddings, valid_metadata)
    
    print("Saving index...")
    index.save("indexes/image_index.faiss", "indexes/image_metadata.pkl")
    
    print(f"✓ Image index built with {len(valid_metadata)} photos ({len(image_paths) - len(valid_metadata)} skipped)")
    return index

def main():
    import argparse
    
    load_dotenv()
    
    parser = argparse.ArgumentParser(description='Build FAISS indices')
    parser.add_argument('--text-only', action='store_true', help='Build only text index')
    parser.add_argument('--image-only', action='store_true', help='Build only image index')
    parser.add_argument('--sample', action='store_true', help='Use sample dataset')
    parser.add_argument('--force', action='store_true', help='Force rebuild even if indices exist')
    parser.add_argument('--limit', type=int, help='Limit number of images to process (for testing)')
    args = parser.parse_args()
    
    print("="*60)
    print("Building FAISS Indices")
    print("="*60)
    
    # Check if indices already exist
    text_index_exists = Path("indexes/text_index.faiss").exists()
    image_index_exists = Path("indexes/image_index.faiss").exists()
    
    if not args.force:
        if text_index_exists and not args.image_only:
            print("\n✓ Text index already exists (use --force to rebuild)")
            if not args.text_only:
                print("  Skipping text index, will build image index only...")
                args.text_only = False
            else:
                print("  Nothing to do!")
                return
        
        if image_index_exists and not args.text_only:
            print("\n✓ Image index already exists (use --force to rebuild)")
            if not args.image_only:
                print("  Skipping image index, will build text index only...")
                args.image_only = False
            else:
                print("  Nothing to do!")
                return
    
    data_path = "data/processed/restaurants_sample.json" if args.sample else "data/processed/restaurants.json"
    
    print(f"\nLoading restaurants from {data_path}...")
    restaurants = load_restaurants(data_path)
    print(f"Loaded {len(restaurants)} restaurants")
    
    if not args.image_only and (args.force or not text_index_exists):
        text_index = build_text_index(restaurants)
    
    if not args.text_only and (args.force or not image_index_exists):
        image_index = build_image_index(restaurants, limit=args.limit)
    
    print("\n" + "="*60)
    print("Index Building Complete!")
    print("="*60)
    print("\nOutput files:")
    if not args.image_only:
        status = "✓ Built" if (args.force or not text_index_exists) else "✓ Exists"
        print(f"  {status}: indexes/text_index.faiss")
        print(f"  {status}: indexes/text_metadata.pkl")
    if not args.text_only:
        status = "✓ Built" if (args.force or not image_index_exists) else "✓ Exists"
        print(f"  {status}: indexes/image_index.faiss")
        print(f"  {status}: indexes/image_metadata.pkl")

if __name__ == "__main__":
    main()
