"""
Simple test script for image-based restaurant search
Uses the existing MultimodalRetriever class
"""

import os
import sys
from pathlib import Path

# Suppress transformers warnings and verbose output
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import warnings
warnings.filterwarnings('ignore')

# Suppress transformers logging
import logging
logging.getLogger('transformers').setLevel(logging.ERROR)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vectorstore.retriever import MultimodalRetriever

def test_image_search():
    """Test image search functionality"""
    
    print("=" * 70)
    print("Image-Based Restaurant Search Test")
    print("=" * 70)
    
    # Initialize retriever
    print("\n1. Initializing retriever...")
    retriever = MultimodalRetriever(verbose=False)
    
    # Load indices
    print("2. Loading indices...")
    try:
        retriever.load_indices()
        print("   ✓ Indices loaded successfully")
    except Exception as e:
        print(f"   ✗ Error loading indices: {e}")
        return
    
    # Test with sample images
    sample_images = [
        "data/sample_images/drink.png",

    ]
    
    for image_path in sample_images:
        full_path = project_root / image_path
        
        if not full_path.exists():
            print(f"\n⚠️  Image not found: {image_path}")
            continue
        
        print(f"\n{'=' * 70}")
        print(f"Searching by image: {image_path}")
        print("=" * 70)
        
        try:
            # Search by image
            results = retriever.search_image(str(full_path), k=5)
            
            if not results:
                print("No results found")
                continue
            
            print(f"\nFound {len(results)} similar restaurants:\n")
            
            for i, result in enumerate(results, 1):
                print(f"{i}. {result.get('name', 'Unknown')}")
                print(f"   Photo ID: {result.get('photo_id', 'N/A')}")
                print(f"   Label: {result.get('label', 'N/A')}")
                print(f"   Cuisine: {result.get('categories', 'N/A')}")
                print(f"   Rating: {result.get('rating', 'N/A')} ⭐")
                
                # Calculate similarity from distance
                if 'distance' in result:
                    similarity = 1 / (1 + result['distance'])
                    print(f"   Similarity: {similarity:.3f}")
                print()
        
        except Exception as e:
            print(f"✗ Error during search: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_image_search()
