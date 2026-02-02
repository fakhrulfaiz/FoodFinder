"""
Comprehensive test script for FoodFinder RAG Agent
Tests both text-based and image-based restaurant search
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vectorstore.retriever import MultimodalRetriever

def print_separator(title=""):
    """Print a formatted separator"""
    print("\n" + "=" * 70)
    if title:
        print(f" {title}")
        print("=" * 70)

def test_text_search(retriever):
    """Test text-based restaurant search"""
    print_separator("TEXT-BASED SEARCH")
    
    queries = [
        "Italian restaurants with good pizza",
        "Best sushi places",
        "Cheap burger joints",
        "Fine dining French cuisine"
    ]
    
    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 70)
        
        try:
            results = retriever.search_text(query, k=3)
            
            if not results:
                print("   No results found")
                continue
            
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result.get('name', 'Unknown')}")
                print(f"   Cuisine: {result.get('categories', 'N/A')}")
                print(f"   Rating: {result.get('rating', 'N/A')} ⭐")
                
                location_parts = []
                if result.get('city'):
                    location_parts.append(result['city'])
                if result.get('state'):
                    location_parts.append(result['state'])
                if location_parts:
                    print(f"   Location: {', '.join(location_parts)}")
                
                if result.get('address'):
                    print(f"   Address: {result['address']}")
        
        except Exception as e:
            print(f"   ✗ Error: {e}")

def test_image_search(retriever):
    """Test image-based restaurant search"""
    print_separator("IMAGE-BASED SEARCH")
    
    sample_images = [
        ("data/sample_images/burger.png", "🍔 Burger"),
        ("data/sample_images/pizza.png", "🍕 Pizza"),
        ("data/sample_images/sushi.png", "🍣 Sushi")
    ]
    
    for image_path, description in sample_images:
        full_path = project_root / image_path
        
        if not full_path.exists():
            print(f"\n⚠️  {description} image not found: {image_path}")
            continue
        
        print(f"\n🖼️  Searching by image: {description}")
        print("-" * 70)
        
        try:
            results = retriever.search_image(str(full_path), k=3)
            
            if not results:
                print("   No results found")
                continue
            
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result.get('name', 'Unknown')}")
                print(f"   Photo ID: {result.get('photo_id', 'N/A')}")
                print(f"   Label: {result.get('label', 'N/A')}")
                print(f"   Cuisine: {result.get('categories', 'N/A')}")
                print(f"   Rating: {result.get('rating', 'N/A')} ⭐")
                
                if 'similarity' in result:
                    print(f"   Similarity: {result['similarity']:.3f}")
        
        except Exception as e:
            print(f"   ✗ Error: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Run all tests"""
    print_separator("FOODFINDER MULTIMODAL SEARCH TEST")
    
    # Initialize retriever
    print("\n📦 Initializing retriever...")
    retriever = MultimodalRetriever()
    
    # Load indices
    print("📂 Loading indices...")
    try:
        retriever.load_indices()
        print("   ✓ Indices loaded successfully")
    except Exception as e:
        print(f"   ✗ Error loading indices: {e}")
        return
    
    # Run text search tests
    test_text_search(retriever)
    
    # Run image search tests
    test_image_search(retriever)
    
    print_separator("TEST COMPLETE")
    print("\n✅ All tests finished!")

if __name__ == "__main__":
    main()
