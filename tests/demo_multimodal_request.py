"""
Demo: How Multimodal RAG actually works
Simulates a user request: "Find a cheap restaurant that serves this [image]"
"""

import os
import sys
from pathlib import Path

# Suppress warnings
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger('transformers').setLevel(logging.ERROR)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vectorstore.retriever import MultimodalRetriever

def hybrid_search_demo():
    print("=" * 80)
    print("DEMO: Advanced Multimodal Search")
    print("User Query: 'Find a cheap restaurant that serves this food'")
    print("Attached Image: [Burger Image]")
    print("=" * 80)
    
    # 1. Initialize
    print("\n1. Initializing System...")
    retriever = MultimodalRetriever(verbose=False)
    retriever.load_indices()
    
    # 2. Simulate Inputs
    query_text = "cheap restaurant"
    query_image = "data/sample_images/burger.png"
    query_image_path = str(project_root / query_image)
    
    # 3. Step 1: Image Search (The "Visual Intent")
    # We search for the food item first because that's the primary constraint
    print(f"\n2. Executing Image Search (Intent: 'serves this food')...")
    # Get more candidates (k=20) to allow for filtering
    image_results = retriever.search_image(query_image_path, k=20)
    
    print(f"   Found {len(image_results)} candidates based on visual similarity.")
    print("   Top 3 Visual Matches:")
    for r in image_results[:3]:
        print(f"   - {r['name']} (Sim: {r['similarity']:.3f})")

    # 4. Step 2: Attribute Filtering (The "Text Constraint")
    # "cheap" usually implies Price Range = 1 or 2
    print(f"\n3. Applying Text Constraints (Intent: '{query_text}')...")
    
    final_results = []
    
    for r in image_results:
        # Check price range (if available) - assuming 'price_range' is in metadata
        # (It was added in the recent update!)
        price = r.get('price_range')
        
        # Logic: If price is low (1 or 2) OR not specified (keep them to be safe)
        is_cheap = price is None or price <= 2
        
        if is_cheap:
            final_results.append(r)
        else:
            print(f"   Filtered out '{r['name']}' (Price: {price})")
            
    # 5. Final Output
    print(f"\n4. Final Recommendations ({len(final_results)} matches):")
    print("-" * 60)
    
    for i, r in enumerate(final_results[:5], 1):
        price_symbol = "$" * r.get('price_range', 0) if r.get('price_range') else "N/A"
        print(f"{i}. {r['name']}")
        print(f"   Matches: Visual (Sim: {r['similarity']:.3f}) + Text (Cheap)")
        print(f"   Cuisine: {r.get('categories', 'N/A')}")
        print(f"   Price: {price_symbol} | Rating: {r.get('rating', 'N/A')}⭐")
        print(f"   Location: {r.get('city', 'N/A')}")
        print()

if __name__ == "__main__":
    hybrid_search_demo()
