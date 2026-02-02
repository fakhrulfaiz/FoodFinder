"""
Simple test script for the FoodFinder RAG agent

This demonstrates how to use the build_graph() function and run queries.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agent.main_agent import FoodFinderAgent


def main():
    """Test the FoodFinder RAG agent"""
    
    print("=" * 60)
    print("FoodFinder RAG Agent Test")
    print("=" * 60)
    
    # Initialize agent
    print("\n1. Initializing agent...")
    agent = FoodFinderAgent(model_name="gpt-4o-mini", temperature=0)
    
    # Build the graph
    print("2. Building LangGraph...")
    graph = agent.build_graph()
    print("   ✓ Graph built successfully!")
    
    # Test query
    test_query = "Find me highly rated Italian restaurants"
    print(f"\n3. Running query: '{test_query}'")
    print("-" * 60)
    
    try:
        # Run the agent
        result = agent.run(test_query)
        
        print("\n📍 Agent Response:")
        print(result["answer"])
        
        print("\n" + "=" * 60)
        print("✓ Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nNote: Make sure you have:")
        print("  1. Built the FAISS indices (run scripts/build_indices.py)")
        print("  2. Set up your OpenAI API key in .env file")
        print("  3. Installed all required dependencies")


def test_streaming():
    """Test streaming mode"""
    print("\n" + "=" * 60)
    print("Testing Streaming Mode")
    print("=" * 60)
    
    agent = FoodFinderAgent()
    agent.build_graph()
    
    query = "Recommend pizza places with outdoor seating"
    print(f"\nQuery: '{query}'")
    print("-" * 60)
    
    try:
        for chunk in agent.stream(query):
            for node, update in chunk.items():
                print(f"\n📦 Update from node: {node}")
                if "messages" in update and update["messages"]:
                    last_msg = update["messages"][-1]
                    print(f"   Type: {type(last_msg).__name__}")
                    if hasattr(last_msg, 'content'):
                        content_preview = str(last_msg.content)[:100]
                        print(f"   Content: {content_preview}...")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    # Run basic test
    main()
    
    # Uncomment to test streaming
    # test_streaming()
