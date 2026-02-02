"""
RAG Text Retrieval Tool
Searches for restaurants based on text queries using the multimodal retriever.
"""

from typing import Optional, Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool


class RAGTextInput(BaseModel):
    """Input schema for RAG text retrieval tool."""
    query: str = Field(
        description="The text search query for finding restaurants. Examples: 'Italian restaurants in Philadelphia', 'Best pizza places with high ratings', 'Sushi restaurants near downtown'"
    )
    k: int = Field(
        default=5,
        description="Number of restaurant results to return (default: 5)"
    )


class RAGTextTool(BaseTool):
    """
    Tool for retrieving restaurant information based on text queries.
    
    Use Case: Search for restaurants using natural language queries about cuisine type,
    location, ratings, price range, or other text-based criteria.
    
    Examples:
    - "Find Italian restaurants in Philadelphia"
    - "Best pizza places with high ratings"
    - "Affordable sushi restaurants"
    - "Restaurants with outdoor seating"
    """
    
    name: str = "rag_text_search"
    description: str = """Search for restaurants using text-based queries. 
    Use this tool when the user asks about restaurants by cuisine type, location, 
    ratings, price, or other text-based criteria. Returns detailed restaurant information 
    including name, cuisine, rating, location, and address."""
    args_schema: Type[BaseModel] = RAGTextInput
    
    retriever: Optional[object] = Field(default=None, exclude=True)
    _indices_loaded: bool = False
    
    def __init__(self, retriever=None, **kwargs):
        super().__init__(**kwargs)
        self.retriever = retriever
        self._indices_loaded = False
    
    def _run(self, query: str, k: int = 5) -> str:
        """Execute the text search."""
        try:
            # Load indices once
            if not self._indices_loaded and self.retriever:
                self.retriever.load_indices()
                self._indices_loaded = True
            
            # Search for restaurants
            results = self.retriever.search_text(query, k=k)
            
            if not results:
                return "No restaurants found matching your query."
            
            # Return results as JSON string
            import json
            return json.dumps(results, indent=2)
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return f"Error retrieving restaurants: {str(e)}\n\nDetails:\n{error_details}"
    
    async def _arun(self, query: str, k: int = 5) -> str:
        """Async version - delegates to sync for now."""
        return self._run(query, k)
