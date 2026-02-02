from typing import Optional, Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool


class RAGImageInput(BaseModel):
    """Input schema for RAG image retrieval tool."""
    image_path: str = Field(
        description="Path to the image file for similarity search. Can be a photo of food, restaurant interior, or ambiance."
    )
    k: int = Field(
        default=5,
        description="Number of similar restaurant results to return (default: 5)"
    )


class RAGImageTool(BaseTool):
    """
    Tool for retrieving similar restaurants based on image similarity.
    
    Use Case: Find restaurants with similar food, ambiance, or style based on an image.
    Uses visual embeddings to match restaurants with similar characteristics.
    
    Examples:
    - "Find restaurants similar to this dish photo"
    - "Restaurants with ambiance like this image"
    - "Places that serve food that looks like this"
    - "Similar restaurant interiors to this picture"
    """
    
    name: str = "rag_image_search"
    description: str = """Search for restaurants using image similarity. 
    Use this tool when the user provides an image and wants to find restaurants with 
    similar food, ambiance, or style. The tool uses visual embeddings to find matches.
    Returns restaurant information with similarity scores."""
    args_schema: Type[BaseModel] = RAGImageInput
    
    retriever: Optional[object] = Field(default=None, exclude=True)
    _indices_loaded: bool = False
    
    def __init__(self, retriever=None, **kwargs):
        super().__init__(**kwargs)
        self.retriever = retriever
        self._indices_loaded = False
    
    def _run(self, image_path: str, k: int = 5) -> str:
        """Execute the image similarity search."""
        try:
            # Load indices once
            if not self._indices_loaded and self.retriever:
                self.retriever.load_indices()
                self._indices_loaded = True
            
            # Search for similar restaurants
            results = self.retriever.search_image(image_path, k=k)
            
            if not results:
                return "No similar restaurants found for the provided image."
            
            # Return results as JSON string (includes similarity scores in metadata)
            import json
            return json.dumps(results, indent=2)
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return f"Error retrieving similar restaurants: {str(e)}\n\nDetails:\n{error_details}"
    
    async def _arun(self, image_path: str, k: int = 5) -> str:
        """Async version - delegates to sync for now."""
        return self._run(image_path, k)
