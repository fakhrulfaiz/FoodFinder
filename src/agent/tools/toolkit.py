"""
Custom Toolkit for FoodFinder Agent
Manages tool instances and automatically passes shared dependencies.
"""

from typing import List, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from .rag_text_tool import RAGTextTool
from .rag_image_tool import RAGImageTool
from .image_qa_tool import ImageQATool


class CustomToolkit(BaseModel):
    """
    Toolkit that manages FoodFinder agent tools and their dependencies.
    
    Automatically injects shared dependencies (retriever, LLM) into tools
    and provides a clean interface for the agent to access all tools.
    """
    
    retriever: Optional[Any] = Field(
        default=None,
        description="MultimodalRetriever instance for RAG tools"
    )
    llm: Optional[Any] = Field(
        default=None,
        description="Language model instance (optional, for future use)"
    )
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, retriever: Any = None, llm: Any = None, **kwargs):
        """
        Initialize the toolkit with shared dependencies.
        
        Args:
            retriever: MultimodalRetriever instance for RAG operations
            llm: Language model instance (optional)
        """
        super().__init__(retriever=retriever, llm=llm, **kwargs)
    
    def get_tools(self) -> List[BaseTool]:
        """
        Get all tools with dependencies injected.
        
        Returns:
            List of instantiated BaseTool objects ready for use
        """
        tools = []
        
        # Add RAG text tool (requires retriever)
        if self.retriever is not None:
            tools.append(RAGTextTool(retriever=self.retriever))
        
        # Add RAG image tool (requires retriever)
        if self.retriever is not None:
            tools.append(RAGImageTool(retriever=self.retriever))
        
        # Add image QA tool (standalone, uses CLIP)
        tools.append(ImageQATool())
        
        return tools
