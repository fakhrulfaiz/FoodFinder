"""
Image Question Answering Tool
Answers questions about food/restaurant images using CLIP model for zero-shot understanding.
"""

from typing import Optional, Type, List
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np


class ImageQAInput(BaseModel):
    """Input schema for image QA tool."""
    image_path: str = Field(
        description="Path to the image file to analyze"
    )
    question: str = Field(
        description="Question to answer about the image. Can be about cuisine type, dish description, ingredients, presentation, etc."
    )


class ImageQATool(BaseTool):
    """
    Tool for answering questions about food/restaurant images using CLIP.
    
    Use Case: Direct image understanding without retrieval. Answers questions about
    what's in an image, cuisine type, dish characteristics, ingredients, presentation style, etc.
    Uses CLIP's vision-language capabilities for zero-shot image understanding.
    
    Examples:
    - "What type of cuisine is in this image?"
    - "Describe the dish in this photo"
    - "What ingredients can you see in this food image?"
    - "Is this a formal or casual dining setting?"
    - "What is the main protein in this dish?"
    """
    
    name: str = "image_qa"
    description: str = """Answer questions about food or restaurant images directly.
    Use this tool when the user wants to understand what's in a specific image without 
    searching for similar restaurants. Can identify cuisine types, describe dishes, 
    detect ingredients, and analyze presentation. Does NOT perform retrieval."""
    args_schema: Type[BaseModel] = ImageQAInput
    
    model: Optional[object] = Field(default=None, exclude=True)
    processor: Optional[object] = Field(default=None, exclude=True)
    device: str = Field(default="cpu", exclude=True)
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", **kwargs):
        super().__init__(**kwargs)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
    
    def _generate_answer_candidates(self, question: str) -> List[str]:
        """Generate plausible answer candidates based on the question type."""
        question_lower = question.lower()
        
        # Cuisine type questions
        if any(word in question_lower for word in ['cuisine', 'type of food', 'what kind']):
            return [
                "Italian cuisine", "Chinese cuisine", "Japanese cuisine", "Mexican cuisine",
                "Indian cuisine", "Thai cuisine", "French cuisine", "American cuisine",
                "Mediterranean cuisine", "Korean cuisine", "Vietnamese cuisine", "Greek cuisine"
            ]
        
        # Dish description questions
        elif any(word in question_lower for word in ['describe', 'what is this', 'what dish']):
            return [
                "a pizza", "a pasta dish", "a burger", "a salad", "a soup",
                "a sandwich", "a steak", "seafood", "a dessert", "a rice dish",
                "noodles", "a vegetable dish", "fried food", "grilled food", "baked food"
            ]
        
        # Ingredient questions
        elif any(word in question_lower for word in ['ingredient', 'what\'s in', 'contains']):
            return [
                "vegetables", "meat", "seafood", "cheese", "pasta", "rice",
                "bread", "sauce", "herbs", "spices", "chicken", "beef",
                "pork", "fish", "tofu", "eggs", "mushrooms", "tomatoes"
            ]
        
        # Setting/ambiance questions
        elif any(word in question_lower for word in ['setting', 'dining', 'atmosphere', 'ambiance']):
            return [
                "formal dining", "casual dining", "fast food", "fine dining",
                "outdoor seating", "indoor seating", "modern decor", "traditional decor",
                "cozy atmosphere", "elegant setting"
            ]
        
        # Protein questions
        elif any(word in question_lower for word in ['protein', 'main ingredient']):
            return [
                "chicken", "beef", "pork", "fish", "shrimp", "tofu",
                "lamb", "turkey", "duck", "seafood", "no protein (vegetarian)"
            ]
        
        # Default general answers
        else:
            return [
                "yes", "no", "possibly", "likely", "unlikely",
                "a food dish", "a restaurant setting", "a beverage",
                "multiple items", "unclear from image"
            ]
    
    def _run(self, image_path: str, question: str) -> str:
        """Answer a question about the image using CLIP."""
        try:
            # Load and process image
            image = Image.open(image_path).convert("RGB")
            
            # Generate answer candidates based on question
            candidates = self._generate_answer_candidates(question)
            
            # Prepare inputs for CLIP
            inputs = self.processor(
                text=candidates,
                images=image,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get similarity scores
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
            
            # Get top 3 answers
            top_probs, top_indices = torch.topk(probs[0], min(3, len(candidates)))
            
            # Format response
            response = f"Question: {question}\n\n"
            response += "Answer based on image analysis:\n"
            
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices), 1):
                confidence = prob.item() * 100
                answer = candidates[idx.item()]
                response += f"{i}. {answer} (confidence: {confidence:.1f}%)\n"
            
            # Add most likely answer as primary response
            best_answer = candidates[top_indices[0].item()]
            best_confidence = top_probs[0].item() * 100
            
            if best_confidence > 50:
                response += f"\nMost likely: {best_answer}"
            else:
                response += f"\nNote: Low confidence in all answers. The image may not clearly show the requested information."
            
            return response
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return f"Error analyzing image: {str(e)}\n\nDetails:\n{error_details}"
    
    async def _arun(self, image_path: str, question: str) -> str:
        """Async version - delegates to sync for now."""
        return self._run(image_path, question)
