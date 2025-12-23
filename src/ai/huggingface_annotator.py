"""
HuggingFace AI Annotator for SuperInsight platform.

Integrates with HuggingFace Transformers for AI pre-annotation.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Union
from uuid import uuid4
import httpx

from .base import AIAnnotator, ModelConfig, Prediction, AIAnnotationError, ModelType
try:
    from models.task import Task
except ImportError:
    from src.models.task import Task


class HuggingFaceAnnotator(AIAnnotator):
    """AI Annotator using HuggingFace models."""
    
    def __init__(self, config: ModelConfig):
        """Initialize HuggingFace annotator."""
        if config.model_type != ModelType.HUGGINGFACE:
            raise ValueError("ModelConfig must have model_type=HUGGINGFACE")
        super().__init__(config)
        
        # Set up API client
        headers = {}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        
        self.client = httpx.AsyncClient(
            headers=headers,
            timeout=config.timeout
        )
        
        # HuggingFace Inference API base URL
        self.api_base = "https://api-inference.huggingface.co/models"
    
    def _validate_config(self) -> None:
        """Validate HuggingFace-specific configuration."""
        if not self.config.model_name:
            raise ValueError("model_name is required for HuggingFace")
        # API key is optional for public models
    
    async def predict(self, task: Task) -> Prediction:
        """
        Generate prediction using HuggingFace model.
        
        Args:
            task: The annotation task to predict
            
        Returns:
            Prediction object with results and confidence score
        """
        start_time = time.time()
        
        try:
            # Get document content from task
            # In a real implementation, you'd fetch the document content
            document_content = f"Task {task.id} content"
            
            # Determine task type and make appropriate API call
            prediction_data, confidence = await self._make_prediction_request(
                document_content, task.project_id
            )
            
            processing_time = time.time() - start_time
            
            return Prediction(
                id=uuid4(),
                task_id=task.id,
                ai_model_config=self.config,
                prediction_data=prediction_data,
                confidence=confidence,
                processing_time=processing_time
            )
            
        except Exception as e:
            raise AIAnnotationError(
                f"HuggingFace prediction failed: {str(e)}",
                model_type="huggingface",
                task_id=task.id
            )
    
    async def _make_prediction_request(self, content: str, project_id: str) -> tuple[Dict[str, Any], float]:
        """Make prediction request based on project type."""
        # Determine the type of task based on project_id or model_name
        if "sentiment" in project_id.lower() or "sentiment" in self.config.model_name.lower():
            return await self._sentiment_analysis(content)
        elif "ner" in project_id.lower() or "token" in self.config.model_name.lower():
            return await self._named_entity_recognition(content)
        elif "classification" in project_id.lower() or "classify" in self.config.model_name.lower():
            return await self._text_classification(content)
        else:
            # Default to text generation
            return await self._text_generation(content)
    
    async def _sentiment_analysis(self, text: str) -> tuple[Dict[str, Any], float]:
        """Perform sentiment analysis."""
        url = f"{self.api_base}/{self.config.model_name}"
        
        payload = {"inputs": text}
        
        try:
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            
            results = response.json()
            
            if isinstance(results, list) and len(results) > 0:
                # Get the highest confidence prediction
                best_result = max(results[0], key=lambda x: x.get("score", 0))
                
                prediction_data = {
                    "sentiment": best_result.get("label", "UNKNOWN"),
                    "all_scores": results[0],
                    "task_type": "sentiment_analysis"
                }
                
                confidence = best_result.get("score", 0.5)
                return prediction_data, confidence
            else:
                return {"error": "No results returned", "task_type": "sentiment_analysis"}, 0.0
                
        except httpx.HTTPError as e:
            raise AIAnnotationError(f"HuggingFace API request failed: {str(e)}", "huggingface")
    
    async def _named_entity_recognition(self, text: str) -> tuple[Dict[str, Any], float]:
        """Perform named entity recognition."""
        url = f"{self.api_base}/{self.config.model_name}"
        
        payload = {"inputs": text}
        
        try:
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            
            results = response.json()
            
            if isinstance(results, list):
                entities = []
                total_confidence = 0.0
                
                for entity in results:
                    entities.append({
                        "text": entity.get("word", ""),
                        "label": entity.get("entity", ""),
                        "confidence": entity.get("score", 0.0),
                        "start": entity.get("start", 0),
                        "end": entity.get("end", 0)
                    })
                    total_confidence += entity.get("score", 0.0)
                
                avg_confidence = total_confidence / len(results) if results else 0.0
                
                prediction_data = {
                    "entities": entities,
                    "task_type": "named_entity_recognition"
                }
                
                return prediction_data, avg_confidence
            else:
                return {"error": "No results returned", "task_type": "named_entity_recognition"}, 0.0
                
        except httpx.HTTPError as e:
            raise AIAnnotationError(f"HuggingFace API request failed: {str(e)}", "huggingface")
    
    async def _text_classification(self, text: str) -> tuple[Dict[str, Any], float]:
        """Perform text classification."""
        url = f"{self.api_base}/{self.config.model_name}"
        
        payload = {"inputs": text}
        
        try:
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            
            results = response.json()
            
            if isinstance(results, list) and len(results) > 0:
                # Get the highest confidence prediction
                best_result = max(results[0], key=lambda x: x.get("score", 0))
                
                prediction_data = {
                    "classification": best_result.get("label", "UNKNOWN"),
                    "all_scores": results[0],
                    "task_type": "text_classification"
                }
                
                confidence = best_result.get("score", 0.5)
                return prediction_data, confidence
            else:
                return {"error": "No results returned", "task_type": "text_classification"}, 0.0
                
        except httpx.HTTPError as e:
            raise AIAnnotationError(f"HuggingFace API request failed: {str(e)}", "huggingface")
    
    async def _text_generation(self, text: str) -> tuple[Dict[str, Any], float]:
        """Perform text generation."""
        url = f"{self.api_base}/{self.config.model_name}"
        
        payload = {
            "inputs": text,
            "parameters": {
                "max_new_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "return_full_text": False
            }
        }
        
        try:
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            
            results = response.json()
            
            if isinstance(results, list) and len(results) > 0:
                generated_text = results[0].get("generated_text", "")
                
                prediction_data = {
                    "generated_text": generated_text,
                    "input_text": text,
                    "task_type": "text_generation"
                }
                
                # For text generation, confidence is harder to determine
                # Use a moderate confidence score
                confidence = 0.7
                return prediction_data, confidence
            else:
                return {"error": "No results returned", "task_type": "text_generation"}, 0.0
                
        except httpx.HTTPError as e:
            raise AIAnnotationError(f"HuggingFace API request failed: {str(e)}", "huggingface")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the HuggingFace model."""
        return {
            "model_type": "huggingface",
            "model_name": self.config.model_name,
            "api_base": self.api_base,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "timeout": self.config.timeout,
            "has_api_key": bool(self.config.api_key)
        }
    
    async def get_model_details(self) -> Dict[str, Any]:
        """Get detailed information about the model from HuggingFace."""
        try:
            # Use HuggingFace Hub API to get model info
            url = f"https://huggingface.co/api/models/{self.config.model_name}"
            response = await self.client.get(url)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            raise AIAnnotationError(f"Failed to get model details: {str(e)}", "huggingface")
    
    async def check_model_availability(self) -> bool:
        """Check if the model is available and accessible."""
        try:
            await self.get_model_details()
            return True
        except Exception:
            return False
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.aclose()