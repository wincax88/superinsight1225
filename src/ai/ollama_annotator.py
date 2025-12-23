"""
Ollama AI Annotator for SuperInsight platform.

Integrates with Ollama local models for AI pre-annotation.
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional
from uuid import uuid4
import httpx

from .base import AIAnnotator, ModelConfig, Prediction, AIAnnotationError, ModelType
try:
    from models.task import Task
except ImportError:
    from src.models.task import Task


class OllamaAnnotator(AIAnnotator):
    """AI Annotator using Ollama local models."""
    
    def __init__(self, config: ModelConfig):
        """Initialize Ollama annotator."""
        if config.model_type != ModelType.OLLAMA:
            raise ValueError("ModelConfig must have model_type=OLLAMA")
        super().__init__(config)
        self.client = httpx.AsyncClient(timeout=config.timeout)
    
    def _validate_config(self) -> None:
        """Validate Ollama-specific configuration."""
        if not self.config.base_url:
            raise ValueError("base_url is required for Ollama")
        if not self.config.model_name:
            raise ValueError("model_name is required for Ollama")
    
    async def predict(self, task: Task) -> Prediction:
        """
        Generate prediction using Ollama model.
        
        Args:
            task: The annotation task to predict
            
        Returns:
            Prediction object with results and confidence score
        """
        start_time = time.time()
        
        try:
            # Get document content from task
            # In a real implementation, you'd fetch the document content
            # For now, we'll use a placeholder
            document_content = f"Task {task.id} content"
            
            # Prepare the prompt for annotation
            prompt = self._prepare_annotation_prompt(document_content, task.project_id)
            
            # Make request to Ollama API
            response = await self._make_ollama_request(prompt)
            
            # Parse response and calculate confidence
            prediction_data, confidence = self._parse_ollama_response(response)
            
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
                f"Ollama prediction failed: {str(e)}",
                model_type="ollama",
                task_id=task.id
            )
    
    async def _make_ollama_request(self, prompt: str) -> Dict[str, Any]:
        """Make request to Ollama API."""
        url = f"{self.config.base_url}/api/generate"
        
        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens
            }
        }
        
        try:
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise AIAnnotationError(f"Ollama API request failed: {str(e)}", "ollama")
    
    def _prepare_annotation_prompt(self, content: str, project_id: str) -> str:
        """Prepare annotation prompt based on project type."""
        # This is a simplified prompt. In practice, you'd have project-specific templates
        base_prompt = f"""
请对以下文本进行标注分析：

文本内容：
{content}

项目类型：{project_id}

请以JSON格式返回标注结果，包含以下字段：
- sentiment: 情感倾向 (positive/negative/neutral)
- entities: 实体识别结果
- categories: 分类标签
- confidence: 置信度 (0.0-1.0)

返回格式：
{{"sentiment": "...", "entities": [...], "categories": [...], "confidence": 0.0}}
"""
        return base_prompt
    
    def _parse_ollama_response(self, response: Dict[str, Any]) -> tuple[Dict[str, Any], float]:
        """Parse Ollama response and extract prediction data and confidence."""
        try:
            response_text = response.get("response", "")
            
            # Try to extract JSON from response
            # Look for JSON-like content in the response
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                prediction_data = json.loads(json_str)
                
                # Extract confidence from prediction or use default
                confidence = prediction_data.get("confidence", 0.7)
                
                # Ensure confidence is in valid range
                confidence = max(0.0, min(1.0, float(confidence)))
                
                return prediction_data, confidence
            else:
                # Fallback: create structured response from text
                return {
                    "raw_response": response_text,
                    "sentiment": "neutral",
                    "entities": [],
                    "categories": []
                }, 0.5
                
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback for unparseable responses
            return {
                "raw_response": response.get("response", ""),
                "error": f"Failed to parse response: {str(e)}",
                "sentiment": "neutral",
                "entities": [],
                "categories": []
            }, 0.3
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the Ollama model."""
        return {
            "model_type": "ollama",
            "model_name": self.config.model_name,
            "base_url": self.config.base_url,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "timeout": self.config.timeout
        }
    
    async def list_available_models(self) -> List[str]:
        """List available Ollama models."""
        try:
            url = f"{self.config.base_url}/api/tags"
            response = await self.client.get(url)
            response.raise_for_status()
            
            data = response.json()
            models = [model["name"] for model in data.get("models", [])]
            return models
            
        except Exception as e:
            raise AIAnnotationError(f"Failed to list Ollama models: {str(e)}", "ollama")
    
    async def check_model_availability(self) -> bool:
        """Check if the configured model is available."""
        try:
            available_models = await self.list_available_models()
            return self.config.model_name in available_models
        except Exception:
            return False
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.aclose()