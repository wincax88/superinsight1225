"""
ChatGLM Open Source Model Annotator for SuperInsight platform.

Integrates with ChatGLM models via local deployment or API endpoints.
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


class ChatGLMAnnotator(AIAnnotator):
    """AI Annotator using ChatGLM models (local or API deployment)."""
    
    def __init__(self, config: ModelConfig):
        """Initialize ChatGLM annotator."""
        # Use HUGGINGFACE type for ChatGLM as it's an open source model
        if config.model_type not in [ModelType.HUGGINGFACE, ModelType.OLLAMA]:
            raise ValueError("ChatGLM should use HUGGINGFACE or OLLAMA model_type")
        super().__init__(config)
        
        # Set up API client
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        
        self.client = httpx.AsyncClient(
            headers=headers,
            timeout=config.timeout
        )
        
        # Default to local deployment endpoint
        self.api_base = self.config.base_url or "http://localhost:8000"
    
    def _validate_config(self) -> None:
        """Validate ChatGLM-specific configuration."""
        if not self.config.model_name:
            raise ValueError("model_name is required for ChatGLM")
        
        # Validate model name is a ChatGLM variant
        valid_models = [
            "chatglm3-6b", "chatglm2-6b", "chatglm-6b",
            "chatglm3-6b-32k", "chatglm2-6b-32k",
            "chatglm3-6b-base", "chatglm2-6b-base"
        ]
        if not any(model in self.config.model_name.lower() for model in valid_models):
            print(f"Warning: {self.config.model_name} may not be a valid ChatGLM model")
    
    async def predict(self, task: Task) -> Prediction:
        """
        Generate prediction using ChatGLM model.
        
        Args:
            task: The annotation task to predict
            
        Returns:
            Prediction object with results and confidence score
        """
        start_time = time.time()
        
        try:
            # Get document content from task
            document_content = f"Task {task.id} content"
            
            # Prepare messages for chat completion
            messages = self._prepare_annotation_messages(document_content, task.project_id)
            
            # Make request to ChatGLM API
            response = await self._make_chat_completion_request(messages)
            
            # Parse response and calculate confidence
            prediction_data, confidence = self._parse_chatglm_response(response)
            
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
                f"ChatGLM prediction failed: {str(e)}",
                model_type="chatglm",
                task_id=task.id
            )
    
    async def _make_chat_completion_request(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Make chat completion request to ChatGLM API."""
        # Try OpenAI-compatible API first
        url = f"{self.api_base}/v1/chat/completions"
        
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False
        }
        
        try:
            response = await self.client.post(url, json=payload)
            if response.status_code == 200:
                return response.json()
        except httpx.HTTPError:
            pass  # Try alternative endpoint
        
        # Try ChatGLM-specific API format
        url = f"{self.api_base}/api/chat"
        
        # Convert messages to ChatGLM format
        history = []
        query = ""
        
        for msg in messages:
            if msg["role"] == "user":
                query = msg["content"]
            elif msg["role"] == "assistant":
                if query:
                    history.append([query, msg["content"]])
                    query = ""
        
        payload = {
            "query": query,
            "history": history,
            "temperature": self.config.temperature,
            "max_length": self.config.max_tokens
        }
        
        try:
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise AIAnnotationError(f"ChatGLM API request failed: {str(e)}", "chatglm")
    
    def _prepare_annotation_messages(self, content: str, project_id: str) -> List[Dict[str, str]]:
        """Prepare chat messages for annotation task."""
        system_prompt = self._get_system_prompt(project_id)
        user_prompt = f"""
请对以下文本进行标注分析：

文本内容：
{content}

请严格按照JSON格式返回标注结果，不要包含任何其他文字说明。
"""
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def _get_system_prompt(self, project_id: str) -> str:
        """Get system prompt based on project type."""
        if "sentiment" in project_id.lower():
            return """你是一个专业的情感分析专家。请分析文本的情感倾向，返回JSON格式：
{"sentiment": "positive/negative/neutral", "confidence": 0.0-1.0, "reasoning": "分析理由"}"""
        
        elif "ner" in project_id.lower() or "entity" in project_id.lower():
            return """你是一个专业的命名实体识别专家。请识别文本中的实体，返回JSON格式：
{"entities": [{"text": "实体文本", "label": "实体类型", "start": 起始位置, "end": 结束位置, "confidence": 0.0-1.0}], "confidence": 0.0-1.0}"""
        
        elif "classification" in project_id.lower():
            return """你是一个专业的文本分类专家。请对文本进行分类，返回JSON格式：
{"category": "分类标签", "confidence": 0.0-1.0, "all_categories": [{"label": "标签", "score": 0.0-1.0}]}"""
        
        else:
            return """你是一个专业的文本标注专家。请对文本进行综合分析，返回JSON格式：
{"sentiment": "情感倾向", "entities": [], "categories": [], "confidence": 0.0-1.0, "summary": "分析摘要"}"""
    
    def _parse_chatglm_response(self, response: Dict[str, Any]) -> tuple[Dict[str, Any], float]:
        """Parse ChatGLM response and extract prediction data and confidence."""
        try:
            content = ""
            
            # Handle OpenAI-compatible format
            if "choices" in response:
                choices = response.get("choices", [])
                if choices:
                    message = choices[0].get("message", {})
                    content = message.get("content", "")
            
            # Handle ChatGLM-specific format
            elif "response" in response:
                content = response.get("response", "")
            
            # Handle direct text response
            elif isinstance(response, str):
                content = response
            
            if not content:
                return {"error": "No content in response"}, 0.0
            
            # Try to extract JSON from response
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            try:
                prediction_data = json.loads(content)
                
                # Extract confidence from prediction or use default
                confidence = prediction_data.get("confidence", 0.75)
                confidence = max(0.0, min(1.0, float(confidence)))
                
                # Add metadata
                prediction_data["model_response"] = {
                    "usage": response.get("usage", {}),
                    "model": self.config.model_name,
                    "api_base": self.api_base
                }
                
                return prediction_data, confidence
                
            except json.JSONDecodeError:
                # Fallback: create structured response from text
                return {
                    "raw_response": content,
                    "sentiment": "neutral",
                    "entities": [],
                    "categories": [],
                    "model_response": {"model": self.config.model_name}
                }, 0.5
                
        except Exception as e:
            return {
                "error": f"Failed to parse response: {str(e)}",
                "raw_response": str(response)
            }, 0.2
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the ChatGLM model."""
        return {
            "model_type": "chatglm",
            "model_name": self.config.model_name,
            "api_base": self.api_base,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "timeout": self.config.timeout,
            "has_api_key": bool(self.config.api_key),
            "deployment_type": "local" if "localhost" in self.api_base else "remote"
        }
    
    async def list_available_models(self) -> List[str]:
        """List available ChatGLM models."""
        return [
            "chatglm3-6b",
            "chatglm3-6b-32k",
            "chatglm3-6b-base",
            "chatglm2-6b",
            "chatglm2-6b-32k",
            "chatglm2-6b-base",
            "chatglm-6b"
        ]
    
    async def check_model_availability(self) -> bool:
        """Check if the ChatGLM model is accessible."""
        try:
            # Try health check endpoint
            health_url = f"{self.api_base}/health"
            response = await self.client.get(health_url)
            if response.status_code == 200:
                return True
        except Exception:
            pass
        
        try:
            # Try a simple test request
            test_messages = [
                {"role": "user", "content": "测试"}
            ]
            
            response = await self._make_chat_completion_request(test_messages)
            return "error" not in str(response).lower()
            
        except Exception:
            return False
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.aclose()