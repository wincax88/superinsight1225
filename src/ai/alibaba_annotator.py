"""
Alibaba Tongyi Qianwen AI Annotator for SuperInsight platform.

Integrates with Alibaba Cloud Tongyi Qianwen API for AI pre-annotation.
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


class AlibabaAnnotator(AIAnnotator):
    """AI Annotator using Alibaba Tongyi Qianwen models."""
    
    def __init__(self, config: ModelConfig):
        """Initialize Alibaba annotator."""
        if config.model_type != ModelType.ALIBABA_TONGYI:
            raise ValueError("ModelConfig must have model_type=ALIBABA_TONGYI")
        super().__init__(config)
        
        # Set up API client with authentication
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
        
        # Alibaba Tongyi API base URL
        self.api_base = self.config.base_url or "https://dashscope.aliyuncs.com/api/v1"
    
    def _validate_config(self) -> None:
        """Validate Alibaba-specific configuration."""
        if not self.config.api_key:
            raise ValueError("api_key is required for Alibaba Tongyi")
        if not self.config.model_name:
            raise ValueError("model_name is required for Alibaba Tongyi")
    
    async def predict(self, task: Task) -> Prediction:
        """
        Generate prediction using Alibaba Tongyi model.
        
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
            
            # Make request to Alibaba API
            response = await self._make_chat_completion_request(messages)
            
            # Parse response and calculate confidence
            prediction_data, confidence = self._parse_alibaba_response(response)
            
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
                f"Alibaba Tongyi prediction failed: {str(e)}",
                model_type="alibaba_tongyi",
                task_id=task.id
            )
    
    async def _make_chat_completion_request(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Make chat completion request to Alibaba API."""
        url = f"{self.api_base}/services/aigc/text-generation/generation"
        
        payload = {
            "model": self.config.model_name,
            "input": {
                "messages": messages
            },
            "parameters": {
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "result_format": "message"
            }
        }
        
        try:
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise AIAnnotationError(f"Alibaba API request failed: {str(e)}", "alibaba_tongyi")
    
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
    
    def _parse_alibaba_response(self, response: Dict[str, Any]) -> tuple[Dict[str, Any], float]:
        """Parse Alibaba response and extract prediction data and confidence."""
        try:
            # Check for API errors
            if "code" in response and response["code"] != "200":
                return {
                    "error": f"Alibaba API error: {response.get('message', 'Unknown error')}",
                    "error_code": response.get("code")
                }, 0.0
            
            output = response.get("output", {})
            choices = output.get("choices", [])
            
            if not choices:
                return {"error": "No choices in response"}, 0.0
            
            message = choices[0].get("message", {})
            content = message.get("content", "")
            
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
                confidence = prediction_data.get("confidence", 0.8)
                confidence = max(0.0, min(1.0, float(confidence)))
                
                # Add metadata
                prediction_data["model_response"] = {
                    "usage": response.get("usage", {}),
                    "request_id": response.get("request_id", ""),
                    "finish_reason": choices[0].get("finish_reason", "")
                }
                
                return prediction_data, confidence
                
            except json.JSONDecodeError:
                # Fallback: create structured response from text
                return {
                    "raw_response": content,
                    "sentiment": "neutral",
                    "entities": [],
                    "categories": [],
                    "model_response": response.get("usage", {})
                }, 0.6
                
        except Exception as e:
            return {
                "error": f"Failed to parse response: {str(e)}",
                "raw_response": str(response)
            }, 0.2
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the Alibaba Tongyi model."""
        return {
            "model_type": "alibaba_tongyi",
            "model_name": self.config.model_name,
            "api_base": self.api_base,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "timeout": self.config.timeout,
            "has_api_key": bool(self.config.api_key)
        }
    
    async def list_available_models(self) -> List[str]:
        """List available Alibaba Tongyi models."""
        return [
            "qwen-turbo",
            "qwen-plus",
            "qwen-max",
            "qwen-max-1201",
            "qwen-max-longcontext",
            "qwen1.5-72b-chat",
            "qwen1.5-14b-chat",
            "qwen1.5-7b-chat"
        ]
    
    async def check_model_availability(self) -> bool:
        """Check if the API key is valid and model is accessible."""
        try:
            # Make a simple test request
            test_messages = [
                {"role": "user", "content": "测试"}
            ]
            
            url = f"{self.api_base}/services/aigc/text-generation/generation"
            payload = {
                "model": self.config.model_name,
                "input": {
                    "messages": test_messages
                },
                "parameters": {
                    "max_tokens": 10
                }
            }
            
            response = await self.client.post(url, json=payload)
            return response.status_code == 200
            
        except Exception:
            return False
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.aclose()