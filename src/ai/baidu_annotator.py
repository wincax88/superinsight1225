"""
Baidu Wenxin AI Annotator for SuperInsight platform.

Integrates with Baidu Wenxin (ERNIE) API for AI pre-annotation.
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


class BaiduAnnotator(AIAnnotator):
    """AI Annotator using Baidu Wenxin (ERNIE) models."""
    
    def __init__(self, config: ModelConfig):
        """Initialize Baidu annotator."""
        if config.model_type != ModelType.BAIDU_WENXIN:
            raise ValueError("ModelConfig must have model_type=BAIDU_WENXIN")
        super().__init__(config)
        
        self.client = httpx.AsyncClient(timeout=config.timeout)
        self.access_token = None
        self.token_expires_at = 0
        
        # Baidu API base URL
        self.api_base = "https://aip.baidubce.com"
    
    def _validate_config(self) -> None:
        """Validate Baidu-specific configuration."""
        if not self.config.api_key:
            raise ValueError("api_key (API Key) is required for Baidu Wenxin")
        # For Baidu, we need both API Key and Secret Key
        # The secret key should be stored in base_url field for this implementation
        if not self.config.base_url:
            raise ValueError("base_url (Secret Key) is required for Baidu Wenxin")
        if not self.config.model_name:
            raise ValueError("model_name is required for Baidu Wenxin")
    
    async def _get_access_token(self) -> str:
        """Get access token for Baidu API."""
        current_time = time.time()
        
        # Check if we have a valid token
        if self.access_token and current_time < self.token_expires_at:
            return self.access_token
        
        # Get new access token
        url = f"{self.api_base}/oauth/2.0/token"
        params = {
            "grant_type": "client_credentials",
            "client_id": self.config.api_key,  # API Key
            "client_secret": self.config.base_url  # Secret Key (stored in base_url)
        }
        
        try:
            response = await self.client.post(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            self.access_token = data["access_token"]
            expires_in = data.get("expires_in", 3600)  # Default 1 hour
            self.token_expires_at = current_time + expires_in - 300  # Refresh 5 minutes early
            
            return self.access_token
            
        except Exception as e:
            raise AIAnnotationError(f"Failed to get Baidu access token: {str(e)}", "baidu_wenxin")
    
    async def predict(self, task: Task) -> Prediction:
        """
        Generate prediction using Baidu Wenxin model.
        
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
            
            # Make request to Baidu API
            response = await self._make_chat_completion_request(messages)
            
            # Parse response and calculate confidence
            prediction_data, confidence = self._parse_baidu_response(response)
            
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
                f"Baidu Wenxin prediction failed: {str(e)}",
                model_type="baidu_wenxin",
                task_id=task.id
            )
    
    async def _make_chat_completion_request(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Make chat completion request to Baidu API."""
        access_token = await self._get_access_token()
        
        # Map model names to Baidu endpoints
        model_endpoints = {
            "ernie-bot": "completions",
            "ernie-bot-turbo": "eb-instant",
            "ernie-bot-4": "completions_pro",
            "ernie-3.5": "completions",
            "ernie-4.0": "completions_pro"
        }
        
        endpoint = model_endpoints.get(self.config.model_name, "completions")
        url = f"{self.api_base}/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/{endpoint}"
        
        params = {"access_token": access_token}
        
        payload = {
            "messages": messages,
            "temperature": self.config.temperature,
            "max_output_tokens": self.config.max_tokens,
            "stream": False
        }
        
        try:
            response = await self.client.post(url, json=payload, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise AIAnnotationError(f"Baidu API request failed: {str(e)}", "baidu_wenxin")
    
    def _prepare_annotation_messages(self, content: str, project_id: str) -> List[Dict[str, str]]:
        """Prepare chat messages for annotation task."""
        system_prompt = self._get_system_prompt(project_id)
        user_prompt = f"""
请对以下文本进行标注分析：

文本内容：
{content}

请严格按照JSON格式返回标注结果，不要包含任何其他文字说明。
"""
        
        # Baidu API format
        messages = [
            {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}
        ]
        
        return messages
    
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
    
    def _parse_baidu_response(self, response: Dict[str, Any]) -> tuple[Dict[str, Any], float]:
        """Parse Baidu response and extract prediction data and confidence."""
        try:
            # Check for API errors
            if "error_code" in response:
                return {
                    "error": f"Baidu API error: {response.get('error_msg', 'Unknown error')}",
                    "error_code": response.get("error_code")
                }, 0.0
            
            result = response.get("result", "")
            if not result:
                return {"error": "No result in response"}, 0.0
            
            # Try to extract JSON from response
            content = result.strip()
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
                    "id": response.get("id", ""),
                    "created": response.get("created", 0)
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
        """Get information about the Baidu Wenxin model."""
        return {
            "model_type": "baidu_wenxin",
            "model_name": self.config.model_name,
            "api_base": self.api_base,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "timeout": self.config.timeout,
            "has_api_key": bool(self.config.api_key),
            "has_secret_key": bool(self.config.base_url)
        }
    
    async def list_available_models(self) -> List[str]:
        """List available Baidu Wenxin models."""
        return [
            "ernie-bot",
            "ernie-bot-turbo", 
            "ernie-bot-4",
            "ernie-3.5",
            "ernie-4.0"
        ]
    
    async def check_model_availability(self) -> bool:
        """Check if the API credentials are valid."""
        try:
            # Try to get access token
            await self._get_access_token()
            return True
        except Exception:
            return False
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.aclose()