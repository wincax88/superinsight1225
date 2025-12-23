"""
腾讯云混元大模型集成
"""
import os
import json
import hmac
import hashlib
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import requests
from urllib.parse import urlencode
import logging

from .base import BaseAnnotator
from ..models.annotation import Annotation, Prediction

logger = logging.getLogger(__name__)

class HunyuanAnnotator(BaseAnnotator):
    """腾讯云混元大模型标注器"""
    
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv('HUNYUAN_API_KEY')
        self.secret_key = os.getenv('HUNYUAN_SECRET_KEY')
        self.region = os.getenv('HUNYUAN_REGION', 'ap-beijing')
        self.endpoint = os.getenv('HUNYUAN_ENDPOINT', 'https://hunyuan.tencentcloudapi.com')
        self.model = os.getenv('HUNYUAN_MODEL', 'hunyuan-lite')
        
        if not self.api_key or not self.secret_key:
            raise ValueError("混元 API 密钥未配置")
    
    def _generate_signature(self, params: Dict[str, Any], timestamp: int) -> str:
        """生成腾讯云 API 签名"""
        # 构建签名字符串
        sorted_params = sorted(params.items())
        query_string = urlencode(sorted_params)
        
        string_to_sign = f"POST\nhunyuan.tencentcloudapi.com\n/\n{query_string}"
        
        # 计算签名
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            string_to_sign.encode('utf-8'),
            hashlib.sha1
        ).hexdigest()
        
        return signature
    
    def _make_request(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """发送 API 请求"""
        timestamp = int(time.time())
        
        # 构建请求参数
        request_params = {
            'Action': action,
            'Region': self.region,
            'Timestamp': timestamp,
            'Nonce': int(time.time() * 1000),
            'SecretId': self.api_key,
            'Version': '2023-09-01',
            **params
        }
        
        # 生成签名
        signature = self._generate_signature(request_params, timestamp)
        request_params['Signature'] = signature
        
        # 发送请求
        try:
            response = requests.post(
                self.endpoint,
                data=request_params,
                headers={
                    'Content-Type': 'application/x-www-form-urlencoded',
                    'User-Agent': 'SuperInsight/1.0'
                },
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            
            if 'Error' in result.get('Response', {}):
                error = result['Response']['Error']
                raise Exception(f"混元 API 错误: {error['Code']} - {error['Message']}")
            
            return result.get('Response', {})
            
        except requests.exceptions.RequestException as e:
            logger.error(f"混元 API 请求失败: {e}")
            raise
    
    def predict(self, text: str, task_type: str = "classification", **kwargs) -> Prediction:
        """使用混元模型进行预测"""
        try:
            # 构建提示词
            prompt = self._build_prompt(text, task_type, **kwargs)
            
            # 调用混元 API
            params = {
                'Model': self.model,
                'Messages': [
                    {
                        'Role': 'user',
                        'Content': prompt
                    }
                ],
                'Temperature': kwargs.get('temperature', 0.7),
                'TopP': kwargs.get('top_p', 0.9),
                'MaxTokens': kwargs.get('max_tokens', 1000)
            }
            
            response = self._make_request('ChatCompletions', params)
            
            # 解析响应
            choices = response.get('Choices', [])
            if not choices:
                raise Exception("混元模型返回空响应")
            
            content = choices[0].get('Message', {}).get('Content', '')
            
            # 解析预测结果
            prediction_data = self._parse_prediction(content, task_type)
            
            # 计算置信度
            confidence = self._calculate_confidence(response, prediction_data)
            
            return Prediction(
                model_name=f"hunyuan-{self.model}",
                prediction_data=prediction_data,
                confidence=confidence,
                raw_response=response,
                created_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"混元模型预测失败: {e}")
            raise
    
    def batch_predict(self, texts: List[str], task_type: str = "classification", **kwargs) -> List[Prediction]:
        """批量预测"""
        predictions = []
        
        for text in texts:
            try:
                prediction = self.predict(text, task_type, **kwargs)
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"批量预测中单个文本失败: {e}")
                # 创建失败的预测结果
                predictions.append(Prediction(
                    model_name=f"hunyuan-{self.model}",
                    prediction_data={"error": str(e)},
                    confidence=0.0,
                    raw_response={},
                    created_at=datetime.now()
                ))
        
        return predictions
    
    def _build_prompt(self, text: str, task_type: str, **kwargs) -> str:
        """构建提示词"""
        if task_type == "classification":
            labels = kwargs.get('labels', ['正面', '负面', '中性'])
            prompt = f"""
请对以下文本进行分类，从给定的标签中选择最合适的一个：

标签选项：{', '.join(labels)}

文本内容：{text}

请以 JSON 格式返回结果，包含以下字段：
- label: 选择的标签
- confidence: 置信度 (0-1)
- reasoning: 分类理由

示例格式：
{{"label": "正面", "confidence": 0.85, "reasoning": "文本表达了积极的情感"}}
"""
        
        elif task_type == "ner":
            entities = kwargs.get('entities', ['人名', '地名', '机构名'])
            prompt = f"""
请对以下文本进行命名实体识别，识别出指定类型的实体：

实体类型：{', '.join(entities)}

文本内容：{text}

请以 JSON 格式返回结果，包含以下字段：
- entities: 实体列表，每个实体包含 text, label, start, end
- confidence: 整体置信度 (0-1)

示例格式：
{{"entities": [{{"text": "北京", "label": "地名", "start": 0, "end": 2}}], "confidence": 0.9}}
"""
        
        elif task_type == "sentiment":
            prompt = f"""
请分析以下文本的情感倾向：

文本内容：{text}

请以 JSON 格式返回结果，包含以下字段：
- sentiment: 情感标签 (positive/negative/neutral)
- score: 情感强度 (-1 到 1)
- confidence: 置信度 (0-1)

示例格式：
{{"sentiment": "positive", "score": 0.7, "confidence": 0.85}}
"""
        
        else:
            # 通用任务
            instruction = kwargs.get('instruction', '请分析以下文本')
            prompt = f"""
{instruction}

文本内容：{text}

请以 JSON 格式返回分析结果。
"""
        
        return prompt
    
    def _parse_prediction(self, content: str, task_type: str) -> Dict[str, Any]:
        """解析预测结果"""
        try:
            # 尝试解析 JSON
            if content.strip().startswith('{'):
                return json.loads(content)
            
            # 如果不是 JSON，尝试提取 JSON 部分
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            # 如果无法解析为 JSON，返回原始内容
            return {"raw_output": content, "parsed": False}
            
        except json.JSONDecodeError:
            logger.warning(f"无法解析混元模型输出为 JSON: {content}")
            return {"raw_output": content, "parsed": False}
    
    def _calculate_confidence(self, response: Dict[str, Any], prediction_data: Dict[str, Any]) -> float:
        """计算置信度"""
        # 如果预测结果中包含置信度，直接使用
        if 'confidence' in prediction_data:
            return float(prediction_data['confidence'])
        
        # 否则基于模型响应计算置信度
        usage = response.get('Usage', {})
        total_tokens = usage.get('TotalTokens', 0)
        
        # 简单的启发式方法：更多的 token 通常意味着更详细的回答
        if total_tokens > 100:
            return 0.8
        elif total_tokens > 50:
            return 0.6
        else:
            return 0.4
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "name": f"hunyuan-{self.model}",
            "provider": "tencent_cloud",
            "version": "2023-09-01",
            "region": self.region,
            "endpoint": self.endpoint,
            "capabilities": [
                "text_classification",
                "named_entity_recognition", 
                "sentiment_analysis",
                "text_generation"
            ]
        }
    
    def health_check(self) -> bool:
        """健康检查"""
        try:
            # 发送简单的测试请求
            test_prediction = self.predict("测试文本", "classification", labels=["测试"])
            return test_prediction.confidence >= 0.0
        except Exception as e:
            logger.error(f"混元模型健康检查失败: {e}")
            return False