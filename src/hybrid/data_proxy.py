"""
混合云数据代理组件
用于在云端和本地环境之间安全传输数据
"""
import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import hashlib
import hmac
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
import requests

logger = logging.getLogger(__name__)

class DataProxy:
    """混合云数据代理"""
    
    def __init__(self):
        self.local_endpoint = os.getenv('LOCAL_ENDPOINT', 'http://localhost:8000')
        self.cloud_endpoint = os.getenv('CLOUD_ENDPOINT')
        self.proxy_key = os.getenv('PROXY_KEY')
        self.encryption_enabled = os.getenv('PROXY_ENCRYPTION', 'true').lower() == 'true'
        
        if not self.cloud_endpoint or not self.proxy_key:
            raise ValueError("混合云配置不完整，需要设置 CLOUD_ENDPOINT 和 PROXY_KEY")
        
        # 初始化加密
        if self.encryption_enabled:
            self.cipher = self._init_cipher()
    
    def _init_cipher(self) -> Fernet:
        """初始化加密器"""
        # 使用 PBKDF2 从密钥派生加密密钥
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'superinsight_hybrid_cloud',
            iterations=100000,
        )
        key = kdf.derive(self.proxy_key.encode())
        return Fernet(key)
    
    def _encrypt_data(self, data: bytes) -> bytes:
        """加密数据"""
        if not self.encryption_enabled:
            return data
        return self.cipher.encrypt(data)
    
    def _decrypt_data(self, data: bytes) -> bytes:
        """解密数据"""
        if not self.encryption_enabled:
            return data
        return self.cipher.decrypt(data)
    
    def _generate_signature(self, data: str, timestamp: int) -> str:
        """生成请求签名"""
        message = f"{data}:{timestamp}:{self.proxy_key}"
        return hmac.new(
            self.proxy_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
    
    def _verify_signature(self, data: str, timestamp: int, signature: str) -> bool:
        """验证请求签名"""
        expected_signature = self._generate_signature(data, timestamp)
        return hmac.compare_digest(expected_signature, signature)
    
    def sync_to_cloud(self, data: Dict[str, Any], data_type: str) -> Dict[str, Any]:
        """同步数据到云端"""
        try:
            # 序列化数据
            data_json = json.dumps(data)
            
            # 加密数据
            encrypted_data = self._encrypt_data(data_json.encode())
            
            # 生成签名
            timestamp = int(datetime.now().timestamp())
            signature = self._generate_signature(data_json, timestamp)
            
            # 发送到云端
            response = requests.post(
                f"{self.cloud_endpoint}/api/hybrid/sync",
                json={
                    'data': encrypted_data.hex(),
                    'data_type': data_type,
                    'timestamp': timestamp,
                    'signature': signature
                },
                headers={
                    'X-Proxy-Key': self.proxy_key,
                    'Content-Type': 'application/json'
                },
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            logger.info(f"数据同步到云端成功: {data_type}")
            return result
            
        except Exception as e:
            logger.error(f"数据同步到云端失败: {e}")
            raise
    
    def sync_from_cloud(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从云端同步数据"""
        try:
            # 生成签名
            query_json = json.dumps(query)
            timestamp = int(datetime.now().timestamp())
            signature = self._generate_signature(query_json, timestamp)
            
            # 从云端获取数据
            response = requests.post(
                f"{self.cloud_endpoint}/api/hybrid/fetch",
                json={
                    'query': query,
                    'timestamp': timestamp,
                    'signature': signature
                },
                headers={
                    'X-Proxy-Key': self.proxy_key,
                    'Content-Type': 'application/json'
                },
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            # 解密数据
            decrypted_data = []
            for item in result.get('data', []):
                encrypted_bytes = bytes.fromhex(item['data'])
                decrypted_bytes = self._decrypt_data(encrypted_bytes)
                decrypted_data.append(json.loads(decrypted_bytes))
            
            logger.info(f"从云端同步数据成功: {len(decrypted_data)} 条")
            return decrypted_data
            
        except Exception as e:
            logger.error(f"从云端同步数据失败: {e}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 检查本地端点
            local_response = requests.get(
                f"{self.local_endpoint}/health",
                timeout=5
            )
            local_healthy = local_response.status_code == 200
            
            # 检查云端端点
            cloud_response = requests.get(
                f"{self.cloud_endpoint}/health",
                headers={'X-Proxy-Key': self.proxy_key},
                timeout=5
            )
            cloud_healthy = cloud_response.status_code == 200
            
            return {
                'local': {
                    'healthy': local_healthy,
                    'endpoint': self.local_endpoint
                },
                'cloud': {
                    'healthy': cloud_healthy,
                    'endpoint': self.cloud_endpoint
                },
                'encryption_enabled': self.encryption_enabled,
                'overall_healthy': local_healthy and cloud_healthy
            }
            
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return {
                'local': {'healthy': False},
                'cloud': {'healthy': False},
                'overall_healthy': False,
                'error': str(e)
            }

class HybridDataSyncService:
    """混合云数据同步服务"""
    
    def __init__(self):
        self.proxy = DataProxy()
        self.sync_interval = int(os.getenv('SYNC_INTERVAL', '300'))  # 默认5分钟
        self.sync_enabled = os.getenv('SYNC_ENABLED', 'true').lower() == 'true'
    
    def sync_annotations(self, annotations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """同步标注数据"""
        if not self.sync_enabled:
            logger.info("数据同步已禁用")
            return {'synced': False, 'reason': 'sync_disabled'}
        
        try:
            result = self.proxy.sync_to_cloud(
                {'annotations': annotations},
                'annotations'
            )
            return {'synced': True, 'count': len(annotations), 'result': result}
        except Exception as e:
            logger.error(f"标注数据同步失败: {e}")
            return {'synced': False, 'error': str(e)}
    
    def sync_quality_issues(self, issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """同步质量问题"""
        if not self.sync_enabled:
            return {'synced': False, 'reason': 'sync_disabled'}
        
        try:
            result = self.proxy.sync_to_cloud(
                {'quality_issues': issues},
                'quality_issues'
            )
            return {'synced': True, 'count': len(issues), 'result': result}
        except Exception as e:
            logger.error(f"质量问题同步失败: {e}")
            return {'synced': False, 'error': str(e)}
    
    def fetch_ai_predictions(self, task_ids: List[str]) -> List[Dict[str, Any]]:
        """从云端获取 AI 预测结果"""
        if not self.sync_enabled:
            return []
        
        try:
            query = {'task_ids': task_ids, 'data_type': 'ai_predictions'}
            predictions = self.proxy.sync_from_cloud(query)
            return predictions
        except Exception as e:
            logger.error(f"获取 AI 预测失败: {e}")
            return []
    
    def fetch_model_updates(self) -> Optional[Dict[str, Any]]:
        """从云端获取模型更新"""
        if not self.sync_enabled:
            return None
        
        try:
            query = {'data_type': 'model_updates', 'latest': True}
            updates = self.proxy.sync_from_cloud(query)
            return updates[0] if updates else None
        except Exception as e:
            logger.error(f"获取模型更新失败: {e}")
            return None

# 全局数据同步服务实例
hybrid_sync_service = HybridDataSyncService()

def get_hybrid_sync_service() -> HybridDataSyncService:
    """获取混合云同步服务实例"""
    return hybrid_sync_service