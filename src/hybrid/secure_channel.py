"""
混合云安全传输通道
提供端到端加密和安全认证
"""
import os
import ssl
import socket
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import jwt
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

class SecureChannel:
    """安全传输通道"""
    
    def __init__(self):
        self.private_key_path = os.getenv('HYBRID_PRIVATE_KEY_PATH')
        self.public_key_path = os.getenv('HYBRID_PUBLIC_KEY_PATH')
        self.ca_cert_path = os.getenv('HYBRID_CA_CERT_PATH')
        self.client_cert_path = os.getenv('HYBRID_CLIENT_CERT_PATH')
        self.client_key_path = os.getenv('HYBRID_CLIENT_KEY_PATH')
        
        # JWT 配置
        self.jwt_secret = os.getenv('HYBRID_JWT_SECRET')
        self.jwt_algorithm = 'HS256'
        self.jwt_expiry = int(os.getenv('HYBRID_JWT_EXPIRY', '3600'))  # 1小时
        
        # 初始化密钥
        self.private_key = self._load_private_key()
        self.public_key = self._load_public_key()
        
        # 配置 SSL 上下文
        self.ssl_context = self._create_ssl_context()
        
        # 配置 HTTP 会话
        self.session = self._create_secure_session()
    
    def _load_private_key(self):
        """加载私钥"""
        if not self.private_key_path or not os.path.exists(self.private_key_path):
            logger.warning("私钥文件不存在，将生成新的密钥对")
            return self._generate_key_pair()
        
        try:
            with open(self.private_key_path, 'rb') as f:
                private_key = serialization.load_pem_private_key(
                    f.read(),
                    password=None,
                    backend=default_backend()
                )
            return private_key
        except Exception as e:
            logger.error(f"加载私钥失败: {e}")
            raise
    
    def _load_public_key(self):
        """加载公钥"""
        if not self.public_key_path or not os.path.exists(self.public_key_path):
            if self.private_key:
                return self.private_key.public_key()
            return None
        
        try:
            with open(self.public_key_path, 'rb') as f:
                public_key = serialization.load_pem_public_key(
                    f.read(),
                    backend=default_backend()
                )
            return public_key
        except Exception as e:
            logger.error(f"加载公钥失败: {e}")
            raise
    
    def _generate_key_pair(self):
        """生成新的密钥对"""
        try:
            # 生成私钥
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
            
            # 保存私钥
            if self.private_key_path:
                os.makedirs(os.path.dirname(self.private_key_path), exist_ok=True)
                with open(self.private_key_path, 'wb') as f:
                    f.write(private_key.private_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PrivateFormat.PKCS8,
                        encryption_algorithm=serialization.NoEncryption()
                    ))
            
            # 保存公钥
            if self.public_key_path:
                public_key = private_key.public_key()
                os.makedirs(os.path.dirname(self.public_key_path), exist_ok=True)
                with open(self.public_key_path, 'wb') as f:
                    f.write(public_key.public_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PublicFormat.SubjectPublicKeyInfo
                    ))
            
            logger.info("生成新的密钥对成功")
            return private_key
            
        except Exception as e:
            logger.error(f"生成密钥对失败: {e}")
            raise
    
    def _create_ssl_context(self) -> ssl.SSLContext:
        """创建 SSL 上下文"""
        context = ssl.create_default_context()
        
        # 配置 CA 证书
        if self.ca_cert_path and os.path.exists(self.ca_cert_path):
            context.load_verify_locations(self.ca_cert_path)
        
        # 配置客户端证书
        if (self.client_cert_path and os.path.exists(self.client_cert_path) and
            self.client_key_path and os.path.exists(self.client_key_path)):
            context.load_cert_chain(self.client_cert_path, self.client_key_path)
        
        # 安全配置
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        
        return context
    
    def _create_secure_session(self) -> requests.Session:
        """创建安全的 HTTP 会话"""
        session = requests.Session()
        
        # 配置重试策略
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # 配置 SSL
        session.verify = self.ca_cert_path if self.ca_cert_path else True
        if self.client_cert_path and self.client_key_path:
            session.cert = (self.client_cert_path, self.client_key_path)
        
        return session
    
    def encrypt_data(self, data: bytes, recipient_public_key: bytes) -> Dict[str, str]:
        """使用混合加密方式加密数据"""
        try:
            # 生成随机 AES 密钥
            aes_key = os.urandom(32)  # 256-bit key
            iv = os.urandom(16)  # 128-bit IV
            
            # 使用 AES 加密数据
            cipher = Cipher(
                algorithms.AES(aes_key),
                modes.CBC(iv),
                backend=default_backend()
            )
            encryptor = cipher.encryptor()
            
            # 填充数据到块大小的倍数
            padding_length = 16 - (len(data) % 16)
            padded_data = data + bytes([padding_length] * padding_length)
            
            encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
            
            # 使用 RSA 加密 AES 密钥
            public_key = serialization.load_pem_public_key(
                recipient_public_key,
                backend=default_backend()
            )
            
            encrypted_key = public_key.encrypt(
                aes_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            return {
                'encrypted_data': encrypted_data.hex(),
                'encrypted_key': encrypted_key.hex(),
                'iv': iv.hex()
            }
            
        except Exception as e:
            logger.error(f"数据加密失败: {e}")
            raise
    
    def decrypt_data(self, encrypted_package: Dict[str, str]) -> bytes:
        """解密数据"""
        try:
            # 解密 AES 密钥
            encrypted_key = bytes.fromhex(encrypted_package['encrypted_key'])
            aes_key = self.private_key.decrypt(
                encrypted_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            # 解密数据
            encrypted_data = bytes.fromhex(encrypted_package['encrypted_data'])
            iv = bytes.fromhex(encrypted_package['iv'])
            
            cipher = Cipher(
                algorithms.AES(aes_key),
                modes.CBC(iv),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()
            
            padded_data = decryptor.update(encrypted_data) + decryptor.finalize()
            
            # 移除填充
            padding_length = padded_data[-1]
            data = padded_data[:-padding_length]
            
            return data
            
        except Exception as e:
            logger.error(f"数据解密失败: {e}")
            raise
    
    def generate_jwt_token(self, payload: Dict[str, Any]) -> str:
        """生成 JWT 令牌"""
        if not self.jwt_secret:
            raise ValueError("JWT 密钥未配置")
        
        # 添加标准声明
        now = datetime.utcnow()
        payload.update({
            'iat': now,
            'exp': now + timedelta(seconds=self.jwt_expiry),
            'iss': 'superinsight-hybrid'
        })
        
        token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
        return token
    
    def verify_jwt_token(self, token: str) -> Dict[str, Any]:
        """验证 JWT 令牌"""
        if not self.jwt_secret:
            raise ValueError("JWT 密钥未配置")
        
        try:
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=[self.jwt_algorithm],
                issuer='superinsight-hybrid'
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise ValueError("令牌已过期")
        except jwt.InvalidTokenError as e:
            raise ValueError(f"无效令牌: {e}")
    
    def secure_request(self, method: str, url: str, **kwargs) -> requests.Response:
        """发送安全请求"""
        try:
            # 生成认证令牌
            token = self.generate_jwt_token({
                'method': method,
                'url': url,
                'timestamp': datetime.utcnow().isoformat()
            })
            
            # 添加认证头
            headers = kwargs.get('headers', {})
            headers['Authorization'] = f'Bearer {token}'
            kwargs['headers'] = headers
            
            # 发送请求
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            
            return response
            
        except Exception as e:
            logger.error(f"安全请求失败: {e}")
            raise
    
    def test_connection(self, endpoint: str) -> Dict[str, Any]:
        """测试连接"""
        try:
            response = self.secure_request('GET', f"{endpoint}/api/hybrid/test")
            
            return {
                'success': True,
                'status_code': response.status_code,
                'response_time': response.elapsed.total_seconds(),
                'ssl_version': getattr(response.raw._connection.sock, 'version', None)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

# 全局安全通道实例
secure_channel = SecureChannel()

def get_secure_channel() -> SecureChannel:
    """获取安全通道实例"""
    return secure_channel