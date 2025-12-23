"""
腾讯云 TCB CloudBase PostgreSQL 数据库连接配置
"""
import os
from typing import Optional
from sqlalchemy import create_engine, Engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
import logging

logger = logging.getLogger(__name__)

class TCBDatabaseConfig:
    """腾讯云 TCB 数据库配置管理"""
    
    def __init__(self):
        self.host = os.getenv('TCB_DB_HOST', 'postgresql.tencentcloudbase.com')
        self.port = int(os.getenv('TCB_DB_PORT', '5432'))
        self.database = os.getenv('TCB_DB_NAME', 'superinsight')
        self.username = os.getenv('TCB_DB_USER')
        self.password = os.getenv('TCB_DB_PASSWORD')
        self.ssl_mode = os.getenv('TCB_DB_SSL_MODE', 'require')
        
        # 连接池配置
        self.pool_size = int(os.getenv('TCB_DB_POOL_SIZE', '10'))
        self.max_overflow = int(os.getenv('TCB_DB_MAX_OVERFLOW', '20'))
        self.pool_timeout = int(os.getenv('TCB_DB_POOL_TIMEOUT', '30'))
        self.pool_recycle = int(os.getenv('TCB_DB_POOL_RECYCLE', '3600'))
        
    def get_database_url(self) -> str:
        """构建数据库连接 URL"""
        if not self.username or not self.password:
            raise ValueError("数据库用户名和密码必须设置")
            
        return (
            f"postgresql://{self.username}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
            f"?sslmode={self.ssl_mode}"
        )
    
    def create_engine(self) -> Engine:
        """创建数据库引擎"""
        database_url = self.get_database_url()
        
        engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=self.pool_size,
            max_overflow=self.max_overflow,
            pool_timeout=self.pool_timeout,
            pool_recycle=self.pool_recycle,
            pool_pre_ping=True,  # 连接前检查
            echo=os.getenv('TCB_DB_ECHO', 'false').lower() == 'true'
        )
        
        logger.info(f"创建 TCB 数据库引擎: {self.host}:{self.port}/{self.database}")
        return engine
    
    def create_session_factory(self) -> sessionmaker:
        """创建会话工厂"""
        engine = self.create_engine()
        return sessionmaker(bind=engine, expire_on_commit=False)

class TCBDatabaseManager:
    """TCB 数据库管理器"""
    
    def __init__(self):
        self.config = TCBDatabaseConfig()
        self.engine: Optional[Engine] = None
        self.session_factory: Optional[sessionmaker] = None
    
    def initialize(self):
        """初始化数据库连接"""
        try:
            self.engine = self.config.create_engine()
            self.session_factory = self.config.create_session_factory()
            
            # 测试连接
            with self.get_session() as session:
                session.execute(text("SELECT 1"))
                
            logger.info("TCB 数据库连接初始化成功")
            
        except Exception as e:
            logger.error(f"TCB 数据库连接初始化失败: {e}")
            raise
    
    def get_session(self) -> Session:
        """获取数据库会话"""
        if not self.session_factory:
            raise RuntimeError("数据库未初始化，请先调用 initialize()")
        
        return self.session_factory()
    
    def health_check(self) -> bool:
        """数据库健康检查"""
        try:
            with self.get_session() as session:
                result = session.execute(text("SELECT 1")).scalar()
                return result == 1
        except Exception as e:
            logger.error(f"数据库健康检查失败: {e}")
            return False
    
    def close(self):
        """关闭数据库连接"""
        if self.engine:
            self.engine.dispose()
            logger.info("TCB 数据库连接已关闭")

# 全局数据库管理器实例
tcb_db_manager = TCBDatabaseManager()

def get_tcb_database_session() -> Session:
    """获取 TCB 数据库会话的便捷函数"""
    return tcb_db_manager.get_session()

def initialize_tcb_database():
    """初始化 TCB 数据库的便捷函数"""
    tcb_db_manager.initialize()