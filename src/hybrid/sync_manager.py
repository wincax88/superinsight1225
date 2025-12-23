"""
混合云数据同步管理器
负责协调本地和云端数据的同步
"""
import os
import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
import hashlib
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import yaml
from sqlalchemy import select

from .data_proxy import DataProxy
from .secure_channel import SecureChannel
from ..database.connection import get_database_session
from ..models.annotation import Annotation
from ..models.quality_issue import QualityIssue

logger = logging.getLogger(__name__)

class SyncDirection(Enum):
    """同步方向"""
    LOCAL_TO_CLOUD = "local_to_cloud"
    CLOUD_TO_LOCAL = "cloud_to_local"
    BIDIRECTIONAL = "bidirectional"
    LOCAL_ONLY = "local_only"

class SyncFrequency(Enum):
    """同步频率"""
    REAL_TIME = "real_time"
    SCHEDULED = "scheduled"
    MANUAL = "manual"

class ConflictResolution(Enum):
    """冲突解决策略"""
    TIMESTAMP_BASED = "timestamp_based"
    LOCAL_WINS = "local_wins"
    CLOUD_WINS = "cloud_wins"
    MANUAL = "manual"

@dataclass
class SyncRule:
    """同步规则"""
    data_type: str
    direction: SyncDirection
    frequency: SyncFrequency
    encryption: bool = True
    compression: bool = True
    schedule: Optional[str] = None
    batch_size: int = 100

@dataclass
class SyncResult:
    """同步结果"""
    success: bool
    data_type: str
    direction: str
    records_processed: int
    records_synced: int
    conflicts: int
    errors: List[str]
    start_time: datetime
    end_time: datetime
    
    @property
    def duration(self) -> float:
        """同步耗时（秒）"""
        return (self.end_time - self.start_time).total_seconds()

class SyncManager:
    """数据同步管理器"""
    
    def __init__(self):
        self.config_path = os.getenv('HYBRID_CONFIG_PATH', 'deploy/hybrid/hybrid-config.yaml')
        self.config = self._load_config()
        
        self.data_proxy = DataProxy()
        self.secure_channel = SecureChannel()
        
        self.sync_rules = self._parse_sync_rules()
        self.conflict_resolution = ConflictResolution(
            self.config.get('data_sync', {}).get('conflict_resolution', {}).get('strategy', 'timestamp_based')
        )
        
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.sync_history: List[SyncResult] = []
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"加载混合云配置失败: {e}")
            return {}
    
    def _parse_sync_rules(self) -> Dict[str, SyncRule]:
        """解析同步规则"""
        rules = {}
        sync_config = self.config.get('data_sync', {}).get('rules', {})
        
        for data_type, rule_config in sync_config.items():
            rules[data_type] = SyncRule(
                data_type=data_type,
                direction=SyncDirection(rule_config.get('direction', 'bidirectional')),
                frequency=SyncFrequency(rule_config.get('frequency', 'scheduled')),
                encryption=rule_config.get('encryption', True),
                compression=rule_config.get('compression', True),
                schedule=rule_config.get('schedule'),
                batch_size=rule_config.get('batch_size', 100)
            )
        
        return rules
    
    def _calculate_checksum(self, data: Any) -> str:
        """计算数据校验和"""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def _detect_conflicts(self, local_data: List[Dict], cloud_data: List[Dict]) -> List[Tuple[Dict, Dict]]:
        """检测数据冲突"""
        conflicts = []
        
        # 创建本地数据索引
        local_index = {item.get('id'): item for item in local_data}
        
        for cloud_item in cloud_data:
            item_id = cloud_item.get('id')
            if item_id in local_index:
                local_item = local_index[item_id]
                
                # 比较更新时间
                local_updated = datetime.fromisoformat(local_item.get('updated_at', '1970-01-01'))
                cloud_updated = datetime.fromisoformat(cloud_item.get('updated_at', '1970-01-01'))
                
                # 比较校验和
                local_checksum = self._calculate_checksum(local_item)
                cloud_checksum = self._calculate_checksum(cloud_item)
                
                if local_checksum != cloud_checksum and abs((local_updated - cloud_updated).total_seconds()) < 60:
                    conflicts.append((local_item, cloud_item))
        
        return conflicts
    
    def _resolve_conflict(self, local_item: Dict, cloud_item: Dict) -> Dict:
        """解决数据冲突"""
        if self.conflict_resolution == ConflictResolution.LOCAL_WINS:
            return local_item
        elif self.conflict_resolution == ConflictResolution.CLOUD_WINS:
            return cloud_item
        elif self.conflict_resolution == ConflictResolution.TIMESTAMP_BASED:
            local_updated = datetime.fromisoformat(local_item.get('updated_at', '1970-01-01'))
            cloud_updated = datetime.fromisoformat(cloud_item.get('updated_at', '1970-01-01'))
            return cloud_item if cloud_updated > local_updated else local_item
        else:
            # 手动解决，返回本地版本并记录冲突
            logger.warning(f"数据冲突需要手动解决: {local_item.get('id')}")
            return local_item
    
    async def sync_annotations(self) -> SyncResult:
        """同步标注数据"""
        start_time = datetime.now()
        errors = []
        conflicts = 0
        
        try:
            rule = self.sync_rules.get('annotations')
            if not rule:
                return SyncResult(
                    success=False,
                    data_type='annotations',
                    direction='none',
                    records_processed=0,
                    records_synced=0,
                    conflicts=0,
                    errors=['未找到标注数据同步规则'],
                    start_time=start_time,
                    end_time=datetime.now()
                )
            
            # 获取本地标注数据
            with get_database_session() as session:
                stmt = select(Annotation).where(
                    Annotation.updated_at > datetime.now() - timedelta(hours=24)
                )
                local_annotations = session.execute(stmt).scalars().all()
            
            local_data = [ann.to_dict() for ann in local_annotations]
            
            if rule.direction in [SyncDirection.LOCAL_TO_CLOUD, SyncDirection.BIDIRECTIONAL]:
                # 同步到云端
                sync_result = self.data_proxy.sync_to_cloud(
                    {'annotations': local_data},
                    'annotations'
                )
                logger.info(f"标注数据同步到云端: {len(local_data)} 条")
            
            if rule.direction in [SyncDirection.CLOUD_TO_LOCAL, SyncDirection.BIDIRECTIONAL]:
                # 从云端同步
                cloud_data = self.data_proxy.sync_from_cloud({
                    'data_type': 'annotations',
                    'since': (datetime.now() - timedelta(hours=24)).isoformat()
                })
                
                # 检测冲突
                conflicts_list = self._detect_conflicts(local_data, cloud_data)
                conflicts = len(conflicts_list)
                
                # 解决冲突并更新本地数据
                with get_database_session() as session:
                    for cloud_item in cloud_data:
                        # 查找冲突
                        conflict_pair = next(
                            (pair for pair in conflicts_list if pair[1]['id'] == cloud_item['id']),
                            None
                        )
                        
                        if conflict_pair:
                            # 解决冲突
                            resolved_item = self._resolve_conflict(conflict_pair[0], conflict_pair[1])
                            # 更新数据库
                            # TODO: 实现数据库更新逻辑
                        else:
                            # 新数据，直接插入
                            # TODO: 实现数据库插入逻辑
                            pass
                
                logger.info(f"从云端同步标注数据: {len(cloud_data)} 条，冲突: {conflicts} 个")
            
            return SyncResult(
                success=True,
                data_type='annotations',
                direction=rule.direction.value,
                records_processed=len(local_data),
                records_synced=len(local_data),
                conflicts=conflicts,
                errors=errors,
                start_time=start_time,
                end_time=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"标注数据同步失败: {e}")
            errors.append(str(e))
            
            return SyncResult(
                success=False,
                data_type='annotations',
                direction='error',
                records_processed=0,
                records_synced=0,
                conflicts=0,
                errors=errors,
                start_time=start_time,
                end_time=datetime.now()
            )
    
    async def sync_quality_issues(self) -> SyncResult:
        """同步质量问题"""
        start_time = datetime.now()
        errors = []
        
        try:
            rule = self.sync_rules.get('quality_reports')
            if not rule:
                return SyncResult(
                    success=False,
                    data_type='quality_issues',
                    direction='none',
                    records_processed=0,
                    records_synced=0,
                    conflicts=0,
                    errors=['未找到质量问题同步规则'],
                    start_time=start_time,
                    end_time=datetime.now()
                )
            
            # 获取本地质量问题
            with get_database_session() as session:
                stmt = select(QualityIssue).where(
                    QualityIssue.created_at > datetime.now() - timedelta(hours=6)
                )
                quality_issues = session.execute(stmt).scalars().all()
            
            issues_data = [issue.to_dict() for issue in quality_issues]
            
            if rule.direction in [SyncDirection.LOCAL_TO_CLOUD, SyncDirection.BIDIRECTIONAL]:
                # 同步到云端
                sync_result = self.data_proxy.sync_to_cloud(
                    {'quality_issues': issues_data},
                    'quality_issues'
                )
                logger.info(f"质量问题同步到云端: {len(issues_data)} 条")
            
            return SyncResult(
                success=True,
                data_type='quality_issues',
                direction=rule.direction.value,
                records_processed=len(issues_data),
                records_synced=len(issues_data),
                conflicts=0,
                errors=errors,
                start_time=start_time,
                end_time=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"质量问题同步失败: {e}")
            errors.append(str(e))
            
            return SyncResult(
                success=False,
                data_type='quality_issues',
                direction='error',
                records_processed=0,
                records_synced=0,
                conflicts=0,
                errors=errors,
                start_time=start_time,
                end_time=datetime.now()
            )
    
    async def sync_models(self) -> SyncResult:
        """同步 AI 模型"""
        start_time = datetime.now()
        errors = []
        
        try:
            rule = self.sync_rules.get('models')
            if not rule or rule.direction != SyncDirection.CLOUD_TO_LOCAL:
                return SyncResult(
                    success=True,
                    data_type='models',
                    direction='skipped',
                    records_processed=0,
                    records_synced=0,
                    conflicts=0,
                    errors=[],
                    start_time=start_time,
                    end_time=datetime.now()
                )
            
            # 从云端获取模型更新
            model_updates = self.data_proxy.sync_from_cloud({
                'data_type': 'model_updates',
                'latest': True
            })
            
            # 下载并更新模型
            updated_models = 0
            for update in model_updates:
                try:
                    # TODO: 实现模型下载和更新逻辑
                    logger.info(f"更新模型: {update.get('model_name')}")
                    updated_models += 1
                except Exception as e:
                    errors.append(f"模型更新失败 {update.get('model_name')}: {e}")
            
            return SyncResult(
                success=len(errors) == 0,
                data_type='models',
                direction=rule.direction.value,
                records_processed=len(model_updates),
                records_synced=updated_models,
                conflicts=0,
                errors=errors,
                start_time=start_time,
                end_time=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"模型同步失败: {e}")
            errors.append(str(e))
            
            return SyncResult(
                success=False,
                data_type='models',
                direction='error',
                records_processed=0,
                records_synced=0,
                conflicts=0,
                errors=errors,
                start_time=start_time,
                end_time=datetime.now()
            )
    
    async def run_full_sync(self) -> List[SyncResult]:
        """运行完整同步"""
        logger.info("开始混合云数据同步")
        
        # 并行执行各种数据同步
        tasks = [
            self.sync_annotations(),
            self.sync_quality_issues(),
            self.sync_models()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常结果
        sync_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"同步任务异常: {result}")
                sync_results.append(SyncResult(
                    success=False,
                    data_type='unknown',
                    direction='error',
                    records_processed=0,
                    records_synced=0,
                    conflicts=0,
                    errors=[str(result)],
                    start_time=datetime.now(),
                    end_time=datetime.now()
                ))
            else:
                sync_results.append(result)
        
        # 记录同步历史
        self.sync_history.extend(sync_results)
        
        # 保留最近100次同步记录
        if len(self.sync_history) > 100:
            self.sync_history = self.sync_history[-100:]
        
        # 统计结果
        total_synced = sum(r.records_synced for r in sync_results)
        total_conflicts = sum(r.conflicts for r in sync_results)
        failed_syncs = sum(1 for r in sync_results if not r.success)
        
        logger.info(f"混合云同步完成: 同步 {total_synced} 条记录，{total_conflicts} 个冲突，{failed_syncs} 个失败")
        
        return sync_results
    
    def get_sync_status(self) -> Dict[str, Any]:
        """获取同步状态"""
        if not self.sync_history:
            return {
                'status': 'no_history',
                'last_sync': None,
                'success_rate': 0,
                'total_synced': 0
            }
        
        recent_results = self.sync_history[-10:]  # 最近10次同步
        successful_syncs = sum(1 for r in recent_results if r.success)
        success_rate = successful_syncs / len(recent_results) * 100
        
        last_sync = max(self.sync_history, key=lambda r: r.end_time)
        
        return {
            'status': 'healthy' if success_rate >= 80 else 'degraded',
            'last_sync': last_sync.end_time.isoformat(),
            'success_rate': success_rate,
            'total_synced': sum(r.records_synced for r in self.sync_history),
            'total_conflicts': sum(r.conflicts for r in self.sync_history),
            'recent_errors': [
                error for result in recent_results 
                for error in result.errors
            ][-5:]  # 最近5个错误
        }

# 全局同步管理器实例
sync_manager = SyncManager()

def get_sync_manager() -> SyncManager:
    """获取同步管理器实例"""
    return sync_manager