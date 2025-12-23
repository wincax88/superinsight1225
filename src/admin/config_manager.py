"""
Configuration management system for SuperInsight platform.

Provides centralized configuration management with validation, persistence, and hot-reloading.
"""

import os
import json
import yaml
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from threading import Lock
import threading

logger = logging.getLogger(__name__)


@dataclass
class ConfigChange:
    """配置变更记录"""
    timestamp: datetime
    section: str
    key: str
    old_value: Any
    new_value: Any
    user: str
    reason: str = ""


class ConfigValidator:
    """配置验证器"""
    
    def __init__(self):
        self.validators = {
            'api_rate_limit': self._validate_positive_int,
            'max_concurrent_jobs': self._validate_positive_int,
            'log_retention_days': self._validate_positive_int,
            'maintenance_mode': self._validate_boolean,
            'auto_backup_enabled': self._validate_boolean,
            'backup_retention_days': self._validate_positive_int,
            'database_pool_size': self._validate_positive_int,
            'redis_max_connections': self._validate_positive_int,
            'ai_request_timeout': self._validate_positive_number,
            'extraction_batch_size': self._validate_positive_int,
        }
    
    def validate(self, key: str, value: Any) -> bool:
        """验证配置值"""
        validator = self.validators.get(key)
        if validator:
            return validator(value)
        return True  # 未知配置项默认通过
    
    def _validate_positive_int(self, value: Any) -> bool:
        """验证正整数"""
        try:
            return isinstance(value, int) and value > 0
        except:
            return False
    
    def _validate_positive_number(self, value: Any) -> bool:
        """验证正数"""
        try:
            return isinstance(value, (int, float)) and value > 0
        except:
            return False
    
    def _validate_boolean(self, value: Any) -> bool:
        """验证布尔值"""
        return isinstance(value, bool)


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_dir: str = ".kiro/config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.config_file = self.config_dir / "system.yaml"
        self.changes_file = self.config_dir / "changes.json"
        
        self._config: Dict[str, Any] = {}
        self._lock = Lock()
        self._validator = ConfigValidator()
        self._changes: List[ConfigChange] = []
        
        # 默认配置
        self._default_config = {
            'system': {
                'api_rate_limit': 1000,
                'max_concurrent_jobs': 10,
                'log_retention_days': 30,
                'maintenance_mode': False,
                'debug_mode': False
            },
            'database': {
                'pool_size': 20,
                'pool_timeout': 30,
                'pool_recycle': 3600,
                'echo_sql': False
            },
            'redis': {
                'max_connections': 50,
                'socket_timeout': 5,
                'socket_connect_timeout': 5,
                'retry_on_timeout': True
            },
            'ai': {
                'request_timeout': 60,
                'max_retries': 3,
                'cache_enabled': True,
                'cache_ttl': 3600
            },
            'extraction': {
                'batch_size': 100,
                'max_file_size_mb': 100,
                'supported_formats': ['pdf', 'docx', 'txt', 'html'],
                'timeout_seconds': 300
            },
            'quality': {
                'auto_check_enabled': True,
                'min_confidence_threshold': 0.7,
                'max_issues_per_batch': 10
            },
            'billing': {
                'auto_calculation': True,
                'currency': 'CNY',
                'default_rate_per_annotation': 0.10,
                'default_rate_per_hour': 50.00
            },
            'security': {
                'session_timeout_minutes': 60,
                'max_login_attempts': 5,
                'password_min_length': 8,
                'require_2fa': False
            },
            'monitoring': {
                'metrics_retention_days': 90,
                'alert_enabled': True,
                'health_check_interval': 30,
                'performance_sampling_rate': 0.1
            },
            'backup': {
                'auto_backup_enabled': True,
                'backup_interval_hours': 24,
                'backup_retention_days': 7,
                'backup_location': './backups'
            }
        }
        
        # 加载配置
        self._load_config()
        self._load_changes()
    
    def _load_config(self):
        """加载配置文件"""
        with self._lock:
            if self.config_file.exists():
                try:
                    with open(self.config_file, 'r', encoding='utf-8') as f:
                        loaded_config = yaml.safe_load(f) or {}
                    
                    # 合并默认配置和加载的配置
                    self._config = self._merge_configs(self._default_config, loaded_config)
                    logger.info(f"配置已从 {self.config_file} 加载")
                    
                except Exception as e:
                    logger.error(f"加载配置文件失败: {e}")
                    self._config = self._default_config.copy()
            else:
                # 使用默认配置并保存
                self._config = self._default_config.copy()
                self._save_config()
                logger.info("使用默认配置并已保存")
    
    def _save_config(self):
        """保存配置文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"配置已保存到 {self.config_file}")
        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")
            raise
    
    def _load_changes(self):
        """加载变更历史"""
        if self.changes_file.exists():
            try:
                with open(self.changes_file, 'r', encoding='utf-8') as f:
                    changes_data = json.load(f)
                
                self._changes = [
                    ConfigChange(
                        timestamp=datetime.fromisoformat(change['timestamp']),
                        section=change['section'],
                        key=change['key'],
                        old_value=change['old_value'],
                        new_value=change['new_value'],
                        user=change['user'],
                        reason=change.get('reason', '')
                    )
                    for change in changes_data
                ]
                
            except Exception as e:
                logger.error(f"加载变更历史失败: {e}")
                self._changes = []
    
    def _save_changes(self):
        """保存变更历史"""
        try:
            changes_data = [
                {
                    'timestamp': change.timestamp.isoformat(),
                    'section': change.section,
                    'key': change.key,
                    'old_value': change.old_value,
                    'new_value': change.new_value,
                    'user': change.user,
                    'reason': change.reason
                }
                for change in self._changes[-1000:]  # 只保留最近1000条记录
            ]
            
            with open(self.changes_file, 'w', encoding='utf-8') as f:
                json.dump(changes_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"保存变更历史失败: {e}")
    
    def _merge_configs(self, default: Dict[str, Any], loaded: Dict[str, Any]) -> Dict[str, Any]:
        """合并配置"""
        result = default.copy()
        
        for key, value in loaded.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get(self, section: str, key: Optional[str] = None, default: Any = None) -> Any:
        """获取配置值"""
        with self._lock:
            if key is None:
                return self._config.get(section, default)
            
            section_config = self._config.get(section, {})
            if isinstance(section_config, dict):
                return section_config.get(key, default)
            
            return default
    
    def set(self, section: str, key: str, value: Any, user: str = "system", reason: str = "") -> bool:
        """设置配置值"""
        with self._lock:
            # 验证配置值
            full_key = f"{section}.{key}" if section else key
            if not self._validator.validate(full_key, value):
                logger.error(f"配置值验证失败: {full_key} = {value}")
                return False
            
            # 获取旧值
            old_value = self.get(section, key)
            
            # 设置新值
            if section not in self._config:
                self._config[section] = {}
            
            if not isinstance(self._config[section], dict):
                self._config[section] = {}
            
            self._config[section][key] = value
            
            # 记录变更
            change = ConfigChange(
                timestamp=datetime.now(),
                section=section,
                key=key,
                old_value=old_value,
                new_value=value,
                user=user,
                reason=reason
            )
            self._changes.append(change)
            
            # 保存配置和变更历史
            try:
                self._save_config()
                self._save_changes()
                logger.info(f"配置已更新: {section}.{key} = {value} (用户: {user})")
                return True
            except Exception as e:
                logger.error(f"保存配置失败: {e}")
                return False
    
    def update_section(self, section: str, updates: Dict[str, Any], user: str = "system", reason: str = "") -> bool:
        """批量更新配置节"""
        success = True
        
        for key, value in updates.items():
            if not self.set(section, key, value, user, reason):
                success = False
        
        return success
    
    def get_all(self) -> Dict[str, Any]:
        """获取所有配置"""
        with self._lock:
            return self._config.copy()
    
    def get_changes(self, limit: int = 100) -> List[ConfigChange]:
        """获取变更历史"""
        with self._lock:
            return self._changes[-limit:] if limit > 0 else self._changes.copy()
    
    def reset_to_default(self, section: Optional[str] = None, user: str = "system", reason: str = "重置为默认值") -> bool:
        """重置为默认配置"""
        with self._lock:
            try:
                if section:
                    if section in self._default_config:
                        old_value = self._config.get(section)
                        self._config[section] = self._default_config[section].copy()
                        
                        # 记录变更
                        change = ConfigChange(
                            timestamp=datetime.now(),
                            section=section,
                            key="*",
                            old_value=old_value,
                            new_value=self._config[section],
                            user=user,
                            reason=reason
                        )
                        self._changes.append(change)
                else:
                    old_config = self._config.copy()
                    self._config = self._default_config.copy()
                    
                    # 记录变更
                    change = ConfigChange(
                        timestamp=datetime.now(),
                        section="*",
                        key="*",
                        old_value=old_config,
                        new_value=self._config,
                        user=user,
                        reason=reason
                    )
                    self._changes.append(change)
                
                self._save_config()
                self._save_changes()
                logger.info(f"配置已重置: {section or 'all'} (用户: {user})")
                return True
                
            except Exception as e:
                logger.error(f"重置配置失败: {e}")
                return False
    
    def export_config(self, file_path: str, format: str = "yaml") -> bool:
        """导出配置"""
        try:
            with self._lock:
                config_data = self._config.copy()
            
            if format.lower() == "json":
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
            else:  # yaml
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"配置已导出到 {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"导出配置失败: {e}")
            return False
    
    def import_config(self, file_path: str, user: str = "system", reason: str = "导入配置") -> bool:
        """导入配置"""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.error(f"配置文件不存在: {file_path}")
                return False
            
            # 根据文件扩展名确定格式
            if file_path.suffix.lower() == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    imported_config = json.load(f)
            else:  # yaml
                with open(file_path, 'r', encoding='utf-8') as f:
                    imported_config = yaml.safe_load(f)
            
            if not isinstance(imported_config, dict):
                logger.error("导入的配置格式无效")
                return False
            
            with self._lock:
                old_config = self._config.copy()
                self._config = self._merge_configs(self._default_config, imported_config)
                
                # 记录变更
                change = ConfigChange(
                    timestamp=datetime.now(),
                    section="*",
                    key="*",
                    old_value=old_config,
                    new_value=self._config,
                    user=user,
                    reason=reason
                )
                self._changes.append(change)
                
                self._save_config()
                self._save_changes()
            
            logger.info(f"配置已从 {file_path} 导入 (用户: {user})")
            return True
            
        except Exception as e:
            logger.error(f"导入配置失败: {e}")
            return False
    
    def validate_all(self) -> Dict[str, List[str]]:
        """验证所有配置"""
        errors = {}
        
        with self._lock:
            for section, section_config in self._config.items():
                if isinstance(section_config, dict):
                    for key, value in section_config.items():
                        full_key = f"{section}.{key}"
                        if not self._validator.validate(full_key, value):
                            if section not in errors:
                                errors[section] = []
                            errors[section].append(f"{key}: 无效值 {value}")
        
        return errors


# 全局配置管理器实例
config_manager = ConfigManager()


def get_config(section: str, key: Optional[str] = None, default: Any = None) -> Any:
    """获取配置值的便捷函数"""
    return config_manager.get(section, key, default)


def set_config(section: str, key: str, value: Any, user: str = "system", reason: str = "") -> bool:
    """设置配置值的便捷函数"""
    return config_manager.set(section, key, value, user, reason)


# 配置热重载支持
class ConfigWatcher:
    """配置文件监控器"""
    
    def __init__(self, config_manager: ConfigManager, check_interval: int = 5):
        self.config_manager = config_manager
        self.check_interval = check_interval
        self.last_modified = None
        self.running = False
        self.thread = None
        
        self._update_last_modified()
    
    def _update_last_modified(self):
        """更新最后修改时间"""
        try:
            if self.config_manager.config_file.exists():
                self.last_modified = self.config_manager.config_file.stat().st_mtime
        except:
            pass
    
    def start(self):
        """启动监控"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._watch_loop, daemon=True)
        self.thread.start()
        logger.info("配置文件监控已启动")
    
    def stop(self):
        """停止监控"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=self.check_interval + 1)
        logger.info("配置文件监控已停止")
    
    def _watch_loop(self):
        """监控循环"""
        import time
        
        while self.running:
            try:
                if self.config_manager.config_file.exists():
                    current_modified = self.config_manager.config_file.stat().st_mtime
                    
                    if self.last_modified and current_modified > self.last_modified:
                        logger.info("检测到配置文件变更，重新加载...")
                        self.config_manager._load_config()
                        self.last_modified = current_modified
                    elif not self.last_modified:
                        self.last_modified = current_modified
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"配置文件监控异常: {e}")
                time.sleep(self.check_interval)


# 全局配置监控器
config_watcher = ConfigWatcher(config_manager)


def start_config_watcher():
    """启动配置监控"""
    config_watcher.start()


def stop_config_watcher():
    """停止配置监控"""
    config_watcher.stop()