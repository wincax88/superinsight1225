#!/usr/bin/env python3
"""
创建测试数据库表的脚本
"""
import sys
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_tables():
    """创建所有数据库表"""
    try:
        print("🔄 正在导入数据库模型...")
        from src.database.connection import Base, db_manager
        from src.database.models import (
            DocumentModel,
            TaskModel,
            BillingRecordModel,
            QualityIssueModel,
            UserModel,
            ProjectPermissionModel,
            IPWhitelistModel,
            AuditLogModel,
            DataMaskingRuleModel
        )
        
        print("🔄 正在初始化数据库连接...")
        db_manager.initialize()
        
        print("🔄 正在测试数据库连接...")
        if not db_manager.test_connection():
            print("❌ 数据库连接失败！请检查 PostgreSQL 是否运行。")
            return False
        
        print("🔄 正在创建数据库表...")
        engine = db_manager.get_engine()
        
        # 创建所有表
        Base.metadata.create_all(bind=engine)
        
        print("✅ 数据库表创建成功！")
        
        # 显示创建的表
        print("\n📋 已创建的表:")
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        for table in tables:
            print(f"   ✓ {table}")
        
        print(f"\n📊 总共创建了 {len(tables)} 个表")
        return True
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = create_tables()
        if success:
            print("\n🎉 数据库初始化完成！可以开始运行测试了。")
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断操作")
        sys.exit(1)
