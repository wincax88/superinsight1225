#!/usr/bin/env python3
"""
测试数据库连接的简单脚本
"""
import sys
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_database_connection():
    """测试数据库连接"""
    try:
        from src.database.connection import db_manager
        
        print("🔄 正在初始化数据库连接...")
        db_manager.initialize()
        
        print("🔄 正在测试数据库连接...")
        if db_manager.test_connection():
            print("✅ 数据库连接成功！")
            
            # 获取数据库信息
            from src.database.connection import get_database_stats
            stats = get_database_stats()
            print(f"📊 数据库信息:")
            for key, value in stats.items():
                print(f"   {key}: {value}")
            
            return True
        else:
            print("❌ 数据库连接失败！")
            return False
            
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_database_connection()
    sys.exit(0 if success else 1)