#!/usr/bin/env python3
"""
创建测试用户的脚本
"""
import sys
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_test_user():
    """创建测试用户"""
    try:
        from src.database.connection import db_manager
        from src.security.controller import SecurityController
        from src.security.models import UserRole
        
        print("🔄 正在初始化数据库连接...")
        db_manager.initialize()
        
        print("🔄 正在创建测试用户...")
        security_controller = SecurityController()
        
        # 使用数据库会话
        with db_manager.get_session() as db:
            # 创建管理员用户
            admin_user = security_controller.create_user(
                username="admin",
                email="admin@superinsight.com",
                password="admin123",
                role=UserRole.ADMIN,
                full_name="系统管理员",
                tenant_id="default",
                db=db
            )
            
            if not admin_user:
                print("❌ 管理员用户创建失败")
                return False
            
            # 创建普通用户
            normal_user = security_controller.create_user(
                username="testuser",
                email="test@superinsight.com", 
                password="test123",
                role=UserRole.BUSINESS_EXPERT,
                full_name="测试用户",
                tenant_id="default",
                db=db
            )
            
            if not normal_user:
                print("❌ 普通用户创建失败")
                return False
        
        print("✅ 测试用户创建成功！")
        print("\n👤 登录账号信息:")
        print("=" * 40)
        print("管理员账号:")
        print(f"  用户名: admin")
        print(f"  密码: admin123")
        print(f"  邮箱: admin@superinsight.com")
        print(f"  角色: 管理员")
        print()
        print("普通用户账号:")
        print(f"  用户名: testuser")
        print(f"  密码: test123")
        print(f"  邮箱: test@superinsight.com")
        print(f"  角色: 业务专家")
        print("=" * 40)
        
        return True
        
    except Exception as e:
        print(f"❌ 创建用户失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = create_test_user()
    sys.exit(0 if success else 1)