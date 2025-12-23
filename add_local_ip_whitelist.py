#!/usr/bin/env python3
"""
添加本地IP到白名单的脚本
"""
import sys
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def add_local_ips():
    """添加本地IP到白名单"""
    try:
        from src.database.connection import db_manager
        from src.security.controller import SecurityController
        from src.security.models import UserModel, UserRole
        
        print("🔄 正在初始化数据库连接...")
        db_manager.initialize()
        
        print("🔄 正在添加本地IP到白名单...")
        security_controller = SecurityController()
        
        # 使用数据库会话
        with db_manager.get_session() as db:
            # 获取管理员用户
            admin_user = db.query(UserModel).filter(
                UserModel.role == UserRole.ADMIN
            ).first()
            
            if not admin_user:
                print("❌ 未找到管理员用户")
                return False
            
            # 添加本地IP地址到白名单
            local_ips = [
                "127.0.0.1",      # localhost IPv4
                "::1",            # localhost IPv6
                "0.0.0.0",        # all interfaces
                "192.168.0.0/16", # private network range
                "10.0.0.0/8",     # private network range
                "172.16.0.0/12"   # private network range
            ]
            
            for ip in local_ips:
                try:
                    success = security_controller.add_ip_to_whitelist(
                        ip_address=ip if "/" not in ip else ip.split("/")[0],
                        ip_range=ip if "/" in ip else None,
                        tenant_id="default",
                        created_by=admin_user.id,
                        description=f"本地开发环境 - {ip}",
                        db=db
                    )
                    if success:
                        print(f"✅ 已添加IP到白名单: {ip}")
                    else:
                        print(f"⚠️ IP可能已存在: {ip}")
                except Exception as e:
                    print(f"❌ 添加IP失败 {ip}: {e}")
        
        print("✅ 本地IP白名单配置完成！")
        return True
        
    except Exception as e:
        print(f"❌ 配置失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = add_local_ips()
    sys.exit(0 if success else 1)