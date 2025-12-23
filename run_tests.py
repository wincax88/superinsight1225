#!/usr/bin/env python3
"""
运行 SuperInsight 平台的完整测试套件
"""
import sys
import os
import subprocess
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def run_command(command, description):
    """运行命令并显示结果"""
    print(f"\n🔄 {description}...")
    print(f"执行命令: {command}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300  # 5分钟超时
        )
        
        if result.returncode == 0:
            print(f"✅ {description} 成功")
            if result.stdout.strip():
                print("输出:")
                print(result.stdout)
        else:
            print(f"❌ {description} 失败")
            if result.stderr.strip():
                print("错误:")
                print(result.stderr)
            if result.stdout.strip():
                print("输出:")
                print(result.stdout)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} 超时")
        return False
    except Exception as e:
        print(f"❌ {description} 异常: {e}")
        return False

def main():
    """主测试流程"""
    print("🚀 SuperInsight 平台测试开始")
    print("=" * 50)
    
    # 测试步骤
    tests = [
        ("python3 test_db_connection.py", "数据库连接测试"),
        ("python3 create_test_db.py", "数据库表创建"),
        ("python3 -m pytest tests/test_database_setup.py -v", "数据库设置测试"),
        ("python3 -m pytest tests/test_system_integration.py -v", "系统集成测试"),
    ]
    
    passed = 0
    total = len(tests)
    
    for command, description in tests:
        if run_command(command, description):
            passed += 1
        else:
            print(f"\n⚠️ {description} 失败，但继续执行其他测试...")
    
    print("\n" + "=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！系统可以正常使用。")
        return True
    else:
        print("⚠️ 部分测试失败，请检查上述错误信息。")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断测试")
        sys.exit(1)