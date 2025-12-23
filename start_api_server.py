#!/usr/bin/env python3
"""
启动 SuperInsight API 服务器进行测试
"""
import sys
import os
import uvicorn
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """启动 API 服务器"""
    print("🚀 启动 SuperInsight API 服务器...")
    print("📊 数据库: PostgreSQL")
    print("🌐 访问地址: http://localhost:8000")
    print("📖 API 文档: http://localhost:8000/docs")
    print("🔍 健康检查: http://localhost:8000/health")
    print("\n按 Ctrl+C 停止服务器")
    
    try:
        uvicorn.run(
            "src.app:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n⚠️ 服务器已停止")

if __name__ == "__main__":
    main()