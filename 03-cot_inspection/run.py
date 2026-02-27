#!/usr/bin/env python3
"""
思维链合成数据质检工具 - 启动脚本
"""
import sys
from app import create_app

def main():
    print("=" * 60)
    print("思维链合成数据质检工具")
    print("=" * 60)
    
    try:
        app = create_app()
        print("\n✅ 启动成功!")
        print("🌐 访问地址: http://localhost:5001")
        print("📊 API健康检查: http://localhost:5001/api/health")
        print("⏹️  按 Ctrl+C 停止服务")
        print("-" * 60)
        
        app.run(debug=True, host='0.0.0.0', port=5001)
        
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        print("请检查:")
        print("1. 依赖包是否安装: pip install -r requirements.txt")
        print("2. 端口5001是否被占用")
        sys.exit(1)

if __name__ == "__main__":
    main()
