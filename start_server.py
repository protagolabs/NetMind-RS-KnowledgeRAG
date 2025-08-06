#!/usr/bin/env python3
"""
KnowledgeRAG API 服务器启动脚本
=============================

作者: Bin Liang
日期: 2025-08-01
描述: 启动 KnowledgeRAG FastAPI 服务器的便捷脚本

功能:
1. 自动检查和安装依赖
2. 验证环境配置
3. 启动 FastAPI 服务器
4. 提供多种启动模式

使用方法:
    python start_server.py              # 开发模式 (默认)
    python start_server.py --prod       # 生产模式
    python start_server.py --port 9000  # 指定端口
    python start_server.py --help       # 显示帮助
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """检查必要的依赖包"""
    required_packages = [
        'fastapi',
        'uvicorn',
        'pydantic',
        'loguru',
        'python-dotenv',
        'mysql-connector-python',
        'pymilvus',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ 缺少必要的依赖包:")
        for package in missing_packages:
            print(f"   - {package}")
        
        install = input("\n是否自动安装缺少的依赖? (y/n): ").strip().lower()
        if install == 'y':
            try:
                subprocess.check_call([
                    sys.executable, '-m', 'pip', 'install'
                ] + missing_packages)
                print("✅ 依赖安装完成!")
            except subprocess.CalledProcessError:
                print("❌ 依赖安装失败，请手动安装")
                return False
        else:
            return False
    
    return True

def check_environment():
    """检查环境配置"""
    print("🔍 检查环境配置...")
    
    # 检查 .env 文件
    env_file = Path('.env')
    if not env_file.exists():
        print("⚠️  未找到 .env 文件，将使用默认配置")
        
        # 创建示例 .env 文件
        create_env = input("是否创建示例 .env 文件? (y/n): ").strip().lower()
        if create_env == 'y':
            create_example_env()
    else:
        print("✅ 找到 .env 文件")
    
    # 检查必要的环境变量
    required_env_vars = [
        'MYSQL_HOST',
        'MYSQL_PORT', 
        'MYSQL_USER',
        'MYSQL_PASSWORD',
        'MYSQL_DB',
        'MILVUS_HOST',
        'MILVUS_PORT'
    ]
    
    missing_vars = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("⚠️  以下环境变量未设置，将使用默认值:")
        for var in missing_vars:
            print(f"   - {var}")
    
    return True

def create_example_env():
    """创建示例 .env 文件"""
    env_content = """# KnowledgeRAG 环境配置
# ===================

# 应用配置
ENVIRONMENT=development
DEBUG=true

# MySQL 数据库配置
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_mysql_password
MYSQL_DB=knowledge_rag
MYSQL_CHARSET=utf8mb4
MYSQL_COLLATION=utf8mb4_unicode_ci
MYSQL_POOL_SIZE=10

# Milvus 向量数据库配置
MILVUS_HOST=127.0.0.1
MILVUS_PORT=19530
MILVUS_COLLECTION=rag_embeddings_v1
MILVUS_ALIAS=default

# 嵌入模型配置
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=auto
EMBEDDING_BATCH_SIZE=32

# 对象存储配置
OBJECT_STORE_TYPE=local
LOCAL_OBJECT_STORE_PATH=./data/local_object_store
LOCAL_OBJECT_STORE_EXPERIMENTS_DIR=experiments
LOCAL_OBJECT_STORE_AUTO_CREATE_DIRS=true
LOCAL_OBJECT_STORE_MAX_FILE_SIZE=104857600

# OpenAI API Key (如果使用 OpenAI 服务)
# OPENAI_API_KEY=your_openai_api_key
"""
    
    with open('.env', 'w', encoding='utf-8') as f:
        f.write(env_content)
    
    print("✅ 已创建示例 .env 文件")
    print("⚠️  请编辑 .env 文件，设置正确的数据库连接信息")

def start_server(host='127.0.0.1', port=8000, reload=True, workers=1, log_level='info'):
    """启动 FastAPI 服务器"""
    print(f"🚀 启动 KnowledgeRAG API 服务器...")
    print(f"   - 地址: http://{host}:{port}")
    print(f"   - 文档: http://{host}:{port}/docs")
    print(f"   - 重载: {'开启' if reload else '关闭'}")
    print(f"   - 工作进程: {workers}")
    print(f"   - 日志级别: {log_level}")
    print()
    
    try:
        import uvicorn
        
        # 设置 Python 路径
        current_dir = Path(__file__).parent.absolute()
        src_dir = current_dir / 'src'
        
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        
        # 启动服务器
        uvicorn.run(
            "app.main_process:app",
            host=host,
            port=port,
            reload=reload,
            workers=workers if not reload else 1,  # reload 模式下只能用单进程
            log_level=log_level,
            access_log=True
        )
        
    except KeyboardInterrupt:
        print("\n👋 服务器已停止")
    except Exception as e:
        print(f"❌ 服务器启动失败: {e}")
        return False
    
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='KnowledgeRAG API 服务器启动脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python start_server.py                    # 开发模式，自动重载
  python start_server.py --prod             # 生产模式，多进程
  python start_server.py --port 9000        # 指定端口
  python start_server.py --host 0.0.0.0     # 允许外部访问
  python start_server.py --workers 4        # 指定工作进程数
        """
    )
    
    parser.add_argument('--host', default='127.0.0.1', 
                       help='服务器绑定地址 (默认: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=8000,
                       help='服务器端口 (默认: 8000)')
    parser.add_argument('--prod', action='store_true',
                       help='生产模式 (关闭自动重载，启用多进程)')
    parser.add_argument('--workers', type=int, default=1,
                       help='工作进程数 (默认: 1，生产模式建议 4)')
    parser.add_argument('--log-level', choices=['critical', 'error', 'warning', 'info', 'debug'],
                       default='info', help='日志级别 (默认: info)')
    parser.add_argument('--skip-checks', action='store_true',
                       help='跳过依赖和环境检查')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("🎯 KnowledgeRAG API 服务器")
    print("=" * 50)
    
    # 检查依赖和环境
    if not args.skip_checks:
        print("🔧 检查依赖包...")
        if not check_dependencies():
            print("❌ 依赖检查失败")
            return 1
        
        if not check_environment():
            print("❌ 环境检查失败")
            return 1
    
    # 确定启动参数
    if args.prod:
        reload = False
        workers = args.workers if args.workers > 1 else 4
        log_level = 'warning'
        print("🏭 生产模式")
    else:
        reload = True
        workers = 1
        log_level = args.log_level
        print("🛠️  开发模式")
    
    # 启动服务器
    success = start_server(
        host=args.host,
        port=args.port,
        reload=reload,
        workers=workers,
        log_level=log_level
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())