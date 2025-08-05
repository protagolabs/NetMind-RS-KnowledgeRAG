#!/bin/bash

# Document Retrieval API 启动脚本
# 用于启动基于 document_retriever.py 的检索API服务

set -e  # 遇到错误时退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查Python环境
check_python() {
    print_info "检查Python环境..."
    
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 未安装"
        exit 1
    fi
    
    python_version=$(python3 --version 2>&1)
    print_success "Python版本: $python_version"
}

# 检查依赖
check_dependencies() {
    print_info "检查依赖包..."
    
    required_packages=("fastapi" "uvicorn" "pydantic" "litellm" "openai" "fitz" "PIL")
    
    for package in "${required_packages[@]}"; do
        if ! python3 -c "import $package" 2>/dev/null; then
            print_warning "缺少依赖包: $package"
            print_info "请运行: pip install $package"
        else
            print_success "✓ $package"
        fi
    done
}

# 检查配置文件
check_config() {
    print_info "检查配置文件..."
    
    config_file="document_processor_config.json"
    
    if [ ! -f "$config_file" ]; then
        print_error "配置文件不存在: $config_file"
        print_info "请先运行文档处理器: python document_processor.py"
        exit 1
    fi
    
    print_success "✓ 配置文件存在: $config_file"
    
    # 检查配置文件中是否有文档信息
    if ! python3 -c "
import json
with open('$config_file', 'r', encoding='utf-8') as f:
    config = json.load(f)
    docs = config.get('documents_info', {})
    if not docs:
        exit(1)
    print(f'找到 {len(docs)} 个文档')
" 2>/dev/null; then
        print_error "配置文件中没有文档信息"
        print_info "请先运行文档处理器处理文档"
        exit 1
    fi
}

# 检查环境变量
check_env() {
    print_info "检查环境变量..."
    
    if [ -z "$OPENAI_API_KEY" ]; then
        print_warning "OPENAI_API_KEY 环境变量未设置"
        print_info "请设置环境变量: export OPENAI_API_KEY='your-api-key'"
        print_info "或者将API密钥添加到 .env 文件中"
        
        # 尝试从.env文件读取
        if [ -f ".env" ]; then
            print_info "尝试从 .env 文件读取API密钥..."
            export $(cat .env | grep -v '^#' | xargs)
        fi
        
        if [ -z "$OPENAI_API_KEY" ]; then
            print_error "无法获取 OpenAI API 密钥"
            exit 1
        fi
    fi
    
    print_success "✓ OpenAI API 密钥已设置"
}

# 检查端口
check_port() {
    local port=${1:-8001}
    
    print_info "检查端口 $port 是否可用..."
    
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        print_warning "端口 $port 已被占用"
        print_info "正在尝试终止占用端口的进程..."
        
        # 尝试终止占用端口的进程
        pkill -f "retrieval_api_server" || true
        sleep 2
        
        if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
            print_error "无法释放端口 $port"
            print_info "请手动终止占用端口的进程或使用其他端口"
            exit 1
        fi
    fi
    
    print_success "✓ 端口 $port 可用"
}

# 启动API服务器
start_api_server() {
    local port=${1:-8001}
    
    print_info "启动 Document Retrieval API 服务器..."
    print_info "端口: $port"
    print_info "API文档: http://localhost:$port/docs"
    print_info "健康检查: http://localhost:$port/health"
    
    echo ""
    print_success "🚀 服务器启动中..."
    echo ""
    
    # 启动服务器
    python3 retrieval_api_server.py
}

# 显示使用说明
show_usage() {
    echo ""
    echo "Document Retrieval API 启动脚本"
    echo "=================================="
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -p, --port PORT     指定端口号 (默认: 8001)"
    echo "  -h, --help          显示此帮助信息"
    echo "  --check-only        仅检查环境，不启动服务器"
    echo ""
    echo "示例:"
    echo "  $0                    # 使用默认端口启动"
    echo "  $0 -p 8002           # 使用端口8002启动"
    echo "  $0 --check-only      # 仅检查环境"
    echo ""
    echo "API端点:"
    echo "  GET  /               # API信息"
    echo "  GET  /health         # 健康检查"
    echo "  GET  /status         # 系统状态"
    echo "  POST /search         # 完整搜索（返回详细结果）"
    echo "  POST /search/simple  # 简单搜索（仅返回答案）"
    echo "  GET  /document/{id}/structure  # 获取文档结构"
    echo "  POST /reload         # 重新加载配置"
    echo ""
}

# 主函数
main() {
    local port=8001
    local check_only=false
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -p|--port)
                port="$2"
                shift 2
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            --check-only)
                check_only=true
                shift
                ;;
            *)
                print_error "未知选项: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    echo ""
    print_info "Document Retrieval API 启动脚本"
    echo "=================================="
    echo ""
    
    # 执行检查
    check_python
    check_dependencies
    check_env
    check_config
    check_port $port
    
    if [ "$check_only" = true ]; then
        print_success "环境检查完成，所有依赖都已满足"
        exit 0
    fi
    
    # 启动服务器
    start_api_server $port
}

# 捕获中断信号
trap 'echo ""; print_warning "收到中断信号，正在关闭服务器..."; exit 0' INT TERM

# 运行主函数
main "$@" 