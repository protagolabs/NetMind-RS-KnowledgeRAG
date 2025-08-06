# KnowledgeRAG API 服务器

基于 FastAPI 的知识检索增强生成 (RAG) 系统，提供文档处理、向量检索和知识问答功能。

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd NetMind-RS-KnowledgeRAG

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，设置数据库连接信息
```

### 2. 启动服务

```bash
# 开发模式 (自动重载)
python start_server.py

# 生产模式 (多进程)
python start_server.py --prod --workers 4

# 指定端口和地址
python start_server.py --host 0.0.0.0 --port 9000
```

### 3. 访问 API 文档

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📋 主要功能

### 文档处理流程
1. **文档上传** (`/upload`) - 上传 Markdown 文档并进行结构化分析
2. **嵌入生成** (`/embedding`) - 为文档和 chunks 生成向量嵌入
3. **数据存储** (`/save-to-db`) - 将数据保存到 MySQL + Milvus

### 知识检索功能
1. **文档检索** (`/doc-retrieval`) - 基于查询检索相关文档
2. **数据集检索** (`/doc-retrieval-by-dataset`) - 在特定数据集中检索
3. **Chunk 检索** (`/chunk-retrieval`) - 检索文档片段
4. **智能匹配** (`/chunk-decision`) - 对结果进行智能筛选

## 🛠️ 技术架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │     MySQL       │    │    Milvus       │
│   Web Server    │◄──►│  结构化数据      │    │   向量数据库     │
│                 │    │   文档元数据     │    │   向量检索      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ▲
         │
┌─────────────────┐
│   Retriever     │
│   单例管理      │
│   连接池       │
└─────────────────┘
```

### 核心组件
- **FastAPI**: 高性能异步 Web 框架
- **MySQL**: 存储文档元数据和结构化信息
- **Milvus**: 专业向量数据库，支持高效相似度搜索
- **Retriever 单例**: 全局维护数据库连接和检索逻辑

## 📖 API 使用示例

### Python 客户端示例

```python
import httpx
import asyncio

async def example_workflow():
    base_url = "http://localhost:8954"
    
    async with httpx.AsyncClient() as client:
        # 1. 上传文档
        upload_response = await client.post(f"{base_url}/upload", json={
            "markdown_documents": [
                "# 机器学习基础\n\n机器学习是人工智能的分支..."
            ],
            "file_names": ["ml_basics.md"]
        })
        
        upload_data = upload_response.json()
        
        # 2. 生成嵌入
        embedding_response = await client.post(f"{base_url}/embedding", json={
            "doc_analysis_list": upload_data["doc_analysis_list"],
            "chunk_analysis_list": upload_data["chunk_analysis_list"]
        })
        
        embedding_data = embedding_response.json()
        
        # 3. 保存到数据库
        await client.post(f"{base_url}/save-to-db", json={
            "doc_embedding_list": embedding_data["doc_embedding_list"],
            "chunk_embedding_list": embedding_data["chunk_embedding_list"]
        })
        
        # 4. 检索文档
        search_response = await client.post(f"{base_url}/doc-retrieval", json={
            "query": "什么是机器学习？"
        })
        
        results = search_response.json()
        print(f"找到 {results['total_results']} 个相关文档")

# 运行示例
asyncio.run(example_workflow())
```

### cURL 示例

```bash
# 健康检查
curl -X GET "http://localhost:8000/health"

# 文档上传
curl -X POST "http://localhost:8000/upload" \
  -H "Content-Type: application/json" \
  -d '{
    "markdown_documents": ["# 测试文档\n\n这是测试内容..."],
    "file_names": ["test.md"]
  }'

# 文档检索
curl -X POST "http://localhost:8000/doc-retrieval" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "机器学习"
  }'
```

## 🐳 Docker 部署

### 单容器部署

```bash
# 构建镜像
docker build -t knowledge-rag-api .

# 运行容器
docker run -p 8000:8000 \
  -e MYSQL_HOST=your-mysql-host \
  -e MYSQL_PASSWORD=your-password \
  -e MILVUS_HOST=your-milvus-host \
  knowledge-rag-api
```

### 完整服务栈部署

```bash
# 启动所有服务 (API + MySQL + Milvus)
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f knowledge-rag-api
```

## 🧪 测试

### 运行测试脚本

```bash
# 测试所有端点
python test_api.py

# 测试特定端点
python test_api.py --endpoint health

# 指定服务器地址
python test_api.py --base-url http://localhost:9000
```

### 手动测试

```bash
# 确保服务器运行
python start_server.py

# 在另一个终端运行测试
python test_api.py --verbose
```

## ⚙️ 配置说明

### 环境变量

```bash
# 应用配置
ENVIRONMENT=development          # 运行环境
DEBUG=true                      # 调试模式

# MySQL 配置
MYSQL_HOST=127.0.0.1           # 数据库主机
MYSQL_PORT=3306                # 数据库端口
MYSQL_USER=root                # 用户名
MYSQL_PASSWORD=your_password    # 密码
MYSQL_DB=knowledge_rag         # 数据库名

# Milvus 配置
MILVUS_HOST=127.0.0.1          # Milvus 主机
MILVUS_PORT=19530              # Milvus 端口

# 嵌入模型配置
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=auto          # 计算设备 (auto/cpu/cuda)
```

### 性能调优

```bash
# 生产环境建议配置
python start_server.py \
  --prod \
  --workers 4 \
  --host 0.0.0.0 \
  --port 8000
```

## 📊 监控和日志

### 健康检查端点

- `GET /` - 基础健康检查
- `GET /health` - 详细系统状态

### 日志配置

日志使用 loguru 库，支持：
- 结构化日志记录
- 自动日志轮转
- 多级别日志输出
- 异步日志写入

## 🔒 安全考虑

### 生产环境安全清单

- [ ] 设置强密码和安全的数据库连接
- [ ] 启用 HTTPS (使用 Nginx 反向代理)
- [ ] 配置防火墙规则
- [ ] 定期更新依赖包
- [ ] 实施 API 访问频率限制
- [ ] 启用访问日志监控

### 数据安全

- 所有数据库查询使用参数化查询
- 输入数据经过 Pydantic 验证
- 敏感信息通过环境变量配置
- 支持数据库连接加密

## 🚀 部署到生产环境

### 1. 服务器准备

```bash
# 安装 Docker 和 Docker Compose
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# 安装 Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### 2. 配置 Nginx (可选)

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 3. 启用 HTTPS

```bash
# 使用 Let's Encrypt
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

## 📞 技术支持

### 常见问题

**Q: 服务启动失败怎么办？**
A: 检查数据库连接配置，确保 MySQL 和 Milvus 服务正常运行。

**Q: 如何修改默认端口？**
A: 使用 `--port` 参数：`python start_server.py --port 9000`

**Q: 如何启用生产模式？**
A: 使用 `--prod` 参数：`python start_server.py --prod --workers 4`

### 联系方式

- 作者: Bin Liang
- 邮箱: [your-email@example.com]
- 项目仓库: [repository-url]

## 📄 许可证

本项目使用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

*最后更新: 2025-08-01*