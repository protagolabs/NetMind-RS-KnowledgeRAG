# KnowledgeRAG API 接口文档

## 概述

KnowledgeRAG API 是一个基于 FastAPI 的知识检索增强生成系统，提供文档处理、嵌入生成、数据存储和知识检索等功能。

### 技术栈
- **框架**: FastAPI (高性能异步 Web 框架)
- **数据库**: MySQL + Milvus (结构化数据 + 向量数据)
- **嵌入模型**: sentence-transformers
- **部署**: Uvicorn ASGI 服务器

### 核心特性
- ✅ 异步处理，高并发性能
- ✅ 自动 API 文档生成 (Swagger/OpenAPI)
- ✅ 数据验证和序列化 (Pydantic)
- ✅ 全局 Retriever 单例管理
- ✅ 完整的错误处理和日志记录
- ✅ CORS 支持，便于前端集成

## 快速开始

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

### 2. 启动服务器

```bash
# 开发模式 (自动重载)
python start_server.py

# 生产模式 (多进程)
python start_server.py --prod --workers 4

# 指定端口
python start_server.py --port 9000

# 允许外部访问
python start_server.py --host 0.0.0.0
```

### 3. 访问 API 文档

启动服务器后，访问以下地址查看自动生成的 API 文档：

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API 端点详细说明

### 系统管理

#### 1. 健康检查

**GET** `/`

简单的健康检查端点。

**响应示例**:
```json
{
  "message": "KnowledgeRAG API 服务正常运行",
  "version": "1.0.0",
  "status": "healthy",
  "retriever_status": "connected"
}
```

#### 2. 详细健康检查

**GET** `/health`

详细的系统状态检查。

**响应示例**:
```json
{
  "status": "healthy",
  "retriever": {
    "connected": true,
    "mysql_connected": true,
    "milvus_connected": true
  },
  "config": {
    "environment": "development",
    "debug": true
  }
}
```

### 文档处理

#### 3. 上传并处理 Markdown 文档

**POST** `/upload`

上传 Markdown 文档并进行结构化分析。

**请求体**:
```json
{
  "markdown_documents": [
    "# 文档标题\n\n这是文档内容...",
    "# 另一个文档\n\n更多内容..."
  ],
  "file_names": [
    "document1.md",
    "document2.md"
  ]
}
```

**响应示例**:
```json
{
  "success": true,
  "message": "文档上传和处理成功",
  "doc_analysis_list": [
    {
      "file_id": "doc_001",
      "file_name": "document1.md",
      "summary": "文档摘要...",
      "key_words": ["关键词1", "关键词2"],
      "insights": ["洞察1", "洞察2"],
      "doc_markdown_content": "# 文档标题\n\n...",
      "source_id": "paper_set_1"
    }
  ],
  "chunk_analysis_list": [
    {
      "chunk_id": "chunk_001",
      "file_id": "doc_001",
      "chunk_content": "chunk 内容...",
      "chunk_summary": "chunk 摘要...",
      "chunk_insights": ["chunk 洞察"],
      "source_id": "paper_set_1"
    }
  ],
  "total_docs": 2,
  "total_chunks": 15
}
```

**字段说明**:
- `markdown_documents`: Markdown 文档内容列表
- `file_names`: 对应的文件名列表
- `doc_analysis_list`: 文档级分析结果
- `chunk_analysis_list`: 文档片段级分析结果

#### 4. 生成文档和 chunk 嵌入

**POST** `/embedding`

为处理后的文档和 chunks 生成嵌入向量。

**请求体**:
```json
{
  "doc_analysis_list": [
    {
      "file_id": "doc_001",
      "summary": "文档摘要...",
      "key_words": ["关键词1"],
      "insights": ["洞察1"]
    }
  ],
  "chunk_analysis_list": [
    {
      "chunk_id": "chunk_001",
      "chunk_content": "chunk 内容...",
      "chunk_summary": "chunk 摘要..."
    }
  ]
}
```

**响应示例**:
```json
{
  "success": true,
  "message": "嵌入生成成功",
  "doc_embedding_list": [
    {
      "file_id": "doc_001",
      "summary": "文档摘要...",
      "summary_embedding": [0.1, 0.2, ...],
      "key_words_embedding": [0.3, 0.4, ...],
      "insights_embedding": [0.5, 0.6, ...]
    }
  ],
  "chunk_embedding_list": [
    {
      "chunk_id": "chunk_001",
      "chunk_content": "chunk 内容...",
      "chunk_embedding": [0.7, 0.8, ...],
      "chunk_insights_embedding": [0.9, 1.0, ...]
    }
  ]
}
```

#### 5. 保存数据到数据库

**POST** `/save-to-db`

将处理后的数据保存到 MySQL 和 Milvus 数据库。

**请求体** (方式一 - 使用文件路径):
```json
{
  "doc_json_file_path": "/path/to/docs.json",
  "chunk_json_file_path": "/path/to/chunks.json"
}
```

**请求体** (方式二 - 直接传递数据):
```json
{
  "doc_embedding_list": [...],
  "chunk_embedding_list": [...]
}
```

**响应示例**:
```json
{
  "success": true,
  "message": "数据保存到数据库成功"
}
```

### 知识检索

#### 6. 文档检索

**POST** `/doc-retrieval`

根据查询文本检索相关文档。

**请求体**:
```json
{
  "query": "机器学习算法的应用"
}
```

**响应示例**:
```json
{
  "success": true,
  "message": "文档检索成功",
  "results": [
    {
      "id": 1,
      "file_id": "doc_001",
      "file_name": "ml_algorithms.md",
      "summary": "机器学习算法综述...",
      "insights": ["关键洞察1", "关键洞察2"],
      "key_words": ["机器学习", "算法", "分类"],
      "doc_markdown_content": "# 机器学习算法\n\n完整文档内容...",
      "chunk_mysql_table": "chunks_doc_001",
      "chunk_milvus_collection": "chunks_vectors_doc_001",
      "chunk_count": 15,
      "processing_status": "completed",
      "created_at": "2025-08-01T10:00:00Z",
      "updated_at": "2025-08-01T10:30:00Z",
      "relevance_score": 0.95,
      "search_field": "fulltext"
    }
  ],
  "total_results": 5
}
```

**字段说明**:
- `id`: 文档在数据库中的主键ID
- `file_id`: 文档的唯一标识符
- `file_name`: 文档文件名
- `summary`: 文档摘要
- `insights`: 文档关键洞察列表
- `key_words`: 文档关键词列表
- `doc_markdown_content`: 完整的Markdown文档内容
- `chunk_mysql_table`: 对应的chunk MySQL表名
- `chunk_milvus_collection`: 对应的chunk Milvus集合名
- `chunk_count`: 文档包含的chunk数量
- `processing_status`: 处理状态（如"completed"）
- `created_at`: 创建时间
- `updated_at`: 更新时间
- `relevance_score`: 相关度分数（0-1）
- `search_field`: 搜索字段标识（如"fulltext"、"summary_embedding"等）

#### 7. 按数据集类型检索文档

**POST** `/doc-retrieval-by-dataset`

在指定数据集类型中检索相关文档。

**请求体**:
```json
{
  "data_set_type": "paper_set_1",
  "query": "深度学习在计算机视觉中的应用"
}
```

**响应示例**:
```json
{
  "success": true,
  "message": "数据集文档检索成功",
  "results": [
    {
      "id": 2,
      "file_id": "doc_002",
      "file_name": "cv_deep_learning.md",
      "summary": "计算机视觉中的深度学习方法...",
      "insights": ["深度学习在CV中的应用", "卷积神经网络优势"],
      "key_words": ["计算机视觉", "深度学习", "CNN"],
      "doc_markdown_content": "# 计算机视觉中的深度学习\n\n完整文档内容...",
      "chunk_mysql_table": "chunks_doc_002",
      "chunk_milvus_collection": "chunks_vectors_doc_002",
      "chunk_count": 22,
      "processing_status": "completed",
      "created_at": "2025-08-01T11:00:00Z",
      "updated_at": "2025-08-01T11:15:00Z"
    }
  ],
  "total_results": 3
}
```

**字段说明**:
- `id`: 文档在数据库中的主键ID
- `file_id`: 文档的唯一标识符
- `file_name`: 文档文件名
- `summary`: 文档摘要
- `insights`: 文档关键洞察列表
- `key_words`: 文档关键词列表
- `doc_markdown_content`: 完整的Markdown文档内容
- `chunk_mysql_table`: 对应的chunk MySQL表名
- `chunk_milvus_collection`: 对应的chunk Milvus集合名
- `chunk_count`: 文档包含的chunk数量
- `processing_status`: 处理状态（如"completed"）
- `created_at`: 创建时间
- `updated_at`: 更新时间

#### 8. Chunk 检索

**POST** `/chunk-retrieval`

在指定文档中检索相关的文档片段。

**请求体**:
```json
{
  "query": "神经网络优化方法",
  "doc_ids": ["doc_001", "doc_002"],
  "each_doc_chunk_number": 10
}
```

**响应示例**:
```json
{
  "success": true,
  "message": "Chunk 检索成功",
  "results": [
    {
      "chunk_id": "chunk_001",
      "file_id": "doc_001",
      "chunk_content": "梯度下降是最常用的优化算法...",
      "chunk_summary": "介绍梯度下降优化方法",
      "chunk_insights": ["梯度下降原理", "优化算法比较"],
      "similarity_score": 0.89,
      "relevance_score": 0.92,
      "search_field": "summary_embedding",
      "metadata": {
        "chunk_index": 5,
        "token_count": 256,
        "section": "优化算法"
      },
      "created_at": "2025-08-01T10:15:00Z"
    }
  ],
  "total_results": 20
}
```

**字段说明**:
- `chunk_id`: chunk的唯一标识符
- `file_id`: 所属文档的ID
- `chunk_content`: chunk的文本内容
- `chunk_summary`: chunk的摘要
- `chunk_insights`: chunk的关键洞察列表
- `similarity_score`: 向量相似度分数（0-1）
- `relevance_score`: 文本相关度分数（0-1）
- `search_field`: 搜索字段标识（如"summary_embedding"、"fulltext"等）
- `metadata`: 包含chunk的元数据信息
  - `chunk_index`: chunk在文档中的索引位置
  - `token_count`: chunk的token数量
  - `section`: chunk所属的章节
- `created_at`: chunk创建时间

**参数说明**:
- `query`: 查询文本
- `doc_ids`: 要搜索的文档 ID 列表
- `each_doc_chunk_number`: 每个文档返回的 chunk 数量 (1-50)

#### 9. Chunk 决策匹配

**POST** `/chunk-decision`

对 chunks 进行智能决策匹配，筛选最相关的内容。

**请求体**:
```json
{
  "query_text": "机器学习模型评估",
  "chunks": [
    {
      "chunk_id": "chunk_001",
      "chunk_content": "交叉验证是评估模型性能的重要方法...",
      "chunk_summary": "模型评估方法介绍"
    },
    {
      "chunk_id": "chunk_002", 
      "chunk_content": "准确率、召回率和 F1 分数是常用指标...",
      "chunk_summary": "评估指标说明"
    }
  ]
}
```

**响应示例**:
```json
{
  "success": true,
  "message": "Chunk 决策匹配成功",
  "results": [
    {
      "chunk_id": "chunk_001",
      "chunk_content": "交叉验证是评估模型性能的重要方法...",
      "relevance_score": 0.94,
      "decision_reason": "直接相关于模型评估方法",
      "is_relevant": true
    },
    {
      "chunk_id": "chunk_002",
      "chunk_content": "准确率、召回率和 F1 分数是常用指标...",
      "relevance_score": 0.91,
      "decision_reason": "涉及具体的评估指标",
      "is_relevant": true
    }
  ],
  "total_results": 2
}
```

## 完整的工作流程示例

### 场景：处理并检索学术论文

```python
import httpx
import asyncio

async def complete_workflow_example():
    """完整工作流程示例"""
    base_url = "http://localhost:8000"
    
    async with httpx.AsyncClient() as client:
        # 1. 上传文档
        upload_response = await client.post(f"{base_url}/upload", json={
            "markdown_documents": [
                "# 深度学习综述\n\n深度学习是机器学习的一个分支...",
                "# 计算机视觉应用\n\n卷积神经网络在图像识别中..."
            ],
            "file_names": ["deep_learning_survey.md", "cv_applications.md"]
        })
        
        upload_data = upload_response.json()
        print(f"文档上传成功: {upload_data['total_docs']} 个文档")
        
        # 2. 生成嵌入
        embedding_response = await client.post(f"{base_url}/embedding", json={
            "doc_analysis_list": upload_data["doc_analysis_list"],
            "chunk_analysis_list": upload_data["chunk_analysis_list"]
        })
        
        embedding_data = embedding_response.json()
        print("嵌入生成完成")
        
        # 3. 保存到数据库
        save_response = await client.post(f"{base_url}/save-to-db", json={
            "doc_embedding_list": embedding_data["doc_embedding_list"],
            "chunk_embedding_list": embedding_data["chunk_embedding_list"]
        })
        
        print("数据保存到数据库完成")
        
        # 4. 文档检索
        doc_search_response = await client.post(f"{base_url}/doc-retrieval", json={
            "query": "深度学习在图像处理中的应用"
        })
        
        doc_results = doc_search_response.json()
        print(f"找到 {doc_results['total_results']} 个相关文档")
        
        # 5. Chunk 检索
        if doc_results["results"]:
            doc_ids = [doc["file_id"] for doc in doc_results["results"][:2]]
            
            chunk_search_response = await client.post(f"{base_url}/chunk-retrieval", json={
                "query": "卷积神经网络的优化方法",
                "doc_ids": doc_ids,
                "each_doc_chunk_number": 5
            })
            
            chunk_results = chunk_search_response.json()
            print(f"找到 {chunk_results['total_results']} 个相关 chunks")

# 运行示例
# asyncio.run(complete_workflow_example())
```

## 错误处理

API 使用标准的 HTTP 状态码和统一的错误响应格式。

### 错误响应格式

```json
{
  "success": false,
  "message": "错误描述信息",
  "status_code": 400
}
```

### 常见错误码

- **400 Bad Request**: 请求参数错误
- **404 Not Found**: 资源不存在
- **500 Internal Server Error**: 服务器内部错误
- **503 Service Unavailable**: 服务不可用 (如数据库连接失败)

### 错误示例

```json
{
  "success": false,
  "message": "文档数量与文件名数量不匹配",
  "status_code": 400
}
```

## 性能优化建议

### 1. 数据库连接
- 使用连接池管理数据库连接
- 合理设置连接池大小 (默认: 10)
- 定期检查连接健康状态

### 2. 批量处理
- 文档上传建议单次不超过 100 个
- Chunk 检索建议单次不超过 1000 个
- 使用异步处理提高并发性能

### 3. 缓存策略
- 对频繁查询的结果进行缓存
- 使用 Redis 等缓存系统
- 设置合理的缓存过期时间

### 4. 生产部署
```bash
# 使用多进程模式
python start_server.py --prod --workers 4 --host 0.0.0.0

# 或使用 Gunicorn
gunicorn src.app.main_process:app -w 4 -k uvicorn.workers.UvicornWorker
```

## 监控和日志

### 日志配置
- 使用 loguru 进行结构化日志记录
- 支持多种日志级别: DEBUG, INFO, WARNING, ERROR
- 日志自动轮转和压缩

### 监控指标
- API 响应时间
- 数据库连接状态
- 内存使用情况
- 错误率统计

## 安全考虑

### 1. 输入验证
- 所有输入参数都经过 Pydantic 验证
- 文件大小限制
- 查询长度限制

### 2. 数据库安全
- 使用参数化查询防止 SQL 注入
- 数据库连接加密
- 定期更新数据库密码

### 3. API 安全
- 添加认证和授权机制 (可选)
- 请求频率限制
- CORS 配置

## 部署清单

### 开发环境
- [ ] Python 3.8+
- [ ] MySQL 8.0+
- [ ] Milvus 2.3+
- [ ] 安装项目依赖
- [ ] 配置 .env 文件
- [ ] 初始化数据库

### 生产环境
- [ ] 配置反向代理 (Nginx)
- [ ] 设置 HTTPS 证书
- [ ] 配置监控和日志
- [ ] 数据库备份策略
- [ ] 容器化部署 (可选)

## 常见问题 (FAQ)

### Q: 如何修改默认端口？
A: 使用 `--port` 参数：`python start_server.py --port 9000`

### Q: 如何启用多进程模式？
A: 使用 `--prod` 参数：`python start_server.py --prod --workers 4`

### Q: 数据库连接失败怎么办？
A: 检查 .env 文件中的数据库配置，确保 MySQL 和 Milvus 服务正常运行

### Q: 如何查看详细的 API 文档？
A: 启动服务器后访问 http://localhost:8000/docs

### Q: 支持哪些文档格式？
A: 目前支持 Markdown 格式，未来将支持 PDF、Word 等格式

### Q: 如何自定义嵌入模型？
A: 在 .env 文件中设置 `EMBEDDING_MODEL` 环境变量

## 技术支持

如有问题或建议，请联系：
- 作者: Bin Liang
- 邮箱: [your-email@example.com]
- 项目仓库: [repository-url]

---

*最后更新: 2025-08-01*