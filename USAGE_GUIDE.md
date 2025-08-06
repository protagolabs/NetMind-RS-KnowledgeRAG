# KnowledgeRAG API 使用指南

## 快速上手

### 1. 启动服务器

```bash
# 基础启动
python start_server.py

# 生产模式启动
python start_server.py --prod --workers 4 --host 0.0.0.0
```

### 2. 访问 API 文档

启动后访问: http://localhost:8000/docs

### 3. 基本使用流程

#### 步骤 1: 上传文档
```bash
curl -X POST "http://localhost:8000/upload" \
  -H "Content-Type: application/json" \
  -d '{
    "markdown_documents": ["# 测试文档\n\n这是内容..."],
    "file_names": ["test.md"]
  }'
```

#### 步骤 2: 生成嵌入
```bash
curl -X POST "http://localhost:8000/embedding" \
  -H "Content-Type: application/json" \
  -d '{
    "doc_analysis_list": [...],
    "chunk_analysis_list": [...]
  }'
```

#### 步骤 3: 保存到数据库
```bash
curl -X POST "http://localhost:8000/save-to-db" \
  -H "Content-Type: application/json" \
  -d '{
    "doc_embedding_list": [...],
    "chunk_embedding_list": [...]
  }'
```

#### 步骤 4: 检索文档
```bash
curl -X POST "http://localhost:8000/doc-retrieval" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "你的查询问题"
  }'
```

## 完整 Python 示例

参考 `test_api.py` 中的 `complete_workflow_example()` 函数。

## 环境配置

复制 `.env.example` 为 `.env` 并修改配置：

```bash
cp .env.example .env
# 编辑 .env 文件设置数据库连接信息
```

## 故障排除

1. **服务启动失败**: 检查数据库连接配置
2. **Retriever 连接失败**: 确保 MySQL 和 Milvus 服务运行
3. **端口被占用**: 使用 `--port` 参数指定其他端口