# 知识索引存储系统

本系统实现了将处理好的文档和chunk数据存储到向量数据库的完整流程。

## 系统架构

### 数据存储结构

1. **文档级别存储**：
   - MySQL表：`documents` - 存储文档基本信息（file_id, summary, insights, key_words等）
   - Milvus集合：`documents_vectors` - 存储summary和key_words的向量
   - Milvus集合：`documents_insights_vectors` - 存储insights向量（每个insight单独一条记录）

2. **Chunk级别存储**（按source_id分组）：
   - MySQL表：`chunks_{source_id}` - 存储chunk基本信息
   - Milvus集合：`chunks_vectors_{source_id}` - 存储chunk的summary和key_words向量
   - Milvus集合：`chunks_insights_vectors_{source_id}` - 存储chunk的insights向量

### 支持的向量索引

1. **文档级别**：
   - `summary_embedding_vector` (1536维)
   - `key_words_embedding_vector` (1536维)  
   - `insights_embedding_vector` (每个文档14个向量，每个1536维)

2. **Chunk级别**：
   - `summary_embedding_vector` (1536维)
   - `key_words_embedding_vector` (1536维)
   - `insights_embedding_vector` (每个chunk8个向量，每个1536维)

### 支持的全文搜索

1. **文档级别**：
   - `summary` 字段
   - `doc_markdown_content` 字段

2. **Chunk级别**：
   - `summary` 字段  
   - `chunk_markdown_content` 字段

## 使用方法

### 1. 环境准备

确保已配置好MySQL和Milvus数据库连接信息。

### 2. 运行数据存储

```bash
cd src/knowledge_rag/knowledge_indexing
python step_2_save_to_db.py
```

### 3. 数据验证

运行后会创建以下数据库结构：

**MySQL表**：
- `documents` - 36条文档记录
- `chunks_{source_id}` X 33个表 - 共1331条chunk记录

**Milvus集合**：
- `documents_vectors` - 36条文档向量记录
- `documents_insights_vectors` - 约504条insights向量记录（36*14）
- `chunks_vectors_{source_id}` X 33个集合 - 共1331条chunk向量记录
- `chunks_insights_vectors_{source_id}` X 33个集合 - 约10648条insights向量记录（1331*8）

## 查询示例

### 1. 根据file_id查询文档

```sql
SELECT * FROM documents WHERE file_id = 'your_file_id';
```

### 2. 根据chunk_id查询chunk

```sql
SELECT * FROM chunks_doc_your_source_id WHERE chunk_id = 'your_chunk_id'; 
```

### 3. 全文搜索示例

```sql
-- 搜索文档
SELECT * FROM documents WHERE MATCH(summary) AGAINST('transformer' IN NATURAL LANGUAGE MODE);

-- 搜索chunk
SELECT * FROM chunks_doc_your_source_id WHERE MATCH(summary) AGAINST('attention' IN NATURAL LANGUAGE MODE);
```

### 4. 向量搜索示例

使用Milvus Python客户端进行向量相似度搜索：

```python
from pymilvus import Collection

# 搜索相似文档
collection = Collection("documents_vectors")
results = collection.search(
    data=[your_query_vector],
    anns_field="summary_embedding",
    param={"metric_type": "L2", "params": {"nprobe": 10}},
    limit=10,
    output_fields=["file_id", "metadata"]
)

# 搜索insights
insights_collection = Collection("documents_insights_vectors")
results = insights_collection.search(
    data=[your_query_vector],
    anns_field="insight_embedding", 
    param={"metric_type": "L2", "params": {"nprobe": 10}},
    limit=10,
    output_fields=["file_id", "insight_text", "metadata"]
)
```

## 设计特点

1. **可扩展性**：chunk按source_id分组存储，避免单表过大
2. **灵活性**：支持多种向量搜索和全文搜索
3. **高效性**：使用批量插入和向量索引优化性能
4. **完整性**：保持MySQL关系数据和Milvus向量数据的一致性

## 文件说明

- `step_2_save_to_db.py` - 主要的数据存储脚本
- `../db_utils/rag_schema_config.py` - 数据库结构配置
- `../db_utils/database_clients.py` - 数据库客户端封装
- `../config.py` - 系统配置文件

## 注意事项

1. 确保MySQL和Milvus服务已启动
2. 确保数据文件路径正确：`experiments_docs_processed/paper_set_1_docs.json` 和 `experiments_docs_processed/paper_set_1_chunks.json`
3. 首次运行会自动创建所有必要的表和集合
4. 大量数据插入可能需要较长时间，请耐心等待