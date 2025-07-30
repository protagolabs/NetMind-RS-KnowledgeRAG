# RAG系统两级数据库架构 🚀

根据你的设计需求，重新实现了**专门为RAG系统优化的两级数据库架构**。

## 🎯 你的需求 vs 我的实现

### ✅ 你的需求
1. **两级数据库结构**：文档级 + Chunk级（每个文档独立的数据库）
2. **两级搜索逻辑**：先搜索文档，再并行搜索相关文档的chunks
3. **灵活设计**：方便后续修改表结构
4. **搜索函数**：独立的关键词匹配和向量相似度搜索函数

### ✅ 我的实现
1. **RAGDocumentManager**：管理文档级数据，自动为每个文档创建独立的chunk级数据库
2. **RAGSearchEngine**：实现完整的两级搜索逻辑
3. **ChunkDataManager**：便捷的chunk数据操作
4. **RAGSchemaConfig**：灵活的表结构配置，支持自定义

## 🏗️ 架构设计

```
RAG两级数据库架构
├── 文档级数据库
│   ├── MySQL: documents表 (文档基本信息、总结、关键词)
│   └── Milvus: documents_vectors集合 (文档级embedding)
│
└── Chunk级数据库 (每个文档独立)
    ├── MySQL: doc_20241218_123456_chunks表 (该文档的所有chunks)
    └── Milvus: doc_20241218_123456_chunks_vectors集合 (chunk向量)
```

## 🔍 搜索流程

```
用户查询 → 文档级搜索 → 找到相关文档 → 并行搜索这些文档的chunks → 返回结果
```

## 🚀 快速开始

### 1. 环境准备
```bash
# 启动数据库服务
cd db_server
docker-compose up -d

# 创建实验环境
python experiment_manager.py --create rag_experiment --researcher "你的名字"
python experiment_manager.py --switch rag_experiment
```

### 2. 基本使用
```python
from knowledge_rag.db_utils import RAGDocumentManager, RAGSearchEngine, ChunkDataManager

# 创建管理器
rag_manager = RAGDocumentManager()
search_engine = RAGSearchEngine()
chunk_manager = ChunkDataManager()

# 添加文档（自动创建chunk级数据库）
doc_id = rag_manager.add_document(
    title="深度学习基础",
    summary="介绍深度学习的基本概念",
    keywords=["深度学习", "神经网络"],
    document_embedding=[0.1, 0.2, ...]  # 文档级向量
)

# 添加chunks到文档
chunk_id = chunk_manager.add_chunk_to_document(
    doc_id=doc_id,
    chunk_text="这是第一个chunk的内容",
    chunk_embedding=[0.3, 0.4, ...]  # chunk级向量
)

# 执行两级搜索
results = search_engine.full_search(
    keywords="深度学习 神经网络",
    query_vector=[0.1, 0.2, ...],
    doc_top_k=10,           # 文档级返回数量
    chunk_top_k_per_doc=5   # 每文档chunk数量
)
```

## 📝 核心功能

### 🎯 独立的搜索函数（满足你的要求）

```python
# 文档级关键词搜索
docs = search_engine.search_documents_by_keywords(
    keywords="机器学习",
    search_fields=["title", "summary", "keywords"]  # 你指定搜索哪些字段
)

# 文档级向量搜索
docs = search_engine.search_documents_by_vector(
    query_vector=[0.1, 0.2, ...],
    vector_field="document_embedding"  # 你指定用哪个向量字段搜索
)

# Chunk级搜索
chunks = search_engine.search_chunks_parallel(
    doc_ids=[123, 456],
    keywords="注意力机制",
    query_vector=[0.3, 0.4, ...],
    vector_field="chunk_embedding"  # 你指定chunk向量字段
)
```

### 🔧 灵活的Schema配置

在 `rag_schema_config.py` 中修改表结构：

```python
def _get_custom_chunk_schema(self) -> Dict:
    return {
        "mysql": {
            "columns": [
                # 必需字段
                {"name": "id", "type": "BIGINT", "auto_increment": True, "primary_key": True},
                {"name": "chunk_text", "type": "LONGTEXT", "not_null": True},
                
                # 🔧 你的自定义字段
                {"name": "custom_field", "type": "VARCHAR(500)"},
                {"name": "custom_score", "type": "DECIMAL(5,2)"},
            ]
        },
        "milvus": {
            "fields": [
                {"name": "id", "type": "INT64", "is_primary": True},
                # 🔧 你的向量字段
                {"name": "custom_embedding", "type": "FLOAT_VECTOR", "dim": 768},
            ]
        }
    }
```

## 📊 文件结构

```
src/knowledge_rag/db_utils/
├── __init__.py                    # 模块导入
├── rag_document_manager.py       # 🔑 文档生命周期管理
├── rag_search_engine.py          # 🔑 两级搜索引擎
├── chunk_data_manager.py         # 🔑 Chunk数据操作
├── rag_schema_config.py          # 🔑 灵活的Schema配置
├── database_clients.py           # 数据库客户端封装
├── rag_example.py                # 完整使用示例
├── README_RAG.md                 # 详细文档
└── README.md                     # 本文件
```

## 🎮 运行示例

```bash
# 运行完整示例，了解所有功能
cd src/knowledge_rag/db_utils
python rag_example.py
```

这个示例会展示：
1. ✅ 创建文档和自动生成chunk级数据库
2. ✅ 添加chunks到各个文档
3. ✅ 文档级搜索（第一级）
4. ✅ Chunk级并行搜索（第二级）
5. ✅ 完整的两级搜索流程
6. ✅ 数据管理操作

## 🎯 关键特性

1. **✅ 两级架构**：文档级 + 每个文档独立的chunk级数据库
2. **✅ 并行搜索**：自动并行搜索相关文档的chunks
3. **✅ 灵活配置**：容易修改表结构和搜索字段
4. **✅ 独立函数**：关键词搜索和向量搜索完全独立
5. **✅ 搜索字段可配置**：你可以指定用哪些字段进行搜索
6. **✅ 返回完整MySQL记录**：搜索结果都是完整的数据库记录

## 📚 详细文档

- **[README_RAG.md](./README_RAG.md)** - 超详细的使用说明、API文档、最佳实践
- **[rag_example.py](./rag_example.py)** - 完整的使用示例代码
- **[rag_schema_config.py](./rag_schema_config.py)** - 在这里修改你的表结构

---

## ✅ 总结

这个实现完全按照你的RAG系统需求设计，提供了：

1. **🎯 精确匹配你的需求**：两级数据库 + 两级搜索
2. **🔧 高度灵活**：表结构、搜索字段都可以自定义
3. **⚡ 高性能**：并行chunk搜索，支持大规模数据
4. **📝 详细文档**：超详细的注释和使用说明
5. **🚀 开箱即用**：完整的示例代码

开始构建你的RAG系统吧！🚀