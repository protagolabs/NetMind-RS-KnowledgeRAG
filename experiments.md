# KnowledgeRAG 本地数据库 & 存储环境技术说明（NotebookLM RAG 能力逆向工程）

**版本：** Draft v0.2 
**日期：** 2025-07-18（更新）
**作者：** XYZ-Algorith-Team (Chengyu Huang, Bin Liang, Hongyi Gu, Yujing Wang, Guocheng Hu)
**适用范围：** 本文档适用于在本地/开发环境中模拟 NotebookLM 风格的多文档 RAG（Retrieval-Augmented Generation）系统，以增强平台内 Agent 的知识访问能力。生产环境部署需基于本文档作适配与加固。

## 🎯 当前项目状态

**已实现核心功能：**
- ✅ 统一配置管理（.env + config.py）
- ✅ 基础存储层（MySQL + Milvus + 本地S3）
- ✅ 向量维度统一管理（1536维）
- ✅ 实验管理系统
- ✅ 完整的RAG演示系统
- ✅ 灵活搜索引擎

**系统已可用于：**
- 文档上传和向量化
- 语义搜索和检索
- 实验环境管理
- 快速原型开发

> 📝 本文档与实际代码同步更新，反映当前项目的真实状态。

---

## 1. 文档目的与目标

本说明用于：

1. **定义本地模拟环境**：基于 *本地文件系统（S3 模拟）+ MySQL（结构化元数据）+ Milvus（向量索引）* 的三层存储架构。
2. **规范数据摄取（ingestion）流水线**：从用户上传文件开始，到清洗、切分（chunking）、嵌入（embedding）、结构化入库、可检索化的全过程。
3. **规定最小必备文件/模块内容要求**：DB 启动脚本、服务访问封装、数据模型、检索接口、引用追踪。
4. **为研发团队提供迭代基线**：在本地端对 RAG 管线做快速实验、指标调优、Agent 集成验证。

> **各位可以把这套本地环境理解为：NotebookLM 的“知识基座（knowledge substrate）”在单机可复现版。**

---

## 2. 场景假设与约束

### 2.1 用户与租户隔离（Multi-Tenancy）

* 平台支持多用户；每位用户上传的文档需逻辑隔离。
* 在本地模拟中，通过 **user\_id 分区目录 + 数据库行级隔离** 实现。
* 后续可扩展为组织级、项目级命名空间。

### 2.2 一次性批量上传、文档复杂性

* 用户可能一次上传多个版本（v1, v2…）或多份相互关联的文档（主文件 + 附件 + 更新说明）。
* 文档间可能存在时间依赖（如“2025-07-16_xx-doc”）、版本 supersedes 关系、交叉引用。
* 用户也可能上传关于某一领域的各种 paper 

### 2.3 Token 成本控制

* 处理的要快，比如 NotebookLM 上传完基本就可以用了，这个作为我们的一个目标速度。
* 检索阶段必须做到**高精度筛选**，仅向 LLM 发送最小必要上下文。
* 支持摘要压缩（context compression）、时间过滤、版本过滤、Cross-Encoder 重排后截断。

---

## 3. 系统总体架构

```
┌──────────────────────────────────────────────────────────────────┐
│                            User Upload                           │
└──────────────┬───────────────────────────────────────────────────┘
               │  (HTTP / CLI / SDK)
               ▼
        Ingestion Orchestrator
               │
               ├─► Object Store (Local S3模拟)
               │       • 原始文件（原貌保存）
               │       • 正规化路径：/user/{uid}/doc/{doc_uuid}/v{n}/...
               │
               ├─► Document Parser / Normalizer
               │       • 文本提取、版面解析、表格分离、元数据抽取
               │
               ├─► Chunker
               │       • 语义/结构分块，建立 chunk_id
               │
               ├─► Embedding Worker
               │       • 生成向量；写 Milvus
               │
               └─► Metadata Writer
                       • 文档/版本/块关系写 MySQL

Query Flow:
User Query → Query Router → Metadata Filter (user, version, time) → Vector KNN + BM25 → Rerank → Context Compression → LLM → Answer + Citations
```

---

## 4. 数据生命周期（Data Lifecycle）

| 阶段 | 输入          | 处理              | 输出          | 写入层                                 |
| -- | ----------- | --------------- | ----------- | ----------------------------------- |
| 上传 | 原始文件流       | 校验、分配 doc\_uuid | 原始文件        | S3 模拟                               |
| 解析 | 原始文件        | 文本抽取、结构化、版本识别   | 正规化文档块      | MySQL (doc\_versions, parsed\_text) |
| 分块 | 结构化文本       | 语义/规则 chunking  | chunk 记录    | MySQL (chunks)                      |
| 向量 | chunk 文本    | Embedding 模型    | 向量数组        | Milvus (embeddings)                 |
| 检索 | 查询          | 向量召回 + 过滤       | top-k chunk | 内存/服务返回                             |
| 生成 | top-k chunk | Prompt 构造 + LLM | 回答 + 引用     | Answer 服务；可写审计表                     |

---

## 5. 数据模型（核心实体）

> 以下为最小化（MVP）字段；生产环境可增加 ACL、加密、审计、扩展属性。

### 5.1 实体关系图（简化）

```
User 1───* Document 1───* DocumentVersion 1───* Chunk 1──1 Embedding
                                  │
                                  └──* ChunkRelation (*跨文档/跨版本引用*)
```

---

### 5.2 表：`users`

| 字段           | 类型                        | 说明                  |
| ------------ | ------------------------- | ------------------- |
| id (PK)      | BIGINT                    | 用户内部主键              |
| external\_id | VARCHAR                   | 外部认证系统 user key（可空） |
| name         | VARCHAR                   | 显示名称                |
| created\_at  | DATETIME                  | 创建时间                |
| status       | ENUM('active','disabled') | 用户状态                |

**最少需含**：`id`, `name`, `created_at`。

---

### 5.3 表：`documents`

| 字段                     | 类型       | 说明              |
| ---------------------- | -------- | --------------- |
| id (PK)                | BIGINT   |                 |
| user\_id (FK users.id) | BIGINT   | 所属用户            |
| doc\_uuid              | CHAR(36) | 跨版本不变的文档全局标识    |
| title                  | VARCHAR  | 文档名（可来自文件名或元数据） |
| mime\_type             | VARCHAR  | 原始文件类型          |
| created\_at            | DATETIME |                 |
| latest\_version\_id    | BIGINT   | 快速指向当前最新版本      |

**最少需含**：`id`, `user_id`, `doc_uuid`, `title`。

---

### 5.4 表：`document_versions`

| 字段                             | 类型                           | 说明                       |
| ------------------------------ | ---------------------------- | ------------------------ |
| id (PK)                        | BIGINT                       |                          |
| document\_id (FK documents.id) | BIGINT                       |                          |
| version\_label                 | VARCHAR                      | 人类可读版本号，如 `v1`, `2024Q4` |
| source\_uri                    | TEXT                         | S3 路径（原始文件）              |
| checksum                       | CHAR(64)                     | SHA256；用于去重              |
| effective\_date                | DATE                         | 内容“截至日期”（业务语义）           |
| uploaded\_at                   | DATETIME                     | 上传时间                     |
| parsed\_status                 | ENUM('pending','ok','error') | 解析状态                     |

**最少需含**：`id`, `document_id`, `source_uri`, `uploaded_at`。

---

### 5.5 表：`chunks`

| 字段                                     | 类型         | 说明                            |
| -------------------------------------- | ---------- | ----------------------------- |
| id (PK)                                | BIGINT     |                               |
| version\_id (FK document\_versions.id) | BIGINT     |                               |
| chunk\_uid                             | CHAR(36)   | 全局 chunk 标识（可用于引用）            |
| seq\_no                                | INT        | 在文档内顺序                        |
| section\_path                          | VARCHAR    | 层级路径：`1.Intro/1.2.Background` |
| page\_no                               | INT        | 原始页码（可空）                      |
| text                                   | MEDIUMTEXT | 清洗后的文本                        |
| token\_count                           | INT        | tokens（估算）                    |

**最少需含**：`id`, `version_id`, `seq_no`, `text`。

---

### 5.6 表：`embeddings`

| 字段                       | 类型       | 说明                                        |
| ------------------------ | -------- | ----------------------------------------- |
| id (PK)                  | BIGINT   |                                           |
| chunk\_id (FK chunks.id) | BIGINT   |                                           |
| model\_name              | VARCHAR  | 使用的 embedding 模型                          |
| dim                      | INT      | 向量维度                                      |
| vector\_ref              | VARCHAR  | Milvus collection + pk（或直接存 FAISS 索引 key） |
| created\_at              | DATETIME |                                           |

**最少需含**：`id`, `chunk_id`, `model_name`, `vector_ref`。

> **注意**：向量本体不存 SQL；仅存引用。Milvus 中需镜像字段：`embedding_id`, `chunk_uid`, `user_id`, `doc_uuid`, `version_label`, `timestamp`，便于过滤。

---

### 5.7 表：`chunk_relations`（可选 / V1.5）

| 字段             | 类型                                                      | 说明    |
| -------------- | ------------------------------------------------------- | ----- |
| id (PK)        | BIGINT                                                  |       |
| src\_chunk\_id | BIGINT                                                  |       |
| dst\_chunk\_id | BIGINT                                                  |       |
| relation\_type | ENUM('refers\_to','updates','duplicates','contradicts') |       |
| weight         | FLOAT                                                   | 关系置信度 |

用于构建 Graph-RAG；早期可空表。

---

### 5.8 表：`ingestion_jobs`（流水监控）

| 字段                    | 类型                                                        | 说明 |
| --------------------- | --------------------------------------------------------- | -- |
| id                    | BIGINT                                                    |    |
| user\_id              | BIGINT                                                    |    |
| document\_version\_id | BIGINT                                                    |    |
| phase                 | ENUM('upload','parse','chunk','embed','complete','error') |    |
| log                   | JSON                                                      |    |
| updated\_at           | DATETIME                                                  |    |

---

## 6. 存储层实现细节

### 6.1 本地 S3 模拟

**路径约定：**

```
./data/object_store/
  └── user_{user_id}/
        └── {doc_uuid}/
              └── v{n}/
                    original.{ext}
                    derived/
                       text.json
                       tables.parquet
```

**最少要求：**

* 原始文件保留原扩展名；不可覆盖（追加版本）。
* 上传即计算并落地 `sha256`；写入 `document_versions.checksum`。
* 解析产物（结构化 JSON、页面切片、图像切片）可放 `derived/` 子目录。

**必备接口（s3\_local.py）**

* `put_object(user_id, doc_uuid, version_label, file_stream) -> source_uri`
* `get_object(source_uri) -> bytes`
* `list_user_docs(user_id) -> [source_uri...]`
* `generate_local_url(source_uri) -> str` （调试用）

---

### 6.2 MySQL 元数据层

**连接配置**：`.env` 或 `config.yaml`；须支持连接池。

**最少接口（mysql\_client.py）**

* `create_document(user_id, title, mime_type) -> doc_id`
* `create_version(doc_id, source_uri, version_label, effective_date=None) -> version_id`
* `bulk_insert_chunks(version_id, chunk_records: List[ChunkIn])`
* `link_embedding(chunk_id, vector_ref, model_name)`
* `get_chunks(version_id, filters=...)`
* `resolve_latest_version(doc_uuid)`
* `fetch_metadata_for_chunks(chunk_ids) -> [{chunk,text,section,page,...}]`

**迁移管理**：使用 Alembic 或自写 SQL 脚本；模拟环境可在 `./db_server/sql/schema.sql` 提供一次建库脚本。

---

### 6.3 Milvus 向量库（当前配置）

**Collection 命名**：`rag_embeddings_v1`

**字段结构（Milvus Schema）**

| 字段             | 类型                    | 注释                        |
| -------------- | --------------------- | ------------------------- |
| embedding\_id  | Int64                 | PK，对应 SQL `embeddings.id` |
| user\_id       | Int64                 | 用于租户过滤                    |
| doc\_uuid      | VarChar(36)           | 文档标识                      |
| version\_label | VarChar(50)           | 版本过滤                      |
| chunk\_uid     | VarChar(36)           | 反查 chunk                  |
| vector         | FloatVector(dim=1536) | 1536维向量                   |
| ts             | Int64                 | 上传时间戳                     |

**索引配置**

* **索引类型**：HNSW（高性能近似最近邻搜索）
* **距离度量**：COSINE（余弦相似度）
* **索引参数**：M=16, efConstruction=200
* **搜索参数**：ef=200

**必备接口（milvus\_client.py）**

* `upsert_embedding(embedding_id, user_id, doc_uuid, version_label, chunk_uid, vector)`
* `search(vector, top_k, user_id=None, doc_uuid=None, version_label=None, ts_range=None) -> [hits]`
* `delete_embeddings(chunk_uid_list)`
* `create_collection_if_not_exists()`

---

## 7. 代码目录结构（当前实际结构）

```
repo_root/
├─ db_server/                    # 数据库服务和实验管理
│   ├─ docker-compose.yml        # 一键启动全部依赖
│   ├─ experiment_manager.py     # 实验管理工具
│   ├─ experiment_data.py        # 实验数据管理
│   ├─ experiment_schemas.py     # 实验模式管理
│   ├─ quick_start_example.py    # 快速开始示例
│   ├─ RAG_EXPERIMENT_GUIDE.md   # 实验指南
│   ├─ schema_templates/         # 模式模板
│   │   ├─ basic_rag.yaml
│   │   ├─ vector_experiment.yaml
│   │   └─ ...
│   └─ sql/                      # 数据库结构定义
│       └─ schema.sql
│
├─ src/                          # 核心源代码
│   └─ knowledge_rag/
│       ├─ __init__.py
│       ├─ config.py             # 配置管理（支持.env文件）
│       ├─ file_process/         # 文件处理模块
│       │   └─ __init__.py       # (待实现具体解析器)
│       ├─ knowledge_process/    # 知识处理模块
│       │   └─ __init__.py       # (待实现分块、嵌入等)
│       ├─ knowledge_retrieval/  # 知识检索模块
│       │   └─ __init__.py       # (待实现检索、重排等)
│       ├─ rag_chat/            # RAG 聊天模块
│       │   └─ __init__.py       # (待实现聊天代理)
│       └─ utils/               # 工具模块（已实现）
│           ├─ __init__.py
│           ├─ s3_local.py       # 本地S3模拟
│           ├─ mysql_client.py   # MySQL客户端
│           ├─ milvus_client.py  # Milvus客户端
│           ├─ flexible_search.py # 灵活搜索引擎
│           └─ logging_utils.py  # 日志工具
│
├─ .env                         # 环境变量配置
├─ experiments.md               # 本技术文档
├─ logs/                        # 日志目录
└─ data/                        # 数据目录
    ├─ local_object_store/      # 本地对象存储
    ├─ mysql/                   # MySQL数据
    └─ milvus/                  # Milvus数据
```

---

## 8. 各 Python 模块当前实现状态

> 下列为 **当前已实现** 和 **待实现** 的模块状态。

### 8.1 `config.py` ✅ **已实现**

* 自动加载 `.env` 文件和环境变量
* 提供完整的配置管理：数据库、Milvus、嵌入模型、Token预算等
* 支持 `get_settings()` 单例模式
* 默认1536维向量配置

### 8.2 `utils/s3_local.py` ✅ **已实现**

* 本地文件系统模拟S3存储
* 支持用户隔离的目录结构
* 文件上传/下载/列表功能
* 自动计算checksum和元数据管理

### 8.3 `utils/mysql_client.py` ✅ **已实现**

* 基于 mysql-connector-python 的连接池
* 文档/版本/块的完整CRUD操作
* 支持批量插入和事务管理
* 实现了 `get_mysql_client()` 单例

### 8.4 `utils/milvus_client.py` ✅ **已实现**

* 完整的Milvus集合管理（创建、索引、加载）
* 支持1536维向量的HNSW索引
* 向量搜索支持metadata过滤
* 批量向量插入和删除功能

### 8.5 `utils/flexible_search.py` ✅ **已实现**

* 灵活的搜索引擎，支持语义搜索、关键词搜索、混合搜索
* 自动从配置获取向量维度
* 支持多表搜索和实验环境切换

### 8.6 `utils/logging_utils.py` ✅ **已实现**

* 结构化日志记录系统
* 支持查询日志、性能日志、错误日志
* 多种日志级别和输出格式

### 8.7 `file_process/` ⏳ **待实现**

* 文档解析器（PDF、Docx、HTML、Markdown等）
* 结构化文本提取和元数据抽取
* 解析错误处理和状态跟踪

### 8.8 `knowledge_process/` ⏳ **待实现**

* 文本分块策略（语义分块、结构分块）
* 嵌入模型封装（OpenAI、BGE、Instructor等）
* 元数据管理和版本控制

### 8.9 `knowledge_retrieval/` ⏳ **待实现**

* 检索管线（过滤、KNN、重排）
* 上下文压缩和Token预算管理
* 引用追踪和租户安全过滤

### 8.10 `rag_chat/` ⏳ **待实现**

* RAG聊天代理
* 对话历史管理
* 智能检索策略

---

## 9. DB 启动脚本和实验管理（./db\_server/）

### 9.1 `docker-compose.yml` ✅ **已实现**

* **一键启动全部服务**：`docker compose up -d`
* **服务包含**：
  * MySQL 8.0.42：端口 3306，数据库 `knowledge_rag`
  * Milvus v2.4.15：端口 19530/9091，支持向量搜索
  * Python 3.12：应用运行环境
* **数据持久化**：所有数据保存在 `./db_server/data/` 目录
* **网络配置**：服务间通过内部网络 `knowledge_rag_network` 通信

### 9.2 `experiment_manager.py` ✅ **已实现**

* 创建、切换、删除实验
* 每个实验有独立的数据库前缀
* 支持交互式实验管理
* 实验配置文件管理

### 9.3 `experiment_data.py` ✅ **已实现**

* 统一数据管理器
* 支持MySQL和Milvus的数据操作
* 实验数据的创建、删除、重置
* 健康检查和状态监控

### 9.4 `experiment_schemas.py` ✅ **已实现**

* 实验模式管理器
* 支持多种预定义模式（basic_rag、vector_experiment等）
* 动态表结构创建和管理
* 模式模板系统

### 9.5 `quick_start_example.py` ✅ **已实现**

* 完整的RAG系统演示
* 文档上传、处理、向量化流程
* 搜索功能测试
* 端到端的系统验证

### 9.6 常用 Docker Compose 命令

```bash
# 启动服务
cd db_server && docker compose up -d

# 停止服务
docker compose down

# 查看服务状态
docker compose ps

# 查看日志
docker compose logs -f

# 重启服务
docker compose restart
```

---

## 10. 配置文件要求

### `.env` 文件 ✅ **已实现**

```bash
# KnowledgeRAG 环境变量配置
# 生产环境请使用安全的配置管理工具

# 数据库配置
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=devpass
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
EMBEDDING_MODEL=bge-small-zh-v1.5
EMBEDDING_DIMENSION=1536
EMBEDDING_BATCH_SIZE=32
EMBEDDING_MAX_SEQ_LENGTH=512
EMBEDDING_DEVICE=auto

# 对象存储配置
OBJECT_STORE_TYPE=local
OBJECT_STORE_BASE_PATH=./data/local_object_store
OBJECT_STORE_AUTO_CREATE_DIRS=true
OBJECT_STORE_EXPERIMENTS_DIR=experiments
OBJECT_STORE_MAX_FILE_SIZE=104857600  # 100MB

# Token 预算配置
MAX_CONTEXT_TOKENS=2048
CHUNK_MAX_TOKENS=350
TOP_K_RAW=20
TOP_M_RERANK=5
COMPRESSION_RATIO=0.5

# 检索配置
VECTOR_SIMILARITY_THRESHOLD=0.7
ENABLE_RERANK=true
RERANK_MODEL=bge-reranker-base
ENABLE_COMPRESSION=true

# 日志配置
LOG_LEVEL=INFO
LOG_DIR=./logs
LOG_MAX_SIZE=10485760  # 10MB
LOG_BACKUP_COUNT=5
LOG_ENABLE_CONSOLE=true
LOG_ENABLE_FILE=true

# 安全配置
SECRET_KEY=your-secret-key-here-change-in-production
ALLOWED_HOSTS=localhost,127.0.0.1
ENABLE_RATE_LIMITING=true
MAX_REQUESTS_PER_MINUTE=60

# 环境配置
ENVIRONMENT=development
DEBUG=true
```

### 配置管理特性

* **自动加载**：`config.py` 会自动加载 `.env` 文件
* **优先级**：环境变量 > .env 文件 > 默认值
* **类型安全**：所有配置项都有类型检查和验证
* **模块化**：按功能分组的配置类
* **向量维度统一**：默认1536维，所有模块从配置获取

> 生产环境建议使用 secret manager（AWS Secrets Manager / GCP Secret Manager）。

---

## 11. 检索与生成流程（Pseudo-code）

```python
from knowledge_rag.utils import retrieval, citation

results = retrieval.search(
    user_id=user_id,
    query=user_query,
    doc_filters={"doc_uuid": selected_docs},
    version_policy="latest",  # or "as_of_date"
    time_range=(None, cutoff_date),
    top_k=20,
    top_m=5,
)

context_blocks = results.context  # [{chunk_uid, text, ...}]
answer, raw = llm_answer(query=user_query, context=context_blocks)

answer_with_refs = citation.attach(answer, context_blocks)
return answer_with_refs
```

---

## 12. Token 成本控制策略（当前配置）

| 控制点               | 实现                   | 当前配置值    | 环境变量                    |
| ----------------- | -------------------- | -------- | ----------------------- |
| 向量召回上限            | top\_k\_raw          | 20       | TOP\_K\_RAW             |
| Rerank 裁剪         | top\_m               | 5        | TOP\_M\_RERANK          |
| 单块最大 token        | chunk\_max\_tokens   | 350      | CHUNK\_MAX\_TOKENS      |
| Context 压缩        | 压缩比                  | 0.5      | COMPRESSION\_RATIO      |
| LLM Prompt Budget | max\_context\_tokens | 2048     | MAX\_CONTEXT\_TOKENS    |
| 向量相似度阈值           | similarity\_threshold | 0.7      | VECTOR\_SIMILARITY\_THRESHOLD |

### 成本控制策略

**当候选块总 token 超预算时：**

1. **优先级排序**：高得分 + 最新版本块优先保留
2. **智能合并**：同章节块进行摘要合并
3. **渐进截断**：超限则截断并记录警告日志
4. **动态调整**：根据查询复杂度调整top_k参数

**配置灵活性：**
- 所有参数可通过 `.env` 文件调整
- 支持实验环境独立配置
- 实时配置重载（重启服务生效）

---

## 13. 测试策略

### 13.1 样本集

* 选择 5\~10 份真实格式（PDF, MD, HTML, PPTX mix）
* 手工标注：

  * 版本依赖问题（v1 vs v2）
  * 时间截断问题（“截至 2023”）
  * 跨文档引用问题（主文档+附录）

### 13.2 自动化测试

* `test_ingestion.py`：上传后应生成 doc/version/chunk/embedding 记录；checksum 唯一。
* `test_retrieval.py`：特定查询必须命中指定 chunk\_id。
* `test_citation.py`：回答中出现引用编号；编号可解析回 source\_uri。
* `test_token_budget.py`：超大文档时输出上下文 < 预算。

---

## 14. 开发工作流建议

### 快速开始流程 ✅

1. **启动依赖服务**：
   ```bash
   cd db_server && docker compose up -d
   ```

2. **安装Python依赖**：
   ```bash
   pip install mysql-connector-python pymilvus python-dotenv
   ```

3. **运行快速开始示例**：
   ```bash
   cd db_server && python quick_start_example.py
   ```

4. **查看系统状态**：
   ```bash
   # 查看服务状态
   docker compose ps
   
   # 查看日志
   docker compose logs -f
   
   # 检查数据库
   python experiment_data.py --action health-check
   ```

### 实验管理工作流 ✅

1. **创建新实验**：
   ```bash
   python experiment_manager.py --action create --name my_experiment
   ```

2. **切换实验环境**：
   ```bash
   python experiment_manager.py --action switch --name my_experiment
   ```

3. **管理实验数据**：
   ```bash
   # 创建实验数据
   python experiment_data.py --action create --experiment my_experiment
   
   # 重置实验数据
   python experiment_data.py --action reset --experiment my_experiment
   ```

### 开发和调试

1. **配置管理**：所有配置通过 `.env` 文件管理，支持热重载
2. **日志系统**：结构化日志保存在 `./logs/` 目录
3. **搜索调试**：使用 `flexible_search.py` 的命令行工具
4. **向量验证**：检查Milvus集合状态和维度配置

---

## 15. 生产化迁移注意事项（展望）

| 本地模拟           | 生产替代                          | 关键差异          |
| -------------- | ----------------------------- | ------------- |
| 本地 dir (S3 模拟) | AWS S3 / GCS / OSS            | IAM、加密、生命周期策略 |
| MySQL          | Aurora MySQL / Cloud SQL      | 高可用、读写分离、备份   |
| Milvus 本地      | Zilliz Cloud / Managed Milvus | 托管缩放、监控、分区策略  |
| Plain Text     | 加密静态存储                        | KMS 集成        |
| 手工租户隔离         | IAM + row-level policy        | 合规性、审计        |

---

## 16. 名词索引（Glossary）

* **Chunk**：经过语义或结构切分的最小检索单元，通常 150–400 tokens。
* **Embedding**：Chunk 文本经编码模型映射到向量空间的浮点数组，当前系统使用1536维。
* **Version Policy**：检索时关于文档版本选择的策略，如 *latest*, *as\_of\_date*, *all\_versions*。
* **Context Compression**：在不显著损失语义覆盖的情况下压缩送入 LLM 的上下文文本量的技术方法。
* **实验环境**：独立的数据库命名空间，用于隔离不同的实验数据和配置。
* **FlexibleSearch**：支持多种搜索策略（语义、关键词、混合）的统一搜索引擎。
* **Vector Dimension**：向量维度，系统统一配置为1536维，可通过环境变量调整。
* **HNSW**：分层可导航小世界图算法，用于高效的近似最近邻搜索。
* **Milvus Collection**：Milvus中的数据集合，类似于关系数据库中的表。

---

## 17. 后续工作（TODO）

### 已完成 ✅

* [x] 统一配置管理（.env + config.py）
* [x] 基础工具模块（MySQL、Milvus、S3本地存储）
* [x] 向量维度统一管理（1536维）
* [x] 实验管理系统
* [x] 快速开始示例
* [x] 日志系统
* [x] 灵活搜索引擎

### 正在进行 🔄

* [ ] 文档解析模块（`file_process/`）
* [ ] 知识处理模块（`knowledge_process/`）
* [ ] 检索管线优化（`knowledge_retrieval/`）

### 计划中 📋

* [ ] 真实嵌入模型集成（替换随机向量）
* [ ] Cross-Encoder rerank 精度实验
* [ ] Token 预算压缩策略 A/B 测试
* [ ] Graph-RAG 支持（chunk\_relations）
* [ ] Web界面和API接口
* [ ] 性能监控和指标收集

### 技术债务 ⚠️

* [ ] 单元测试覆盖率提升
* [ ] 错误处理和异常管理
* [ ] 生产环境安全配置
* [ ] 数据库迁移工具（Alembic）
* [ ] 容器化部署优化

---

## 📝 文档更新记录

### v0.2 (2025-07-18)
- 🔄 **与实际代码同步**：更新所有模块状态为当前真实情况
- ✅ **已实现功能标记**：明确标识已完成和待实现的功能
- 📊 **配置文件更新**：反映当前完整的.env配置
- 🔧 **工作流更新**：更新为当前实际可用的命令和流程
- 📐 **向量维度统一**：文档全面更新为1536维配置
- 🎯 **项目状态明确**：增加当前项目状态说明

### v0.1 (2025-07-18)
- 📋 **初始架构设计**：定义系统总体架构和数据模型
- 📁 **目录结构规划**：制定代码组织结构
- 🔗 **接口定义**：明确各模块间的接口规范

---

> 💡 **提示**：本文档将与项目开发保持同步更新，确保准确反映系统当前状态。如发现文档与实际代码不符，请及时反馈。

