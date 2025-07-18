# KnowledgeRAG 本地数据库 & 存储环境技术说明（NotebookLM RAG 能力逆向工程）

**版本：** Draft v0.1 
**日期：** 2025-07-18
**作者：** XYZ-Algorith-Team (Chengyu Huang, Bin Liang, Hongyi Gu, Yujing Wang, Guocheng Hu)
**适用范围：** 本文档适用于在本地/开发环境中模拟 NotebookLM 风格的多文档 RAG（Retrieval-Augmented Generation）系统，以增强平台内 Agent 的知识访问能力。生产环境部署需基于本文档作适配与加固。

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

### 6.3 Milvus 向量库（Zilliz 本地版）

**Collection 命名建议**：`rag_embeddings_v1`

**字段结构（Milvus Schema）**

| 字段             | 类型                 | 注释                        |
| -------------- | ------------------ | ------------------------- |
| embedding\_id  | Int64              | PK，对应 SQL `embeddings.id` |
| user\_id       | Int64              | 用于租户过滤                    |
| doc\_uuid      | VarChar            | 文档标识                      |
| version\_label | VarChar            | 版本过滤                      |
| chunk\_uid     | VarChar            | 反查 chunk                  |
| vector         | FloatVector(dim=D) | 向量                        |
| ts             | Int64              | 上传时间戳                     |

**索引参数建议**

* HNSW 或 IVF\_FLAT；D<1600 时 HNSW 性能良好。
* Metric：COSINE 或 IP（与 embedding 模型一致）。

**必备接口（milvus\_client.py）**

* `upsert_embedding(embedding_id, user_id, doc_uuid, version_label, chunk_uid, vector)`
* `search(vector, top_k, user_id=None, doc_uuid=None, version_label=None, ts_range=None) -> [hits]`
* `delete_embeddings(chunk_uid_list)`
* `create_collection_if_not_exists()`

---

## 7. 代码目录结构（Bin Liang 暂时的计划）

```
repo_root/
├─ db_server/
│   ├─ docker-compose.yml     # 一键启动全部依赖
│   ├─ experiment_manager.py  # 实验管理工具
│   ├─ manage_table.py        # 数据库管理工具
│   └─ sql/                   # 数据库结构定义
│
├─ src/
│   └─ knowledge_rag/
│       ├─ __init__.py
│       ├─ file_process/
│       │   └─ parsing.py     # 文档解析入口
│       ├─ knowledge_process/
│       │   ├─ chunking.py    # 分块策略
│       │   ├─ embeddings.py  # 嵌入模型包装
│       │   └─ metadata.py    # 写入/查询文档元数据
│       ├─ knowledge_retrieval/
│       │   ├─ retriebal_agent.py # 使用 agentic search, 去生成不同的
│       │   ├─ retrieval.py   # 查询管线 (filter + KNN + rerank)
│       │   ├─ compression.py # context 压缩
│       │   ├─ citation.py    # chunk->source 引用封装
│       │   └─ tenancy.py     # user_id 安全过滤
│       ├─ rag_chat/
│       │   └─ rag_chat_agent.py
│       ├─ config.py          # 读取 env；集中配置端口/路径/模型
│       ├─ utils/
│       │   ├─ s3_local.py
│       │   ├─ mysql_client.py
│       │   ├─ milvus_client.py
│       │   └─ logging_utils.py
│       ├─ ingestion_pipeline.py
│       ├─ qa_pipeline.py
└─      └─ demo_app.py        # 最小 web / CLI 示例
```

---

## 8. 各 Python 模块最少应包含内容

> 下列为 **MVP 必备接口**。更多扩展接口可随迭代添加。

### 8.1 `config.py`

* 读取 `.env`、环境变量、或 `config.yaml`。
* 导出：DB 连接串、Milvus host/port、数据目录根路径、默认 embedding 模型名、最大 token 预算。
* 提供 `get_settings()` 单例。

### 8.2 `s3_local.py`

* 本地路径到逻辑 URI 映射函数。
* 文件写入/读取/列目录。
* 计算并返回 checksum。
* 权限检查：校验 user\_id 所有权。

### 8.3 `mysql_client.py`

* 基础连接池（SQLAlchemy 或 pymysql）。
* 文档/版本/块的 CRUD。
* 查询最新版本：`get_latest_version(doc_uuid)`。
* 根据过滤条件批量获取 chunk 元数据。
* 事务封装（bulk insert）。

### 8.4 `milvus_client.py`

* Collection 初始化；schema 检查。
* upsert/search/delete。
* top\_k 搜索带 metadata 过滤。
* 批量向量插入（embedding worker 调用）。

### 8.5 `parsing.py`

* 根据 MIME 路由到相应解析器（PDF、Docx、HTML、Markdown…）。
* 返回结构化块（段落、标题、页码、表格分离）。
* 捕获解析错误并写 ingestion\_jobs。

### 8.6 `chunking.py`

* 支持两级策略：结构优先（按标题/节）、长度回退（按 token 限制切分）。
* 重叠边界（sliding overlap tokens）。
* 输出 `ChunkIn` dataclass：{seq\_no, text, section\_path, page\_no, token\_count}。

### 8.7 `embeddings.py`

* 模型注册：OpenAI, local BGE, Instructor 等。
* 批量向量化（返回 numpy 数组）。
* 调用 `milvus_client.upsert_embedding()`。

### 8.8 `metadata.py`

* 高层封装：`register_new_document(upload_info)`。
* `record_chunks(version_id, chunks)`。
* `get_context_for_chunk_ids(chunk_ids)`。

### 8.9 `retrieval.py`

* 输入：`user_id`, `query`, {doc\_filters, time\_range, version\_policy}\`。
* 步骤：

  1. Query embedding
  2. Metadata filter（user\_id 必填）
  3. Milvus KNN top\_k\_raw
  4. 可选 BM25/Lucene hybrid merge
  5. Cross-Encoder rerank -> top\_m
  6. Token 预算裁剪
  7. 返回 chunk payloads

### 8.10 `compression.py`

* 可选：Mini-LLM 摘要每个 chunk；或句子抽取（TextRank）。
* 控制总 token <= budget。

### 8.11 `citation.py`

* 将检索得到的块映射为 `[^{n}]` 式引用。
* 支持 chunk→版本→文档→源文件路径的级联。
* JSON Schema：

  ```json
  {
    "citation_id": 1,
    "chunk_uid": "...",
    "doc_title": "...",
    "version_label": "v2",
    "page_no": 17,
    "source_uri": "s3://..."
  }
  ```

### 8.12 `tenancy.py`

* 每次读取/检索必须传入 `user_id`。
* 统一断言：结果集中所有记录都匹配 user\_id；否则抛异常。

### 8.13 `logging_utils.py`

* 统一结构化日志；记录 query、检索耗时、top\_k 命中、token 用量。

---

## 9. DB 启动脚本（./db\_server/）最少内容要求

### 9.1 通用要求

* 可重复执行（幂等）。
* 检查依赖容器 / 进程是否已运行。
* 指定端口、挂载卷路径（持久化）。
* 打印连接信息（host/port/user/password）。

### 9.2 `docker-compose.yml`（统一服务管理）

* **一键启动全部服务**：`docker-compose up -d`
* **服务包含**：
  * MySQL 8.0.42：端口 3306，数据库 `knowledge_rag`
  * Milvus v2.5.14：端口 19530/9091，支持向量搜索
  * MinIO：端口 9000/9001，S3 兼容对象存储
  * Python 3.12：应用运行环境
* **数据持久化**：所有数据保存在 `./data/` 目录
* **网络配置**：服务间通过内部网络 `knowledge_rag_network` 通信

### 9.3 `manage_table.py`

* 因为是实验性的项目，会很频繁的修改数据结构。
* 使用 python 去清空对应数据库，建立新的 table 什么的。都要有。

### 9.4 `experiment_manager.py`

* 创建、切换、删除实验。
* 每个实验有独立的数据库前缀。

### 9.5 常用 Docker Compose 命令

```bash
# 启动服务
cd db_server && docker-compose up -d

# 停止服务
docker-compose down

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f

# 重启服务
docker-compose restart
```

---

## 10. 配置文件要求

### `.env` 示例（开发用）

```
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=devpass
MYSQL_DB=knowledge_rag

MILVUS_HOST=127.0.0.1
MILVUS_PORT=19530

OBJECT_STORE_ROOT=./data/object_store
EMBEDDING_MODEL=bge-small-zh-v1.5
MAX_CONTEXT_TOKENS=2048
```

> 生产环境需使用 secret manager（AWS Secrets Manager / GCP Secret Manager）。

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

## 12. Token 成本控制策略（集成要求）

| 控制点               | 实现                   | 默认值      |
| ----------------- | -------------------- | -------- |
| 向量召回上限            | top\_k\_raw          | 20       |
| Rerank 裁剪         | top\_m               | 5        |
| 单块最大 token        | chunk\_max\_tokens   | 350      |
| Context 压缩        | 压缩比                  | 0.5（必要时） |
| LLM Prompt Budget | max\_context\_tokens | 2048     |

当候选块总 token 超预算：

1. 优先保留高得分 + 最新版本块；
2. 同章节块合并摘要；
3. 超限则截断并记警告日志。

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

1. **启动依赖**：`cd db_server && docker-compose up -d`。
2. **载入样本文档**：`scripts/ingest_sample_docs.py`。
3. **运行交互式 QA Demo**：`python -m src.knowledge_rag.demo_app`。
4. **运行单元测试**：`pytest -q`。
5. **观察指标**：检索召回率、token 使用日志、响应延迟。

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
* **Embedding**：Chunk 文本经编码模型映射到向量空间的浮点数组。
* **Version Policy**：检索时关于文档版本选择的策略，如 *latest*, *as\_of\_date*, *all\_versions*。
* **Context Compression**：在不显著损失语义覆盖的情况下压缩送入 LLM 的上下文文本量的技术方法。

---

## 17. 后续工作（TODO）

* [ ] 表结构初稿 -> Alembic migration
* [ ] MinIO vs 本地目录选型
* [ ] Embedding 模型切换性能对比
* [ ] Cross-Encoder rerank 精度实验
* [ ] Token 预算压缩策略 A/B 测试
* [ ] Graph-RAG 支持（chunk\_relations）

