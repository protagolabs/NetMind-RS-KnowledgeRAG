# KnowledgeRAG 实验系统深度解析与RAG实验设计指南

## 🎯 系统架构原理

### 📊 系统分层架构

KnowledgeRAG 实验系统采用**分层架构**设计，让您可以：
- **隔离不同实验**：每个实验有独立的数据环境
- **快速切换实验**：无需重新配置数据库连接
- **标准化流程**：统一的实验管理和数据操作接口

### 🔧 核心组件解析

#### 1. **实验管理层**
- **experiment_manager.py**：用户友好的实验管理界面
- **experiment_data.py**：底层数据管理器，支持多数据源
- **experiment_schemas.py**：数据库表结构模板管理

#### 2. **数据存储层**
- **MySQL**：存储文档元数据、版本信息、文本块（chunks）
- **Milvus**：存储向量embeddings，支持语义搜索
- **本地对象存储**：存储原始文件（PDF、DOC、图片等）

#### 3. **应用工具层**
- **mysql_client.py**：MySQL操作封装
- **milvus_client.py**：Milvus向量操作封装
- **s3_local.py**：本地文件存储操作
- **flexible_search.py**：统一搜索接口

---

## 🚀 实验管理的核心价值

### 🎯 为什么需要实验管理？

#### 1. **数据隔离**
```bash
# 不同实验有独立的数据环境
实验A: knowledge_rag_experiment_a
实验B: knowledge_rag_experiment_b
```

#### 2. **版本管理**
```bash
# 同一个实验可以有多个版本
knowledge_rag_my_experiment/
├── v1.0/  # 第一版实验
├── v1.1/  # 改进版本
└── v2.0/  # 重大更新
```

#### 3. **快速切换**
```bash
# 无需重新配置，一键切换实验环境
python experiment_manager.py --switch experiment_a
python experiment_manager.py --switch experiment_b
```

#### 4. **标准化流程**
- 统一的数据表结构
- 标准化的embedding存储
- 一致的搜索接口

---

## 🔬 完整的RAG实验设计流程

### 📋 Phase 1: 实验规划设计

#### 1.1 定义实验目标
```python
# 例：设计一个多文档问答系统
实验名称: "multi_doc_qa_v1"
研究目标: 测试不同embedding模型在多文档检索中的效果
评估指标: 准确率、召回率、响应时间
```

#### 1.2 设计数据结构
```sql
-- 规划需要的表结构
documents: 文档基本信息
document_versions: 文档版本管理
chunks: 文本分块存储
embeddings: 向量索引映射
search_logs: 搜索日志记录
```

### 📋 Phase 2: 创建实验环境

#### 2.1 创建实验
```bash
# 方式1: 交互式创建（推荐）
python experiment_manager.py --interactive

# 方式2: 命令行创建
python experiment_manager.py --create multi_doc_qa_v1 \
  --researcher "张三" \
  --description "多文档问答系统实验" \
  --template basic_rag
```

#### 2.2 验证实验环境
```bash
# 检查实验创建结果
python experiment_manager.py --info multi_doc_qa_v1

# 检查数据存储状态
python experiment_data.py --action health-check
```

### 📋 Phase 3: 数据上传与处理

#### 3.1 文档上传实现
```python
# 使用 s3_local.py 上传文档
from knowledge_rag.utils.s3_local import S3LocalClient

s3_client = S3LocalClient()

# 上传文档
def upload_document(user_id: int, doc_uuid: str, version: str, 
                   file_path: str, filename: str):
    """上传文档到本地对象存储"""
    with open(file_path, 'rb') as file:
        uri = s3_client.put_object(
            user_id=user_id,
            doc_uuid=doc_uuid,
            version_label=version,
            filename=filename,
            file_stream=file,
            content_type="application/pdf"
        )
    return uri
```

#### 3.2 文档元数据存储
```python
# 使用 mysql_client.py 存储元数据
from knowledge_rag.utils.mysql_client import get_mysql_client

mysql_client = get_mysql_client()

# 创建文档记录
doc_id = mysql_client.create_document(
    user_id=1,
    title="技术文档.pdf",
    mime_type="application/pdf"
)

# 创建版本记录
version_id = mysql_client.create_version(
    doc_id=doc_id,
    source_uri=uri,  # 来自S3上传的URI
    version_label="v1.0",
    checksum="sha256_hash_value"
)
```

#### 3.3 文本分块处理
```python
# 文档解析和分块
def process_document(doc_id: int, version_id: int, file_content: str):
    """处理文档：解析、分块、存储"""
    
    # 1. 文档解析（示例）
    chunks = split_text_into_chunks(file_content, chunk_size=512)
    
    # 2. 存储chunks到MySQL
    chunk_records = []
    for i, chunk_text in enumerate(chunks):
        chunk_uid = str(uuid.uuid4())
        chunk_record = ChunkIn(
            seq_no=i,
            chunk_uid=chunk_uid,
            text=chunk_text,
            token_count=len(chunk_text.split())
        )
        chunk_records.append(chunk_record)
    
    # 批量插入chunks
    mysql_client.create_chunks(version_id, chunk_records)
    
    return chunk_records
```

### 📋 Phase 4: 向量化与索引

#### 4.1 生成Embeddings
```python
# 使用 milvus_client.py 存储向量
from knowledge_rag.utils.milvus_client import get_milvus_client
import numpy as np

milvus_client = get_milvus_client()

def generate_embeddings(chunk_records: List[ChunkIn], doc_uuid: str, version: str):
    """生成并存储向量embeddings"""
    
    for chunk in chunk_records:
        # 1. 生成embedding向量（示例使用随机向量）
        # 实际应用中使用 OpenAI、Sentence-BERT 等模型
        embedding_vector = np.random.rand(768).tolist()  # 768维向量
        
        # 2. 存储到Milvus
        success = milvus_client.upsert_embedding(
            embedding_id=hash(chunk.chunk_uid),
            user_id=1,
            doc_uuid=doc_uuid,
            version_label=version,
            chunk_uid=chunk.chunk_uid,
            vector=embedding_vector
        )
        
        if success:
            print(f"✅ 向量存储成功: {chunk.chunk_uid}")
```

### 📋 Phase 5: 搜索功能实现

#### 5.1 语义搜索实现
```python
# 使用 flexible_search.py 进行搜索
from knowledge_rag.utils.flexible_search import FlexibleSearchEngine, SearchQuery

def semantic_search(query_text: str, experiment_name: str, top_k: int = 5):
    """执行语义搜索"""
    
    # 1. 初始化搜索引擎
    search_engine = FlexibleSearchEngine(experiment_name=experiment_name)
    
    # 2. 构造搜索查询
    query = SearchQuery(
        query_text=query_text,
        query_type="semantic",
        top_k=top_k,
        threshold=0.7,
        experiment_name=experiment_name
    )
    
    # 3. 执行搜索
    results = search_engine.search(query)
    
    # 4. 返回结果
    return results
```

#### 5.2 混合搜索实现
```python
def hybrid_search(query_text: str, filters: Dict[str, Any], top_k: int = 5):
    """混合搜索：向量搜索 + 关键词搜索"""
    
    # 1. 向量搜索
    vector_results = semantic_search(query_text, "multi_doc_qa_v1", top_k)
    
    # 2. 关键词搜索
    keyword_query = SearchQuery(
        query_text=query_text,
        query_type="keyword",
        filters=filters,
        top_k=top_k
    )
    
    search_engine = FlexibleSearchEngine("multi_doc_qa_v1")
    keyword_results = search_engine.search(keyword_query)
    
    # 3. 结果融合
    merged_results = merge_search_results(vector_results, keyword_results)
    
    return merged_results
```

### 📋 Phase 6: 实验测试与评估

#### 6.1 创建测试集
```python
# 创建评估测试集
test_cases = [
    {
        "question": "什么是机器学习？",
        "expected_docs": ["doc1", "doc2"],
        "ground_truth": "机器学习是人工智能的一个分支..."
    },
    {
        "question": "深度学习的优势是什么？",
        "expected_docs": ["doc3"],
        "ground_truth": "深度学习具有自动特征提取能力..."
    }
]
```

#### 6.2 批量测试
```python
def run_experiment_test(test_cases: List[Dict], experiment_name: str):
    """运行实验测试"""
    
    results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"\n🔍 测试用例 {i+1}: {test_case['question']}")
        
        # 1. 执行搜索
        search_results = semantic_search(
            query_text=test_case['question'],
            experiment_name=experiment_name,
            top_k=5
        )
        
        # 2. 评估结果
        evaluation = evaluate_search_results(search_results, test_case)
        
        # 3. 记录结果
        results.append({
            'test_case': test_case,
            'search_results': search_results,
            'evaluation': evaluation
        })
        
        print(f"   准确率: {evaluation['accuracy']:.2f}")
        print(f"   召回率: {evaluation['recall']:.2f}")
    
    return results
```

#### 6.3 性能评估
```python
def evaluate_search_results(search_results: List[SearchResult], test_case: Dict):
    """评估搜索结果质量"""
    
    # 计算准确率
    retrieved_docs = [result.metadata.get('doc_uuid') for result in search_results]
    expected_docs = test_case['expected_docs']
    
    # 计算指标
    precision = len(set(retrieved_docs) & set(expected_docs)) / len(retrieved_docs)
    recall = len(set(retrieved_docs) & set(expected_docs)) / len(expected_docs)
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': precision,
        'recall': recall,
        'f1_score': f1_score,
        'retrieved_docs': retrieved_docs,
        'expected_docs': expected_docs
    }
```

---

## 🎯 实际操作示例

### 📝 完整的RAG实验实现

```python
#!/usr/bin/env python3
"""
完整的RAG实验示例
实验名称: multi_doc_qa_v1
目标: 测试多文档问答系统的效果
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any

# 导入工具模块
from knowledge_rag.utils.s3_local import S3LocalClient
from knowledge_rag.utils.mysql_client import get_mysql_client
from knowledge_rag.utils.milvus_client import get_milvus_client
from knowledge_rag.utils.flexible_search import FlexibleSearchEngine, SearchQuery

class RAGExperiment:
    """RAG实验类"""
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.s3_client = S3LocalClient()
        self.mysql_client = get_mysql_client()
        self.milvus_client = get_milvus_client()
        self.search_engine = FlexibleSearchEngine(experiment_name)
    
    def upload_and_process_document(self, file_path: str, user_id: int = 1):
        """上传并处理文档"""
        
        # 1. 上传文档
        filename = Path(file_path).name
        doc_uuid = str(uuid.uuid4())
        
        with open(file_path, 'rb') as file:
            uri = self.s3_client.put_object(
                user_id=user_id,
                doc_uuid=doc_uuid,
                version_label="v1.0",
                filename=filename,
                file_stream=file
            )
        
        # 2. 创建文档记录
        doc_id = self.mysql_client.create_document(
            user_id=user_id,
            title=filename,
            mime_type="application/pdf"
        )
        
        # 3. 创建版本记录
        version_id = self.mysql_client.create_version(
            doc_id=doc_id,
            source_uri=uri,
            version_label="v1.0",
            checksum="sha256_hash"
        )
        
        # 4. 处理文档内容
        content = self.extract_text_from_pdf(file_path)
        chunks = self.split_text_into_chunks(content)
        
        # 5. 存储chunks和embeddings
        self.store_chunks_and_embeddings(version_id, chunks, doc_uuid)
        
        return doc_id, version_id
    
    def test_search_functionality(self, test_questions: List[str]):
        """测试搜索功能"""
        
        for question in test_questions:
            print(f"\n🔍 测试问题: {question}")
            
            # 执行搜索
            query = SearchQuery(
                query_text=question,
                query_type="semantic",
                top_k=3,
                experiment_name=self.experiment_name
            )
            
            results = self.search_engine.search(query)
            
            # 显示结果
            for i, result in enumerate(results, 1):
                print(f"   {i}. 得分: {result.score:.3f}")
                print(f"      内容: {result.content[:100]}...")
                print(f"      来源: {result.metadata.get('doc_uuid', 'unknown')}")

# 使用示例
if __name__ == "__main__":
    # 1. 创建实验
    experiment = RAGExperiment("multi_doc_qa_v1")
    
    # 2. 上传测试文档
    docs_folder = "./test_documents/"
    for doc_file in Path(docs_folder).glob("*.pdf"):
        experiment.upload_and_process_document(str(doc_file))
    
    # 3. 测试搜索
    test_questions = [
        "什么是机器学习？",
        "深度学习的主要应用领域有哪些？",
        "如何评估模型性能？"
    ]
    
    experiment.test_search_functionality(test_questions)
```

---

## 🎯 实验管理最佳实践

### 📊 1. 实验命名规范
```bash
# 推荐命名格式
{项目名}_{实验类型}_{版本号}
例如：
- qa_system_v1
- search_engine_baseline
- embedding_comparison_v2
```

### 📊 2. 版本控制策略
```bash
# 版本控制建议
v1.0  # 初始版本
v1.1  # 小幅改进
v2.0  # 重大更新
```

### 📊 3. 实验数据管理
```bash
# 定期备份重要实验
python experiment_data.py --action backup-exp --experiment important_experiment

# 清理过期实验
python experiment_manager.py --delete old_experiment --force
```

---

## 🚀 进阶功能

### 🔧 自定义表结构
```python
# 在 experiment_schemas.py 中定义自定义模板
custom_schema = {
    "name": "advanced_rag",
    "tables": {
        "documents": {
            "fields": [
                {"name": "id", "type": "INT", "primary": True},
                {"name": "title", "type": "VARCHAR(255)"},
                {"name": "category", "type": "VARCHAR(100)"},
                {"name": "priority", "type": "INT"},
                {"name": "created_at", "type": "TIMESTAMP"}
            ]
        },
        "custom_embeddings": {
            "fields": [
                {"name": "id", "type": "INT", "primary": True},
                {"name": "doc_id", "type": "INT"},
                {"name": "embedding_model", "type": "VARCHAR(50)"},
                {"name": "vector_data", "type": "JSON"}
            ]
        }
    }
}
```

### 🔧 多模型比较实验
```python
# 比较不同embedding模型的效果
models = ["sentence-bert", "openai-ada", "custom-model"]

for model in models:
    experiment_name = f"embedding_comparison_{model}"
    # 创建实验并测试
    run_embedding_experiment(model, experiment_name)
```

---

## 📋 常见问题解答

### ❓ Q: 实验之间的数据会互相影响吗？
**A**: 不会。每个实验有独立的数据库和存储目录，完全隔离。

### ❓ Q: 如何在实验之间共享数据？
**A**: 可以使用备份/恢复功能，或者设计共享数据表结构。

### ❓ Q: 实验删除后能恢复吗？
**A**: 删除前建议先备份。删除操作会清理所有相关数据。

### ❓ Q: 如何监控实验性能？
**A**: 使用健康检查功能和自定义日志记录。

---

## 🎯 总结

KnowledgeRAG 实验系统为您提供了：

1. **完整的数据管理**：MySQL + Milvus + 本地存储
2. **灵活的实验环境**：快速创建、切换、删除实验
3. **标准化的工具链**：统一的数据操作和搜索接口
4. **可扩展的架构**：支持自定义表结构和搜索算法

通过这个系统，您可以：
- 快速验证RAG算法效果
- 比较不同模型性能
- 管理复杂的实验数据
- 复现和分享实验结果

**开始您的RAG实验之旅吧！** 🚀 