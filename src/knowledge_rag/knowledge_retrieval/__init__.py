"""
知识检索模块 - Knowledge Retrieval Module
====================================

本模块实现了RAG系统的核心检索功能，提供多种检索策略和智能匹配机制，
是连接知识索引和答案生成的关键桥梁。

核心功能：
1. 多层次检索：文档级和片段级的层次化检索
2. 混合搜索：结合向量搜索和全文搜索的优势
3. 智能匹配：基于LLM的相关性判断
4. 高效检索：并行处理和优化策略

模块组成：
- db_retriever.py：核心检索引擎，实现多种检索策略
- db_retriever_client.py：HTTP客户端，提供API接口封装
- doc_matching.py：文档级智能相关性判断
- chunk_matching.py：片段级智能相关性判断

检索策略：
1. 向量搜索：基于语义相似度的深度理解
2. 全文搜索：基于关键词的精确匹配
3. 混合搜索：结合两种策略的优势
4. 两阶段检索：先粗筛选后精匹配

技术特点：
- 双数据库架构（MySQL + Milvus）
- 异步并发处理
- 智能相关性评估
- 可配置的检索参数
- 完整的错误处理机制

应用场景：
- 问答系统的上下文检索
- 文档推荐和发现
- 内容搜索和过滤
- 知识库查询

工作流程：
用户查询 → 查询向量化 → 多策略检索 → 相关性判断 → 结果排序 → 返回匹配内容

作者: NetMind-RS-KnowledgeRAG Team
"""

# 导入核心检索功能
from knowledge_rag.knowledge_retrieval.db_retriever import (
    DBRetriever,
    create_retriever
)

# 导入HTTP客户端
from .db_retriever_client import (
    KnowledgeRAGClient,
    KnowledgeRAGClientError,
    APIConnectionError,
    APIResponseError,
    KnowledgeRAGValidationError,
    # 请求模型
    UploadRequest,
    EmbeddingRequest,
    SaveToDBRequest,
    DocRetrievalRequest,
    DocRetrievalByDataSetRequest,
    ChunkRetrievalRequest,
    ChunkDecisionRequest
)

# 导入匹配模块
from .doc_matching import (
    DocMatchingResult,
    DocMatchingResultWithIsRelated,
    doc_matching,
    make_decision_of_doc_matching
)

from .chunk_matching import (
    ChunkMatchingResult,
    ChunkMatchingResultWithIsRelated,
    chunk_matching,
    make_decision_of_chunk_matching
)

# 定义模块的公共接口
__all__ = [
    # 核心检索引擎
    'DBRetriever',
    'create_retriever',
    
    # HTTP客户端和异常
    'KnowledgeRAGClient',
    'KnowledgeRAGClientError',
    'APIConnectionError', 
    'APIResponseError',
    'KnowledgeRAGValidationError',
    
    # 请求模型
    'UploadRequest',
    'EmbeddingRequest',
    'SaveToDBRequest',
    'DocRetrievalRequest',
    'DocRetrievalByDataSetRequest',
    'ChunkRetrievalRequest',
    'ChunkDecisionRequest',
    
    # 匹配结果模型和函数
    'DocMatchingResult',
    'DocMatchingResultWithIsRelated',
    'doc_matching',
    'make_decision_of_doc_matching',
    'ChunkMatchingResult',
    'ChunkMatchingResultWithIsRelated',
    'chunk_matching',
    'make_decision_of_chunk_matching'
]
