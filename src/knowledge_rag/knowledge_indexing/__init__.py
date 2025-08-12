"""
知识索引模块 - Knowledge Indexing Module
====================================

本模块负责将处理好的文档分析结果转换为可搜索的知识索引，
是整个RAG系统中连接文档处理和知识检索的关键环节。

核心功能：
1. 向量化处理：将文档和文档块的文本内容转换为向量表示
2. 数据库存储：将向量化结果保存到MySQL和Milvus数据库
3. 索引构建：创建高效的搜索索引结构
4. 批量处理：支持大规模文档集合的高效处理

模块组成：
- step_1_get_embedding.py：向量化嵌入生成
- step_2_save_to_db.py：数据库存储和索引构建

处理流程：
文档分析结果 → 向量化处理 → 数据库存储 → 索引构建 → 可搜索知识库

技术特点：
- 异步并发处理提高效率
- 分层存储策略优化性能
- 错误恢复机制确保稳定性
- 进度跟踪便于监控

应用场景：
- 知识库初始化
- 增量内容更新
- 大规模文档索引
- 企业知识管理

作者: NetMind-RS-KnowledgeRAG Team
"""

# 导入核心功能模块
from .step_1_get_embedding import (
    get_data,
    get_embedding_normal,
    embedding_doc,
    embedding_chunk,
    DocAnalysisWithEmbedding,
    ChunkAnalysisWithEmbedding
)

from .step_2_save_to_db import (
    DataToDBSaver
)

# 定义模块的公共接口
__all__ = [
    # 数据加载和向量化
    'get_data',
    'get_embedding_normal',
    'embedding_doc', 
    'embedding_chunk',
    
    # 数据模型
    'DocAnalysisWithEmbedding',
    'ChunkAnalysisWithEmbedding',
    
    # 数据库存储
    'DataToDBSaver'
]
