"""
KnowledgeRAG 数据库工具模块 - RAG系统专用

这个模块为RAG系统提供了完整的两级数据库架构支持：

📊 两级数据库设计：
1. 文档级：存储文档总结、关键词、文档级embedding
2. Chunk级：每个文档都有独立的MySQL表和Milvus集合存储chunks

🔍 两级搜索流程：
1. 第一步：文档级搜索 -> 找到相关文档
2. 第二步：并行搜索相关文档的chunks -> 获取最终结果

🛠️ 核心组件：
- RAGDocumentManager: 管理文档生命周期和chunk级数据库创建
- RAGSearchEngine: 实现两级搜索逻辑
- ChunkDataManager: 提供便捷的chunk数据操作
- RAGSchemaConfig: 灵活的schema配置管理

使用示例：
    # 1. 创建文档管理器
    from knowledge_rag.db_utils import RAGDocumentManager, RAGSearchEngine, ChunkDataManager
    
    rag_manager = RAGDocumentManager()
    search_engine = RAGSearchEngine()
    chunk_manager = ChunkDataManager()
    
    # 2. 添加文档（自动创建chunk级数据库）
    doc_id = rag_manager.add_document(
        title="AI研究论文",
        summary="这是一篇关于AI的研究论文",
        keywords=["AI", "machine learning"],
        document_embedding=[0.1, 0.2, 0.3, ...]
    )
    
    # 3. 添加chunks到文档
    chunk_id = chunk_manager.add_chunk_to_document(
        doc_id=doc_id,
        chunk_text="这是第一个chunk的内容",
        chunk_embedding=[0.4, 0.5, 0.6, ...]
    )
    
    # 4. 执行两级搜索
    results = search_engine.full_search(
        keywords="AI machine learning",
        query_vector=[0.1, 0.2, 0.3, ...],
        doc_top_k=10,
        chunk_top_k_per_doc=5
    )

作者: XYZ-Algorithm-Team
"""

from .rag_document_manager import RAGDocumentManager
from .rag_search_engine import RAGSearchEngine
from .chunk_data_manager import ChunkDataManager
from .rag_schema_config import RAGSchemaConfig
from .database_clients import MySQLClient, MilvusClient, ObjectStoreClient

__all__ = [
    # 核心管理器
    'RAGDocumentManager',
    'RAGSearchEngine', 
    'ChunkDataManager',
    'RAGSchemaConfig',
    
    # 数据库客户端
    'MySQLClient',
    'MilvusClient',
    'ObjectStoreClient'
]

__version__ = '2.0.0'
__description__ = 'RAG系统两级数据库架构'

# 便捷的导入别名
DocumentManager = RAGDocumentManager
SearchEngine = RAGSearchEngine
SchemaConfig = RAGSchemaConfig