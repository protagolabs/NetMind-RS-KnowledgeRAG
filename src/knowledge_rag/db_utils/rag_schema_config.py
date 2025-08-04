"""
RAG Schema 配置管理器

这个模块提供了灵活的schema配置管理，方便用户自定义：
1. 文档级数据库结构（documents表和documents_vectors集合）
2. Chunk级数据库结构（每个文档的chunk表和chunk向量集合）
3. 搜索字段配置

用户可以在这里轻松修改数据库结构，无需修改核心代码。

作者: XYZ-Algorithm-Team
"""

from typing import Dict, List, Any, Optional
import json

class RAGSchemaConfig:
    """
    RAG系统的Schema配置管理器
    
    提供灵活的配置接口，用户可以自定义：
    - 文档级MySQL表结构
    - 文档级Milvus集合结构  
    - Chunk级MySQL表结构
    - Chunk级Milvus集合结构
    - 搜索字段配置
    
    使用示例：
        config = RAGSchemaConfig()
        
        # 获取默认配置
        doc_mysql_schema = config.get_document_mysql_schema()
        
        # 获取自定义配置
        chunk_schema = config.get_custom_chunk_schema("research_papers")
    """
    
    def __init__(self):
        """初始化schema配置管理器"""
        pass
    
    # ======================================
    # 工具方法
    # ======================================
    
    def get_chunk_table_name(self, source_id: str) -> str:
        """
        根据source_id生成chunk表名
        
        Args:
            source_id: 文档的source_id
            
        Returns:
            str: 格式化的表名
        """
        # 清理source_id，确保符合MySQL表名规范
        clean_source_id = source_id.replace('-', '_').replace('.', '_')
        return f"chunks_{clean_source_id}"
    
    def get_chunk_collection_name(self, source_id: str) -> str:
        """
        根据source_id生成chunk的Milvus集合名
        
        Args:
            source_id: 文档的source_id
            
        Returns:
            str: 格式化的集合名
        """
        clean_source_id = source_id.replace('-', '_').replace('.', '_')
        return f"chunks_vectors_{clean_source_id}"
    
    def get_chunk_insights_collection_name(self, source_id: str) -> str:
        """
        根据source_id生成chunk insights的Milvus集合名
        
        Args:
            source_id: 文档的source_id
            
        Returns:
            str: 格式化的集合名
        """
        clean_source_id = source_id.replace('-', '_').replace('.', '_')
        return f"chunks_insights_vectors_{clean_source_id}"
    
    # ======================================
    # 文档级数据库Schema配置
    # ======================================
    
    def get_document_mysql_schema(self, schema_type: str = "default") -> Dict:
        """
        获取文档级MySQL表结构配置
        
        这个表存储文档的基本信息、总结、关键词等，用于第一级搜索
        
        Args:
            schema_type: schema类型 ("default", "academic", "business", "custom")
        
        Returns:
            Dict: 文档级MySQL表结构配置
        """
        
        if schema_type == "academic":
            return self._get_academic_document_schema()
        elif schema_type == "business":
            return self._get_business_document_schema()
        elif schema_type == "custom":
            return self._get_custom_document_schema()
        else:
            return self._get_default_document_schema()
    
    def _get_default_document_schema(self) -> Dict:
        """默认的文档级MySQL表结构"""
        return {
            "table_name": "rag_documents",
            "columns": [
                # 基本字段
                {"name": "id", "type": "BIGINT", "auto_increment": True, "primary_key": True, "comment": "文档唯一标识"},
                {"name": "file_id", "type": "VARCHAR(255)", "not_null": True, "unique": True, "comment": "原始数据中的file_id"},
                {"name": "file_name", "type": "VARCHAR(255)", "not_null": True, "comment": "文档文件名"},
                {"name": "summary", "type": "TEXT", "comment": "文档总结摘要"},
                {"name": "insights", "type": "JSON", "comment": "文档洞察列表"},
                {"name": "key_words", "type": "JSON", "comment": "关键词列表"},
                {"name": "doc_markdown_content", "type": "LONGTEXT", "comment": "文档markdown内容"},
                
                # chunk级数据库信息
                {"name": "chunk_mysql_table", "type": "VARCHAR(255)", "comment": "chunk级MySQL表名"},
                {"name": "chunk_milvus_collection", "type": "VARCHAR(255)", "comment": "chunk级Milvus集合名"},
                
                # 统计和状态信息
                {"name": "chunk_count", "type": "INT", "default": "0", "comment": "chunk数量"},
                {"name": "processing_status", "type": "ENUM('pending', 'processing', 'completed', 'failed')", "default": "'pending'", "comment": "处理状态"},
                
                # 时间戳
                {"name": "created_at", "type": "DATETIME", "default": "CURRENT_TIMESTAMP", "comment": "创建时间"},
                {"name": "updated_at", "type": "DATETIME", "default": "CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP", "comment": "更新时间"}
            ],
            "indexes": [
                {"name": "idx_file_id", "columns": ["file_id"], "type": "UNIQUE"},
                {"name": "idx_file_name", "columns": ["file_name"], "type": "INDEX"},
                {"name": "idx_status", "columns": ["processing_status"], "type": "INDEX"},
                {"name": "idx_created_at", "columns": ["created_at"], "type": "INDEX"},
                {"name": "ft_summary", "columns": ["summary"], "type": "FULLTEXT"},
                {"name": "ft_doc_content", "columns": ["doc_markdown_content"], "type": "FULLTEXT"}
            ]
        }
    
    def _get_academic_document_schema(self) -> Dict:
        """学术论文的文档级表结构"""
        base_schema = self._get_default_document_schema()
        
        # 添加学术相关字段
        academic_columns = [
            {"name": "authors", "type": "TEXT", "comment": "作者列表，用分号分隔"},
            {"name": "journal", "type": "VARCHAR(500)", "comment": "期刊或会议名称"},
            {"name": "publish_year", "type": "INT", "comment": "发表年份"},
            {"name": "doi", "type": "VARCHAR(200)", "comment": "DOI标识符"},
            {"name": "arxiv_id", "type": "VARCHAR(50)", "comment": "ArXiv ID"},
            {"name": "citation_count", "type": "INT", "default": "0", "comment": "引用次数"},
            {"name": "research_field", "type": "VARCHAR(200)", "comment": "研究领域"},
            {"name": "abstract", "type": "TEXT", "comment": "论文摘要"}
        ]
        
        # 在metadata字段前插入学术字段
        insert_pos = next(i for i, col in enumerate(base_schema["columns"]) if col["name"] == "metadata")
        for i, col in enumerate(academic_columns):
            base_schema["columns"].insert(insert_pos + i, col)
        
        # 添加学术相关索引
        academic_indexes = [
            {"name": "idx_authors", "columns": ["authors"], "type": "INDEX"},
            {"name": "idx_journal", "columns": ["journal"], "type": "INDEX"},
            {"name": "idx_year", "columns": ["publish_year"], "type": "INDEX"},
            {"name": "idx_field", "columns": ["research_field"], "type": "INDEX"},
            {"name": "unique_doi", "columns": ["doi"], "type": "UNIQUE"},
            {"name": "ft_abstract", "columns": ["abstract"], "type": "FULLTEXT"}
        ]
        
        base_schema["indexes"].extend(academic_indexes)
        return base_schema
    
    def _get_business_document_schema(self) -> Dict:
        """商业文档的文档级表结构"""
        base_schema = self._get_default_document_schema()
        
        # 添加商业相关字段
        business_columns = [
            {"name": "department", "type": "VARCHAR(200)", "comment": "部门"},
            {"name": "document_type", "type": "ENUM('report', 'proposal', 'contract', 'manual', 'other')", "default": "'other'", "comment": "文档类型"},
            {"name": "confidentiality", "type": "ENUM('public', 'internal', 'confidential', 'secret')", "default": "'internal'", "comment": "保密级别"},
            {"name": "owner", "type": "VARCHAR(200)", "comment": "文档负责人"},
            {"name": "reviewers", "type": "TEXT", "comment": "审阅人员列表"},
            {"name": "version", "type": "VARCHAR(50)", "default": "'1.0'", "comment": "文档版本"},
            {"name": "expiry_date", "type": "DATE", "comment": "文档有效期"}
        ]
        
        insert_pos = next(i for i, col in enumerate(base_schema["columns"]) if col["name"] == "metadata")
        for i, col in enumerate(business_columns):
            base_schema["columns"].insert(insert_pos + i, col)
        
        # 添加商业相关索引
        business_indexes = [
            {"name": "idx_department", "columns": ["department"], "type": "INDEX"},
            {"name": "idx_doc_type", "columns": ["document_type"], "type": "INDEX"},
            {"name": "idx_confidentiality", "columns": ["confidentiality"], "type": "INDEX"},
            {"name": "idx_owner", "columns": ["owner"], "type": "INDEX"},
            {"name": "idx_version", "columns": ["version"], "type": "INDEX"}
        ]
        
        base_schema["indexes"].extend(business_indexes)
        return base_schema
    
    def _get_custom_document_schema(self) -> Dict:
        """
        自定义文档级表结构
        
        用户可以在这里定义完全自定义的文档级表结构
        """
        # 这里是示例，用户可以根据需要修改
        return {
            "table_name": "documents",
            "columns": [
                # 必需的基本字段（不要删除这些）
                {"name": "id", "type": "BIGINT", "auto_increment": True, "primary_key": True, "comment": "文档唯一标识"},
                {"name": "title", "type": "VARCHAR(1000)", "not_null": True, "comment": "文档标题"},
                {"name": "summary", "type": "TEXT", "comment": "文档总结摘要"},
                {"name": "keywords", "type": "TEXT", "comment": "关键词，用逗号分隔"},
                
                # 系统必需字段（不要删除这些）
                {"name": "chunk_mysql_table", "type": "VARCHAR(255)", "comment": "chunk级MySQL表名"},
                {"name": "chunk_milvus_collection", "type": "VARCHAR(255)", "comment": "chunk级Milvus集合名"},
                {"name": "chunk_schema_config", "type": "JSON", "comment": "chunk级表结构配置"},
                {"name": "chunk_count", "type": "INT", "default": "0", "comment": "chunk数量"},
                {"name": "processing_status", "type": "ENUM('pending', 'processing', 'completed', 'failed')", "default": "'pending'", "comment": "处理状态"},
                
                # 🔧 在这里添加你的自定义字段
                {"name": "custom_field_1", "type": "VARCHAR(500)", "comment": "自定义字段1"},
                {"name": "custom_field_2", "type": "TEXT", "comment": "自定义字段2"},
                {"name": "custom_json_field", "type": "JSON", "comment": "自定义JSON字段"},
                
                # 系统时间戳字段
                {"name": "metadata", "type": "JSON", "comment": "文档元数据信息"},
                {"name": "file_path", "type": "VARCHAR(1000)", "comment": "原始文件路径"},
                {"name": "file_size", "type": "BIGINT", "comment": "文件大小（字节）"},
                {"name": "created_at", "type": "DATETIME", "default": "CURRENT_TIMESTAMP", "comment": "创建时间"},
                {"name": "updated_at", "type": "DATETIME", "default": "CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP", "comment": "更新时间"}
            ],
            "indexes": [
                # 基本索引（建议保留）
                {"name": "idx_title", "columns": ["title"], "type": "INDEX"},
                {"name": "idx_status", "columns": ["processing_status"], "type": "INDEX"},
                {"name": "ft_keywords", "columns": ["keywords"], "type": "FULLTEXT"},
                {"name": "ft_summary", "columns": ["summary"], "type": "FULLTEXT"},
                
                # 🔧 在这里添加你的自定义索引
                {"name": "idx_custom_field_1", "columns": ["custom_field_1"], "type": "INDEX"},
                {"name": "ft_custom_field_2", "columns": ["custom_field_2"], "type": "FULLTEXT"}
            ]
        }
    
    def get_document_milvus_schema(self, vector_dim: int = 768, schema_type: str = "default") -> Dict:
        """
        获取文档级Milvus集合结构配置
        
        Args:
            vector_dim: 向量维度
            schema_type: schema类型
        
        Returns:
            Dict: 文档级Milvus集合结构配置
        """
        
        if schema_type == "academic":
            return self._get_academic_document_milvus_schema(vector_dim)
        elif schema_type == "custom":
            return self._get_custom_document_milvus_schema(vector_dim)
        else:
            return self._get_default_document_milvus_schema(vector_dim)
    
    def _get_default_document_milvus_schema(self, vector_dim: int) -> Dict:
        """默认的文档级Milvus集合结构 - 支持多个向量字段"""
        return {
            "collection_name_suffix": "documents_vectors",
            "fields": [
                {"name": "id", "type": "INT64", "is_primary": True, "auto_id": False, "comment": "文档ID"},
                {"name": "file_id", "type": "VARCHAR", "max_length": 255, "comment": "原始文件ID"},
                {"name": "summary_embedding", "type": "FLOAT_VECTOR", "dim": vector_dim, "comment": "摘要embedding向量"},
                {"name": "key_words_embedding", "type": "FLOAT_VECTOR", "dim": vector_dim, "comment": "关键词embedding向量"},
                {"name": "metadata", "type": "JSON", "comment": "向量元数据"}
            ],
            "indexes": [
                {"field": "summary_embedding", "type": "IVF_FLAT", "metric": "L2", "params": {"nlist": 1024}},
                {"field": "key_words_embedding", "type": "IVF_FLAT", "metric": "COSINE", "params": {"nlist": 512}}
            ]
        }
    
    def _get_academic_document_milvus_schema(self, vector_dim: int) -> Dict:
        """学术论文的文档级Milvus集合结构"""
        base_schema = self._get_default_document_milvus_schema(vector_dim)
        
        # 添加学术相关向量字段
        academic_fields = [
            {"name": "title_embedding", "type": "FLOAT_VECTOR", "dim": vector_dim, "comment": "标题embedding向量"},
            {"name": "abstract_embedding", "type": "FLOAT_VECTOR", "dim": vector_dim, "comment": "摘要embedding向量"},
            {"name": "authors_embedding", "type": "FLOAT_VECTOR", "dim": vector_dim, "comment": "作者embedding向量"}
        ]
        
        # 在metadata字段前插入
        insert_pos = next(i for i, field in enumerate(base_schema["fields"]) if field["name"] == "metadata")
        for i, field in enumerate(academic_fields):
            base_schema["fields"].insert(insert_pos + i, field)
        
        # 添加对应索引
        academic_indexes = [
            {"field": "title_embedding", "type": "IVF_FLAT", "metric": "L2", "params": {"nlist": 1024}},
            {"field": "abstract_embedding", "type": "IVF_FLAT", "metric": "L2", "params": {"nlist": 1024}},
            {"field": "authors_embedding", "type": "IVF_FLAT", "metric": "COSINE", "params": {"nlist": 512}}
        ]
        
        base_schema["indexes"].extend(academic_indexes)
        return base_schema
    
    def get_document_insights_milvus_schema(self, vector_dim: int = 1536) -> Dict:
        """
        获取文档级insights向量集合结构配置
        
        由于每个文档有多个insights向量，需要单独的集合存储
        """
        return {
            "collection_name_suffix": "documents_insights_vectors",
            "fields": [
                {"name": "id", "type": "INT64", "is_primary": True, "auto_id": True, "comment": "自增ID"},
                {"name": "doc_id", "type": "INT64", "comment": "文档ID，关联到documents表"},
                {"name": "file_id", "type": "VARCHAR", "max_length": 255, "comment": "原始文件ID"},
                {"name": "insight_index", "type": "INT64", "comment": "insight在列表中的索引"},
                {"name": "insight_text", "type": "VARCHAR", "max_length": 65535, "comment": "insight原文"},
                {"name": "insight_embedding", "type": "FLOAT_VECTOR", "dim": vector_dim, "comment": "insight向量"},
                {"name": "metadata", "type": "JSON", "comment": "额外元数据"}
            ],
            "indexes": [
                {"field": "insight_embedding", "type": "IVF_FLAT", "metric": "L2", "params": {"nlist": 1024}}
            ]
        }
    
    def _get_custom_document_milvus_schema(self, vector_dim: int) -> Dict:
        """
        自定义文档级Milvus集合结构
        
        用户可以在这里定义自己需要的向量字段
        """
        return {
            "collection_name_suffix": "documents_vectors",
            "fields": [
                # 必需字段（不要删除）
                {"name": "id", "type": "INT64", "is_primary": True, "auto_id": False, "comment": "文档ID"},
                
                # 🔧 在这里定义你的向量字段
                {"name": "main_content_embedding", "type": "FLOAT_VECTOR", "dim": vector_dim, "comment": "主要内容向量"},
                {"name": "summary_embedding", "type": "FLOAT_VECTOR", "dim": vector_dim, "comment": "总结向量"},
                {"name": "keywords_embedding", "type": "FLOAT_VECTOR", "dim": vector_dim, "comment": "关键词向量"},
                {"name": "custom_embedding", "type": "FLOAT_VECTOR", "dim": vector_dim, "comment": "自定义向量字段"},
                
                # 元数据字段
                {"name": "metadata", "type": "JSON", "comment": "向量元数据"}
            ],
            "indexes": [
                # 🔧 在这里定义对应的向量索引
                {"field": "main_content_embedding", "type": "IVF_FLAT", "metric": "L2", "params": {"nlist": 1024}},
                {"field": "summary_embedding", "type": "IVF_FLAT", "metric": "L2", "params": {"nlist": 1024}},
                {"field": "keywords_embedding", "type": "IVF_FLAT", "metric": "COSINE", "params": {"nlist": 512}},
                {"field": "custom_embedding", "type": "IVF_SQ8", "metric": "COSINE", "params": {"nlist": 2048}}
            ]
        }
    
    # ======================================
    # Chunk级数据库Schema配置
    # ======================================
    
    def get_chunk_schema(self, schema_type: str = "default") -> Dict:
        """
        获取chunk级数据库结构配置
        
        Args:
            schema_type: schema类型 ("default", "academic", "business", "custom")
        
        Returns:
            Dict: chunk级数据库结构配置，包含mysql和milvus两部分
        """
        
        if schema_type == "academic":
            return self._get_academic_chunk_schema()
        elif schema_type == "business":
            return self._get_business_chunk_schema()
        elif schema_type == "custom":
            return self._get_custom_chunk_schema()
        else:
            return self._get_default_chunk_schema()
    
    def _get_default_chunk_schema(self) -> Dict:
        """默认的chunk级数据库结构 - 适配新的数据格式"""
        return {
            "mysql": {
                "columns": [
                    {"name": "id", "type": "BIGINT", "auto_increment": True, "primary_key": True, "comment": "chunk ID"},
                    {"name": "source_id", "type": "VARCHAR(255)", "not_null": True, "comment": "来源文档的source_id"},
                    {"name": "chunk_id", "type": "VARCHAR(255)", "not_null": True, "unique": True, "comment": "原始数据中的chunk_id"},
                    {"name": "summary", "type": "TEXT", "comment": "chunk总结"},
                    {"name": "insights", "type": "JSON", "comment": "chunk洞察列表"},
                    {"name": "key_words", "type": "JSON", "comment": "关键词列表"},
                    {"name": "chunk_markdown_content", "type": "LONGTEXT", "not_null": True, "comment": "chunk的markdown内容"},
                    {"name": "metadata", "type": "JSON", "comment": "chunk元数据"},
                    {"name": "created_at", "type": "DATETIME", "default": "CURRENT_TIMESTAMP", "comment": "创建时间"}
                ],
                "indexes": [
                    {"name": "idx_source_id", "columns": ["source_id"], "type": "INDEX"},
                    {"name": "idx_chunk_id", "columns": ["chunk_id"], "type": "UNIQUE"},
                    {"name": "ft_summary", "columns": ["summary"], "type": "FULLTEXT"},
                    {"name": "ft_chunk_content", "columns": ["chunk_markdown_content"], "type": "FULLTEXT"}
                ]
            },
            "milvus": {
                "fields": [
                    {"name": "id", "type": "INT64", "is_primary": True, "auto_id": False, "comment": "chunk ID"},
                    {"name": "source_id", "type": "VARCHAR", "max_length": 255, "comment": "来源文档的source_id"},
                    {"name": "chunk_id", "type": "VARCHAR", "max_length": 255, "comment": "原始chunk_id"},
                    {"name": "summary_embedding", "type": "FLOAT_VECTOR", "dim": 1536, "comment": "总结向量"},
                    {"name": "key_words_embedding", "type": "FLOAT_VECTOR", "dim": 1536, "comment": "关键词向量"},
                    {"name": "metadata", "type": "JSON", "comment": "chunk元数据"}
                ],
                "indexes": [
                    {"field": "summary_embedding", "type": "IVF_FLAT", "metric": "L2", "params": {"nlist": 1024}},
                    {"field": "key_words_embedding", "type": "IVF_FLAT", "metric": "COSINE", "params": {"nlist": 512}}
                ]
            }
        }
    
    def get_chunk_insights_milvus_schema(self, vector_dim: int = 1536) -> Dict:
        """
        获取chunk级insights向量集合结构配置
        
        由于每个chunk有多个insights向量，需要单独的集合存储
        """
        return {
            "collection_name_suffix": "chunks_insights_vectors",
            "fields": [
                {"name": "id", "type": "INT64", "is_primary": True, "auto_id": True, "comment": "自增ID"},
                {"name": "chunk_id", "type": "VARCHAR", "max_length": 255, "comment": "原始chunk_id"},
                {"name": "source_id", "type": "VARCHAR", "max_length": 255, "comment": "来源文档的source_id"},
                {"name": "insight_index", "type": "INT64", "comment": "insight在列表中的索引"},
                {"name": "insight_text", "type": "VARCHAR", "max_length": 65535, "comment": "insight原文"},
                {"name": "insight_embedding", "type": "FLOAT_VECTOR", "dim": vector_dim, "comment": "insight向量"},
                {"name": "metadata", "type": "JSON", "comment": "额外元数据"}
            ],
            "indexes": [
                {"field": "insight_embedding", "type": "IVF_FLAT", "metric": "L2", "params": {"nlist": 1024}}
            ]
        }
    
    def _get_academic_chunk_schema(self) -> Dict:
        """学术论文的chunk级数据库结构"""
        base_schema = self._get_default_chunk_schema()
        
        # 添加学术相关字段
        academic_mysql_columns = [
            {"name": "section_type", "type": "ENUM('abstract', 'introduction', 'method', 'result', 'conclusion', 'reference', 'other')", "default": "'other'", "comment": "章节类型"},
            {"name": "section_title", "type": "VARCHAR(500)", "comment": "章节标题"},
            {"name": "figures", "type": "JSON", "comment": "图表信息"},
            {"name": "formulas", "type": "JSON", "comment": "公式信息"},
            {"name": "citations", "type": "TEXT", "comment": "引用文献"},
            {"name": "importance_score", "type": "DECIMAL(3,2)", "comment": "重要性评分"}
        ]
        
        # 插入到metadata字段前
        insert_pos = next(i for i, col in enumerate(base_schema["mysql"]["columns"]) if col["name"] == "metadata")
        for i, col in enumerate(academic_mysql_columns):
            base_schema["mysql"]["columns"].insert(insert_pos + i, col)
        
        # 添加学术相关索引
        academic_mysql_indexes = [
            {"name": "idx_section_type", "columns": ["section_type"], "type": "INDEX"},
            {"name": "idx_section_title", "columns": ["section_title"], "type": "INDEX"},
            {"name": "idx_importance", "columns": ["importance_score"], "type": "INDEX"}
        ]
        base_schema["mysql"]["indexes"].extend(academic_mysql_indexes)
        
        # 添加学术相关向量字段
        academic_milvus_fields = [
            {"name": "section_embedding", "type": "FLOAT_VECTOR", "dim": 768, "comment": "章节向量"},
            {"name": "formula_embedding", "type": "FLOAT_VECTOR", "dim": 768, "comment": "公式向量"}
        ]
        
        insert_pos = next(i for i, field in enumerate(base_schema["milvus"]["fields"]) if field["name"] == "metadata")
        for i, field in enumerate(academic_milvus_fields):
            base_schema["milvus"]["fields"].insert(insert_pos + i, field)
        
        # 添加对应向量索引
        academic_milvus_indexes = [
            {"field": "section_embedding", "type": "IVF_FLAT", "metric": "L2", "params": {"nlist": 1024}},
            {"field": "formula_embedding", "type": "IVF_FLAT", "metric": "L2", "params": {"nlist": 512}}
        ]
        base_schema["milvus"]["indexes"].extend(academic_milvus_indexes)
        
        return base_schema
    
    def _get_business_chunk_schema(self) -> Dict:
        """商业文档的chunk级数据库结构"""
        base_schema = self._get_default_chunk_schema()
        
        # 添加商业相关字段
        business_mysql_columns = [
            {"name": "content_type", "type": "ENUM('text', 'table', 'chart', 'list', 'other')", "default": "'text'", "comment": "内容类型"},
            {"name": "business_unit", "type": "VARCHAR(200)", "comment": "业务单元"},
            {"name": "priority", "type": "ENUM('low', 'medium', 'high', 'critical')", "default": "'medium'", "comment": "优先级"},
            {"name": "action_required", "type": "BOOLEAN", "default": "FALSE", "comment": "是否需要行动"},
            {"name": "responsible_person", "type": "VARCHAR(200)", "comment": "负责人"},
            {"name": "due_date", "type": "DATE", "comment": "截止日期"}
        ]
        
        insert_pos = next(i for i, col in enumerate(base_schema["mysql"]["columns"]) if col["name"] == "metadata")
        for i, col in enumerate(business_mysql_columns):
            base_schema["mysql"]["columns"].insert(insert_pos + i, col)
        
        # 添加商业相关索引
        business_mysql_indexes = [
            {"name": "idx_content_type", "columns": ["content_type"], "type": "INDEX"},
            {"name": "idx_business_unit", "columns": ["business_unit"], "type": "INDEX"},
            {"name": "idx_priority", "columns": ["priority"], "type": "INDEX"},
            {"name": "idx_action_required", "columns": ["action_required"], "type": "INDEX"},
            {"name": "idx_due_date", "columns": ["due_date"], "type": "INDEX"}
        ]
        base_schema["mysql"]["indexes"].extend(business_mysql_indexes)
        
        return base_schema
    
    def _get_custom_chunk_schema(self) -> Dict:
        """
        自定义chunk级数据库结构
        
        用户可以在这里定义完全自定义的chunk级数据库结构
        """
        return {
            "mysql": {
                "columns": [
                    # 必需的基本字段（不要删除这些）
                    {"name": "id", "type": "BIGINT", "auto_increment": True, "primary_key": True, "comment": "chunk ID"},
                    {"name": "chunk_index", "type": "INT", "not_null": True, "comment": "chunk在文档中的序号"},
                    {"name": "chunk_text", "type": "LONGTEXT", "not_null": True, "comment": "chunk文本内容"},
                    
                    # 🔧 在这里添加你的自定义字段
                    {"name": "custom_title", "type": "VARCHAR(500)", "comment": "自定义标题字段"},
                    {"name": "custom_category", "type": "VARCHAR(200)", "comment": "自定义分类字段"},
                    {"name": "custom_tags", "type": "TEXT", "comment": "自定义标签字段"},
                    {"name": "custom_score", "type": "DECIMAL(5,2)", "comment": "自定义评分字段"},
                    {"name": "custom_json_data", "type": "JSON", "comment": "自定义JSON数据"},
                    
                    # 系统字段（建议保留）
                    {"name": "keywords", "type": "TEXT", "comment": "chunk关键词"},
                    {"name": "metadata", "type": "JSON", "comment": "chunk元数据"},
                    {"name": "token_count", "type": "INT", "comment": "token数量"},
                    {"name": "created_at", "type": "DATETIME", "default": "CURRENT_TIMESTAMP", "comment": "创建时间"}
                ],
                "indexes": [
                    # 系统索引（建议保留）
                    {"name": "idx_chunk_index", "columns": ["chunk_index"], "type": "INDEX"},
                    {"name": "fulltext_content", "columns": ["chunk_text"], "type": "FULLTEXT"},
                    {"name": "fulltext_keywords", "columns": ["keywords"], "type": "FULLTEXT"},
                    
                    # 🔧 在这里添加你的自定义索引
                    {"name": "idx_custom_category", "columns": ["custom_category"], "type": "INDEX"},
                    {"name": "idx_custom_score", "columns": ["custom_score"], "type": "INDEX"},
                    {"name": "fulltext_custom_tags", "columns": ["custom_tags"], "type": "FULLTEXT"}
                ]
            },
            "milvus": {
                "fields": [
                    # 必需字段（不要删除）
                    {"name": "id", "type": "INT64", "is_primary": True, "auto_id": False, "comment": "chunk ID"},
                    
                    # 🔧 在这里定义你的向量字段
                    {"name": "main_embedding", "type": "FLOAT_VECTOR", "dim": 768, "comment": "主要文本向量"},
                    {"name": "title_embedding", "type": "FLOAT_VECTOR", "dim": 768, "comment": "标题向量"},
                    {"name": "custom_embedding", "type": "FLOAT_VECTOR", "dim": 384, "comment": "自定义向量字段"},
                    
                    # 元数据字段
                    {"name": "metadata", "type": "JSON", "comment": "chunk元数据"}
                ],
                "indexes": [
                    # 🔧 在这里定义对应的向量索引
                    {"field": "main_embedding", "type": "IVF_FLAT", "metric": "L2", "params": {"nlist": 1024}},
                    {"field": "title_embedding", "type": "IVF_FLAT", "metric": "L2", "params": {"nlist": 512}},
                    {"field": "custom_embedding", "type": "IVF_SQ8", "metric": "COSINE", "params": {"nlist": 2048}}
                ]
            }
        }
    
    # ======================================
    # 搜索配置
    # ======================================
    
    def get_search_config(self, config_type: str = "default") -> Dict:
        """
        获取搜索字段配置
        
        Args:
            config_type: 配置类型 ("default", "academic", "business", "custom")
        
        Returns:
            Dict: 搜索配置
        """
        
        if config_type == "academic":
            return self._get_academic_search_config()
        elif config_type == "business":
            return self._get_business_search_config()
        elif config_type == "custom":
            return self._get_custom_search_config()
        else:
            return self._get_default_search_config()
    
    def _get_default_search_config(self) -> Dict:
        """默认搜索配置"""
        return {
            # 文档级搜索字段配置
            "document_level": {
                "mysql_search_fields": ["title", "summary", "keywords"],
                "milvus_vector_fields": ["document_embedding", "summary_embedding", "keywords_embedding"],
                "default_vector_field": "document_embedding",
                "similarity_threshold": 0.5
            },
            
            # Chunk级搜索字段配置
            "chunk_level": {
                "mysql_search_fields": ["chunk_text", "chunk_title", "keywords"],
                "milvus_vector_fields": ["chunk_embedding", "title_embedding"],
                "default_vector_field": "chunk_embedding",
                "similarity_threshold": 0.6
            },
            
            # 搜索参数配置
            "search_params": {
                "doc_top_k": 10,
                "chunk_top_k_per_doc": 5,
                "max_total_chunks": 50,
                "keyword_weight": 0.3,
                "vector_weight": 0.7
            }
        }
    
    def _get_academic_search_config(self) -> Dict:
        """学术论文搜索配置"""
        base_config = self._get_default_search_config()
        
        # 扩展文档级搜索字段
        base_config["document_level"]["mysql_search_fields"].extend(["authors", "abstract", "journal"])
        base_config["document_level"]["milvus_vector_fields"].extend(["title_embedding", "abstract_embedding", "authors_embedding"])
        
        # 扩展chunk级搜索字段
        base_config["chunk_level"]["mysql_search_fields"].extend(["section_title", "citations"])
        base_config["chunk_level"]["milvus_vector_fields"].extend(["section_embedding", "formula_embedding"])
        
        # 调整搜索参数
        base_config["search_params"]["similarity_threshold"] = 0.7
        base_config["search_params"]["chunk_top_k_per_doc"] = 8
        
        return base_config
    
    def _get_business_search_config(self) -> Dict:
        """商业文档搜索配置"""
        base_config = self._get_default_search_config()
        
        # 扩展文档级搜索字段
        base_config["document_level"]["mysql_search_fields"].extend(["department", "owner"])
        
        # 扩展chunk级搜索字段
        base_config["chunk_level"]["mysql_search_fields"].extend(["business_unit", "responsible_person"])
        
        # 调整搜索参数
        base_config["search_params"]["keyword_weight"] = 0.4
        base_config["search_params"]["vector_weight"] = 0.6
        
        return base_config
    
    def _get_custom_search_config(self) -> Dict:
        """
        自定义搜索配置
        
        用户可以在这里定义自己的搜索字段和参数
        """
        return {
            # 文档级搜索字段配置
            "document_level": {
                # 🔧 定义文档级MySQL搜索字段
                "mysql_search_fields": ["title", "summary", "keywords", "custom_field_1", "custom_field_2"],
                
                # 🔧 定义文档级Milvus向量搜索字段
                "milvus_vector_fields": ["main_content_embedding", "summary_embedding", "custom_embedding"],
                
                # 🔧 定义默认向量字段
                "default_vector_field": "main_content_embedding",
                
                # 🔧 定义相似度阈值
                "similarity_threshold": 0.6
            },
            
            # Chunk级搜索字段配置
            "chunk_level": {
                # 🔧 定义chunk级MySQL搜索字段
                "mysql_search_fields": ["chunk_text", "custom_title", "custom_tags", "keywords"],
                
                # 🔧 定义chunk级Milvus向量搜索字段
                "milvus_vector_fields": ["main_embedding", "title_embedding", "custom_embedding"],
                
                # 🔧 定义默认向量字段
                "default_vector_field": "main_embedding",
                
                # 🔧 定义相似度阈值
                "similarity_threshold": 0.7
            },
            
            # 搜索参数配置
            "search_params": {
                # 🔧 调整搜索参数
                "doc_top_k": 15,
                "chunk_top_k_per_doc": 8,
                "max_total_chunks": 100,
                "keyword_weight": 0.2,
                "vector_weight": 0.8
            }
        }
    
    # ======================================
    # 工具方法
    # ======================================
    
    def validate_schema(self, schema: Dict, schema_type: str) -> bool:
        """
        验证schema配置的有效性
        
        Args:
            schema: schema配置
            schema_type: schema类型 ("document_mysql", "document_milvus", "chunk")
        
        Returns:
            bool: 是否有效
        """
        
        try:
            if schema_type == "document_mysql":
                return self._validate_mysql_schema(schema)
            elif schema_type == "document_milvus":
                return self._validate_milvus_schema(schema)
            elif schema_type == "chunk":
                return (self._validate_mysql_schema(schema.get("mysql", {})) and
                       self._validate_milvus_schema(schema.get("milvus", {})))
            else:
                return False
                
        except Exception as e:
            print(f"Schema验证失败: {e}")
            return False
    
    def _validate_mysql_schema(self, schema: Dict) -> bool:
        """验证MySQL schema"""
        if not schema.get("columns"):
            return False
        
        # 检查是否有主键
        has_primary_key = any(col.get("primary_key", False) for col in schema["columns"])
        if not has_primary_key:
            return False
        
        # 检查必需字段
        required_fields = ["name", "type"]
        for column in schema["columns"]:
            if not all(field in column for field in required_fields):
                return False
        
        return True
    
    def _validate_milvus_schema(self, schema: Dict) -> bool:
        """验证Milvus schema"""
        if not schema.get("fields"):
            return False
        
        # 检查是否有主键
        has_primary_key = any(field.get("is_primary", False) for field in schema["fields"])
        if not has_primary_key:
            return False
        
        # 检查向量字段的维度
        for field in schema["fields"]:
            if field.get("type") == "FLOAT_VECTOR" and not field.get("dim"):
                return False
        
        return True