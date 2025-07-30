"""
RAGDocumentManager - RAG系统的两级数据库管理器

这个模块负责管理RAG系统的两级数据库架构：
1. 文档级别：存储文档的总结、关键词、文档级embedding
2. Chunk级别：每个文档都有独立的MySQL表和Milvus集合存储chunks

主要功能：
- 创建文档时自动创建对应的chunk级数据库
- 管理文档级和chunk级的数据结构
- 提供灵活的schema配置
- 支持文档的增删改查

作者: XYZ-Algorithm-Team
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from db_server.experiment_data import UnifiedDataManager, SERVICES_CONFIG
from .database_clients import MySQLClient, MilvusClient, ObjectStoreClient

logger = logging.getLogger(__name__)

class RAGDocumentManager:
    """
    RAG文档管理器 - 管理两级数据库架构
    
    架构设计：
    1. 文档级别：
       - documents表：存储文档ID、标题、总结、关键词等
       - documents_vectors集合：存储文档级embedding向量
    
    2. Chunk级别：
       - 每个文档创建独立的MySQL表：doc_{doc_id}_chunks
       - 每个文档创建独立的Milvus集合：doc_{doc_id}_chunks_vectors
    
    使用示例：
        # 创建管理器
        rag_manager = RAGDocumentManager()
        
        # 添加文档（自动创建chunk级数据库）
        doc_id = rag_manager.add_document(
            title="AI研究论文",
            summary="这篇论文介绍了...",
            keywords=["AI", "机器学习"],
            document_embedding=[0.1, 0.2, ...],
            chunk_schema_config={...}  # chunk级表结构配置
        )
        
        # 向文档添加chunks
        rag_manager.add_chunks_to_document(doc_id, chunks_data)
    """
    
    def __init__(self, experiment_name: str = None):
        """
        初始化RAG文档管理器
        
        Args:
            experiment_name: 实验名称，如果不提供则使用当前活跃实验
        """
        self.experiment_name = experiment_name or self._get_current_experiment()
        
        # 初始化统一数据管理器
        self.data_manager = UnifiedDataManager(SERVICES_CONFIG)
        connection_status = self.data_manager.connect_all()
        
        if not all(connection_status.values()):
            logger.warning(f"部分服务连接失败: {connection_status}")
        
        # 初始化各个客户端
        self.mysql_client = MySQLClient(self.data_manager.mysql_conn)
        self.milvus_client = MilvusClient(self.data_manager.milvus_conn)
        self.object_store_client = ObjectStoreClient(self.data_manager.object_store_base_path)
        
        # 数据库名称
        self.db_name = f"knowledge_rag_{self.experiment_name}"
        
        # 确保文档级数据库存在
        self._ensure_document_level_tables()
        
        logger.info(f"RAGDocumentManager 初始化完成，实验: {self.experiment_name}")
    
    def _get_current_experiment(self) -> str:
        """获取当前活跃的实验名称"""
        config_file = Path("db_server/current_experiment.yaml")
        if config_file.exists():
            import yaml
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    return config.get('current_experiment', 'default_experiment')
            except Exception as e:
                logger.warning(f"读取当前实验配置失败: {e}")
        return 'default_experiment'
    
    def _ensure_document_level_tables(self):
        """确保文档级数据表存在"""
        
        # 1. 创建文档级MySQL表
        create_documents_table_sql = f"""
        CREATE TABLE IF NOT EXISTS `{self.db_name}`.`documents` (
            `id` BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '文档唯一标识',
            `title` VARCHAR(1000) NOT NULL COMMENT '文档标题',
            `summary` TEXT COMMENT '文档总结摘要',
            `keywords` TEXT COMMENT '关键词，用逗号分隔',
            `metadata` JSON COMMENT '文档元数据信息',
            
            -- chunk级数据库信息
            `chunk_mysql_table` VARCHAR(255) COMMENT 'chunk级MySQL表名',
            `chunk_milvus_collection` VARCHAR(255) COMMENT 'chunk级Milvus集合名',
            `chunk_schema_config` JSON COMMENT 'chunk级表结构配置',
            
            -- 文档状态和统计
            `chunk_count` INT DEFAULT 0 COMMENT 'chunk数量',
            `processing_status` ENUM('pending', 'processing', 'completed', 'failed') DEFAULT 'pending' COMMENT '处理状态',
            `file_path` VARCHAR(1000) COMMENT '原始文件路径',
            `file_size` BIGINT COMMENT '文件大小（字节）',
            
            -- 时间戳
            `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
            `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
            
            -- 索引
            INDEX `idx_title` (`title`(100)),
            INDEX `idx_status` (`processing_status`),
            INDEX `idx_created_at` (`created_at`),
            FULLTEXT INDEX `ft_keywords` (`keywords`),
            FULLTEXT INDEX `ft_summary` (`summary`)
            
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci 
        COMMENT='RAG系统文档级数据表';
        """
        
        try:
            self.mysql_client.execute(create_documents_table_sql)
            logger.info("文档级MySQL表检查/创建完成")
        except Exception as e:
            logger.error(f"创建文档级MySQL表失败: {e}")
            raise
        
        # 2. 创建文档级Milvus集合
        documents_collection_name = f"{self.db_name}_documents_vectors"
        self._ensure_document_level_milvus_collection(documents_collection_name)
    
    def _ensure_document_level_milvus_collection(self, collection_name: str):
        """确保文档级Milvus集合存在"""
        
        # 默认的文档级向量集合结构
        document_schema_config = {
            "fields": [
                {"name": "id", "type": "INT64", "is_primary": True, "auto_id": False, "comment": "文档ID"},
                {"name": "document_embedding", "type": "FLOAT_VECTOR", "dim": 768, "comment": "文档级embedding向量"},
                {"name": "summary_embedding", "type": "FLOAT_VECTOR", "dim": 768, "comment": "摘要embedding向量"},
                {"name": "keywords_embedding", "type": "FLOAT_VECTOR", "dim": 768, "comment": "关键词embedding向量"},
                {"name": "metadata", "type": "JSON", "comment": "向量元数据"}
            ],
            "indexes": [
                {"field": "document_embedding", "type": "IVF_FLAT", "metric": "L2", "params": {"nlist": 1024}},
                {"field": "summary_embedding", "type": "IVF_FLAT", "metric": "L2", "params": {"nlist": 1024}},
                {"field": "keywords_embedding", "type": "IVF_FLAT", "metric": "COSINE", "params": {"nlist": 512}}
            ]
        }
        
        try:
            success = self.milvus_client.create_collection(collection_name, document_schema_config)
            if success:
                logger.info(f"文档级Milvus集合检查/创建完成: {collection_name}")
            else:
                logger.warning(f"文档级Milvus集合创建失败: {collection_name}")
        except Exception as e:
            logger.error(f"创建文档级Milvus集合失败: {e}")
    
    def add_document(
        self,
        title: str,
        summary: str = None,
        keywords: List[str] = None,
        metadata: Dict = None,
        document_embedding: List[float] = None,
        summary_embedding: List[float] = None,
        keywords_embedding: List[float] = None,
        chunk_schema_config: Dict = None,
        file_path: str = None,
        file_size: int = None
    ) -> int:
        """
        添加新文档到RAG系统
        
        这个方法会：
        1. 在documents表中插入文档记录
        2. 在documents_vectors集合中插入文档级向量
        3. 为该文档创建独立的chunk级MySQL表和Milvus集合
        
        Args:
            title: 文档标题
            summary: 文档总结
            keywords: 关键词列表
            metadata: 文档元数据
            document_embedding: 文档级embedding向量
            summary_embedding: 摘要embedding向量  
            keywords_embedding: 关键词embedding向量
            chunk_schema_config: chunk级表结构配置
            file_path: 原始文件路径
            file_size: 文件大小
        
        Returns:
            int: 文档ID
        """
        
        try:
            # 1. 生成chunk级数据库名称
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 毫秒级时间戳
            chunk_mysql_table = f"doc_chunks_{timestamp}"
            chunk_milvus_collection = f"doc_chunks_vectors_{timestamp}"
            
            # 2. 插入文档记录到MySQL
            insert_sql = f"""
            INSERT INTO `{self.db_name}`.`documents` 
            (title, summary, keywords, metadata, chunk_mysql_table, chunk_milvus_collection, 
             chunk_schema_config, file_path, file_size)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            values = (
                title,
                summary,
                ','.join(keywords) if keywords else None,
                json.dumps(metadata or {}, ensure_ascii=False),
                chunk_mysql_table,
                chunk_milvus_collection,
                json.dumps(chunk_schema_config or {}, ensure_ascii=False),
                file_path,
                file_size
            )
            
            doc_id = self.mysql_client.execute(insert_sql, values, fetch_lastrowid=True)
            logger.info(f"文档记录插入成功，ID: {doc_id}")
            
            # 3. 插入文档级向量到Milvus
            if any([document_embedding, summary_embedding, keywords_embedding]):
                self._insert_document_vectors(
                    doc_id, document_embedding, summary_embedding, keywords_embedding, metadata
                )
            
            # 4. 创建该文档的chunk级数据库
            self._create_chunk_level_database(doc_id, chunk_mysql_table, chunk_milvus_collection, chunk_schema_config)
            
            logger.info(f"文档 '{title}' 添加完成，ID: {doc_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            # 清理可能已创建的数据
            try:
                if 'doc_id' in locals():
                    self._cleanup_failed_document_creation(doc_id, chunk_mysql_table, chunk_milvus_collection)
            except:
                pass
            raise
    
    def _insert_document_vectors(
        self, 
        doc_id: int, 
        document_embedding: List[float], 
        summary_embedding: List[float], 
        keywords_embedding: List[float],
        metadata: Dict
    ):
        """插入文档级向量到Milvus"""
        
        documents_collection_name = f"{self.db_name}_documents_vectors"
        
        vector_data = {
            "id": doc_id,
            "metadata": metadata or {}
        }
        
        # 只添加非空的向量字段
        if document_embedding:
            vector_data["document_embedding"] = document_embedding
        else:
            # 如果没有提供向量，使用零向量占位
            vector_data["document_embedding"] = [0.0] * 768
            
        if summary_embedding:
            vector_data["summary_embedding"] = summary_embedding
        else:
            vector_data["summary_embedding"] = [0.0] * 768
            
        if keywords_embedding:
            vector_data["keywords_embedding"] = keywords_embedding
        else:
            vector_data["keywords_embedding"] = [0.0] * 768
        
        try:
            milvus_id = self.milvus_client.insert(documents_collection_name, vector_data)
            logger.info(f"文档向量插入成功，Milvus ID: {milvus_id}")
        except Exception as e:
            logger.error(f"插入文档向量失败: {e}")
            raise
    
    def _create_chunk_level_database(
        self, 
        doc_id: int, 
        mysql_table_name: str, 
        milvus_collection_name: str, 
        schema_config: Dict
    ):
        """为文档创建chunk级数据库"""
        
        # 使用默认的chunk级schema如果没有提供配置
        if not schema_config:
            schema_config = self._get_default_chunk_schema()
        
        try:
            # 1. 创建chunk级MySQL表
            self._create_chunk_mysql_table(mysql_table_name, schema_config.get('mysql', {}))
            logger.info(f"Chunk级MySQL表创建成功: {mysql_table_name}")
            
            # 2. 创建chunk级Milvus集合
            self._create_chunk_milvus_collection(milvus_collection_name, schema_config.get('milvus', {}))
            logger.info(f"Chunk级Milvus集合创建成功: {milvus_collection_name}")
            
        except Exception as e:
            logger.error(f"创建chunk级数据库失败: {e}")
            raise
    
    def _create_chunk_mysql_table(self, table_name: str, mysql_schema: Dict):
        """创建chunk级MySQL表"""
        
        # 如果没有提供schema，使用默认的
        if not mysql_schema:
            mysql_schema = self._get_default_chunk_schema()['mysql']
        
        # 构建CREATE TABLE SQL
        columns = mysql_schema.get('columns', [])
        if not columns:
            raise ValueError("Chunk级MySQL schema必须包含字段定义")
        
        column_definitions = []
        for column in columns:
            col_sql = self._generate_mysql_column_sql(column)
            column_definitions.append(f"    {col_sql}")
        
        # 外键约束
        for fk in mysql_schema.get('foreign_keys', []):
            fk_sql = self._generate_mysql_foreign_key_sql(fk)
            column_definitions.append(f"    {fk_sql}")
        
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS `{self.db_name}`.`{table_name}` (
{',\n'.join(column_definitions)}
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci 
        COMMENT='文档chunk级数据表';
        """
        
        self.mysql_client.execute(create_table_sql)
        
        # 创建索引
        for index in mysql_schema.get('indexes', []):
            index_sql = self._generate_mysql_index_sql(table_name, index)
            self.mysql_client.execute(index_sql)
    
    def _create_chunk_milvus_collection(self, collection_name: str, milvus_schema: Dict):
        """创建chunk级Milvus集合"""
        
        if not milvus_schema:
            milvus_schema = self._get_default_chunk_schema()['milvus']
        
        success = self.milvus_client.create_collection(collection_name, milvus_schema)
        if not success:
            raise Exception(f"创建Milvus集合失败: {collection_name}")
    
    def get_document_info(self, doc_id: int) -> Optional[Dict]:
        """
        获取文档详细信息
        
        Args:
            doc_id: 文档ID
        
        Returns:
            Dict: 文档信息
        """
        query_sql = f"""
        SELECT * FROM `{self.db_name}`.`documents` WHERE id = %s
        """
        
        result = self.mysql_client.fetch_one(query_sql, (doc_id,))
        if result:
            # 解析JSON字段
            if result.get('metadata'):
                result['metadata'] = json.loads(result['metadata'])
            if result.get('chunk_schema_config'):
                result['chunk_schema_config'] = json.loads(result['chunk_schema_config'])
            
            # 解析关键词
            if result.get('keywords'):
                result['keywords'] = result['keywords'].split(',')
        
        return result
    
    def list_documents(self, status: str = None, limit: int = 100, offset: int = 0) -> List[Dict]:
        """
        列出文档
        
        Args:
            status: 处理状态过滤
            limit: 返回记录数量
            offset: 偏移量
        
        Returns:
            List[Dict]: 文档列表
        """
        where_clause = ""
        params = []
        
        if status:
            where_clause = "WHERE processing_status = %s"
            params.append(status)
        
        query_sql = f"""
        SELECT id, title, summary, keywords, processing_status, chunk_count, 
               file_path, file_size, created_at, updated_at
        FROM `{self.db_name}`.`documents`
        {where_clause}
        ORDER BY created_at DESC
        LIMIT %s OFFSET %s
        """
        params.extend([limit, offset])
        
        results = self.mysql_client.fetch_all(query_sql, params)
        
        # 处理结果
        for result in results:
            if result.get('keywords'):
                result['keywords'] = result['keywords'].split(',')
        
        return results
    
    def delete_document(self, doc_id: int, force: bool = False) -> bool:
        """
        删除文档及其所有chunk级数据
        
        Args:
            doc_id: 文档ID
            force: 是否强制删除
        
        Returns:
            bool: 删除是否成功
        """
        
        # 获取文档信息
        doc_info = self.get_document_info(doc_id)
        if not doc_info:
            logger.warning(f"文档 ID {doc_id} 不存在")
            return False
        
        if not force:
            print(f"⚠️  即将删除文档: {doc_info['title']}")
            print(f"   - Chunk MySQL表: {doc_info['chunk_mysql_table']}")
            print(f"   - Chunk Milvus集合: {doc_info['chunk_milvus_collection']}")
            print(f"   - Chunk数量: {doc_info['chunk_count']}")
            
            confirm = input("确认删除吗？(yes/no): ")
            if confirm.lower() != 'yes':
                print("取消删除")
                return False
        
        try:
            # 1. 删除chunk级MySQL表
            if doc_info['chunk_mysql_table']:
                drop_sql = f"DROP TABLE IF EXISTS `{self.db_name}`.`{doc_info['chunk_mysql_table']}`"
                self.mysql_client.execute(drop_sql)
                logger.info(f"Chunk级MySQL表已删除: {doc_info['chunk_mysql_table']}")
            
            # 2. 删除chunk级Milvus集合
            if doc_info['chunk_milvus_collection']:
                self.milvus_client.drop_collection(doc_info['chunk_milvus_collection'])
                logger.info(f"Chunk级Milvus集合已删除: {doc_info['chunk_milvus_collection']}")
            
            # 3. 删除文档级向量
            documents_collection_name = f"{self.db_name}_documents_vectors"
            self.milvus_client.delete(documents_collection_name, doc_id)
            logger.info(f"文档级向量已删除: {doc_id}")
            
            # 4. 删除文档记录
            delete_sql = f"DELETE FROM `{self.db_name}`.`documents` WHERE id = %s"
            self.mysql_client.execute(delete_sql, (doc_id,))
            
            logger.info(f"文档 '{doc_info['title']}' (ID: {doc_id}) 删除完成")
            return True
            
        except Exception as e:
            logger.error(f"删除文档失败: {e}")
            return False
    
    def update_document_status(self, doc_id: int, status: str, chunk_count: int = None) -> bool:
        """
        更新文档处理状态
        
        Args:
            doc_id: 文档ID
            status: 新状态
            chunk_count: chunk数量
        
        Returns:
            bool: 更新是否成功
        """
        
        set_clauses = ["processing_status = %s", "updated_at = CURRENT_TIMESTAMP"]
        params = [status]
        
        if chunk_count is not None:
            set_clauses.append("chunk_count = %s")
            params.append(chunk_count)
        
        params.append(doc_id)
        
        update_sql = f"""
        UPDATE `{self.db_name}`.`documents` 
        SET {', '.join(set_clauses)}
        WHERE id = %s
        """
        
        try:
            affected_rows = self.mysql_client.execute(update_sql, params)
            return affected_rows > 0
        except Exception as e:
            logger.error(f"更新文档状态失败: {e}")
            return False
    
    def _get_default_chunk_schema(self) -> Dict:
        """获取默认的chunk级schema配置"""
        
        return {
            "mysql": {
                "columns": [
                    {"name": "id", "type": "BIGINT", "auto_increment": True, "primary_key": True, "comment": "chunk ID"},
                    {"name": "chunk_index", "type": "INT", "not_null": True, "comment": "chunk在文档中的序号"},
                    {"name": "chunk_text", "type": "LONGTEXT", "not_null": True, "comment": "chunk文本内容"},
                    {"name": "chunk_title", "type": "VARCHAR(500)", "comment": "chunk标题"},
                    {"name": "keywords", "type": "TEXT", "comment": "chunk关键词"},
                    {"name": "metadata", "type": "JSON", "comment": "chunk元数据"},
                    {"name": "token_count", "type": "INT", "comment": "token数量"},
                    {"name": "created_at", "type": "DATETIME", "default": "CURRENT_TIMESTAMP", "comment": "创建时间"}
                ],
                "indexes": [
                    {"name": "idx_chunk_index", "columns": ["chunk_index"], "type": "INDEX"},
                    {"name": "fulltext_content", "columns": ["chunk_text"], "type": "FULLTEXT"},
                    {"name": "fulltext_keywords", "columns": ["keywords"], "type": "FULLTEXT"}
                ]
            },
            "milvus": {
                "fields": [
                    {"name": "id", "type": "INT64", "is_primary": True, "auto_id": False, "comment": "chunk ID"},
                    {"name": "chunk_embedding", "type": "FLOAT_VECTOR", "dim": 768, "comment": "chunk文本向量"},
                    {"name": "title_embedding", "type": "FLOAT_VECTOR", "dim": 768, "comment": "chunk标题向量"},
                    {"name": "metadata", "type": "JSON", "comment": "chunk元数据"}
                ],
                "indexes": [
                    {"field": "chunk_embedding", "type": "IVF_FLAT", "metric": "L2", "params": {"nlist": 1024}},
                    {"field": "title_embedding", "type": "IVF_FLAT", "metric": "L2", "params": {"nlist": 512}}
                ]
            }
        }
    
    def _generate_mysql_column_sql(self, column: Dict) -> str:
        """生成MySQL字段定义SQL"""
        col_name = column['name']
        col_type = column['type']
        
        sql_parts = [f"`{col_name}`", col_type]
        
        if column.get('not_null', False):
            sql_parts.append("NOT NULL")
        
        if column.get('auto_increment', False):
            sql_parts.append("AUTO_INCREMENT")
        
        if column.get('primary_key', False):
            sql_parts.append("PRIMARY KEY")
        
        default_value = column.get('default')
        if default_value is not None:
            if isinstance(default_value, str) and default_value.upper() not in ['CURRENT_TIMESTAMP', 'NULL']:
                sql_parts.append(f"DEFAULT {default_value}")
            else:
                sql_parts.append(f"DEFAULT {default_value}")
        
        comment = column.get('comment')
        if comment:
            sql_parts.append(f"COMMENT '{comment}'")
        
        return " ".join(sql_parts)
    
    def _generate_mysql_foreign_key_sql(self, fk: Dict) -> str:
        """生成MySQL外键约束SQL"""
        column = fk['column']
        ref_table = fk['ref_table']
        ref_column = fk['ref_column']
        
        fk_sql = f"FOREIGN KEY (`{column}`) REFERENCES `{ref_table}`(`{ref_column}`)"
        
        if fk.get('on_delete'):
            fk_sql += f" ON DELETE {fk['on_delete']}"
        
        if fk.get('on_update'):
            fk_sql += f" ON UPDATE {fk['on_update']}"
        
        return fk_sql
    
    def _generate_mysql_index_sql(self, table_name: str, index: Dict) -> str:
        """生成MySQL索引创建SQL"""
        index_name = index['name']
        columns = index['columns']
        index_type = index.get('type', 'INDEX')
        
        if index_type.upper() == 'FULLTEXT':
            sql = f"CREATE FULLTEXT INDEX `{index_name}` ON `{self.db_name}`.`{table_name}` ({', '.join([f'`{col}`' for col in columns])})"
        elif index_type.upper() == 'UNIQUE':
            sql = f"CREATE UNIQUE INDEX `{index_name}` ON `{self.db_name}`.`{table_name}` ({', '.join([f'`{col}`' for col in columns])})"
        else:
            sql = f"CREATE INDEX `{index_name}` ON `{self.db_name}`.`{table_name}` ({', '.join([f'`{col}`' for col in columns])})"
        
        return sql
    
    def _cleanup_failed_document_creation(self, doc_id: int, mysql_table: str, milvus_collection: str):
        """清理失败的文档创建"""
        try:
            # 删除文档记录
            delete_sql = f"DELETE FROM `{self.db_name}`.`documents` WHERE id = %s"
            self.mysql_client.execute(delete_sql, (doc_id,))
            
            # 删除可能已创建的表和集合
            if mysql_table:
                drop_sql = f"DROP TABLE IF EXISTS `{self.db_name}`.`{mysql_table}`"
                self.mysql_client.execute(drop_sql)
            
            if milvus_collection:
                self.milvus_client.drop_collection(milvus_collection)
            
            # 删除文档级向量
            documents_collection_name = f"{self.db_name}_documents_vectors"
            self.milvus_client.delete(documents_collection_name, doc_id)
            
        except Exception as e:
            logger.error(f"清理失败的文档创建时出错: {e}")