"""
数据存储引擎 - 将处理好的文档和chunk数据存储到数据库
====================================================

【核心职责】
这个脚本是整个知识库构建流程的关键环节，负责将经过处理的文档数据持久化存储：
1. 读取experiments_docs_processed/中的JSON文件（来自step_1处理结果）
2. 动态创建对应的MySQL表和Milvus集合（按需扩展）
3. 批量插入文档和chunk数据到数据库（高效存储）
4. 创建向量索引以支持高效搜索（性能优化）

【存储架构设计】
采用分层存储策略，平衡查询效率和存储效率：

文档级别（Document Level）：
- MySQL表：rag_documents（统一存储所有文档元数据）
- Milvus集合：documents_vectors（文档摘要和关键词向量）
- Milvus集合：documents_insights_vectors（文档核心观点向量）

Chunk级别（Chunk Level）：
- MySQL表：chunks_{source_id}（按文档分组存储chunk内容）
- Milvus集合：chunks_vectors_{source_id}（chunk向量数据）
- Milvus集合：chunks_insights_vectors_{source_id}（chunk观点向量）

【为什么这样设计？】
1. 文档级统一存储：便于全局搜索和文档管理
2. Chunk级分组存储：避免单表过大，提高查询性能
3. 向量分离存储：不同类型向量有不同的搜索需求
4. 动态表创建：支持增量添加新文档，无需预定义所有表

【数据流向】
JSON文件 → 内存处理 → MySQL结构化存储 → Milvus向量存储 → 索引优化

【技术特点】
- 批量操作：提高插入效率
- 事务安全：确保数据一致性
- 错误恢复：跳过问题数据，继续处理
- 进度跟踪：实时显示处理进度
- 资源管理：自动清理数据库连接

作者: XYZ-Algorithm-Team
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import mysql.connector
from tqdm import tqdm

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.knowledge_rag.config import KnowledgeRAGSettings
from src.knowledge_rag.db_utils.rag_schema_config import RAGSchemaConfig
from src.knowledge_rag.db_utils.database_clients import MySQLClient, MilvusClient

# Milvus连接
try:
    from pymilvus import connections
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    logging.warning("PyMilvus 未安装，Milvus 功能不可用")

logger = logging.getLogger(__name__)

class DataToDBSaver:
    """
    数据库存储器 - 将处理好的数据存储到MySQL和Milvus
    ===============================================
    
    【核心功能】
    这个类是整个数据存储流程的核心引擎，负责：
    1. 管理MySQL和Milvus双数据库连接
    2. 动态创建数据库表结构和向量集合
    3. 批量高效地插入大量数据
    4. 处理数据转换和格式验证
    5. 提供错误恢复和进度跟踪
    
    【设计模式】
    采用"工厂模式"和"策略模式"：
    - 工厂模式：根据source_id动态创建表和集合
    - 策略模式：针对不同类型数据采用不同的存储策略
    - 批量处理：提高大数据量插入效率
    - 容错设计：单个数据失败不影响整体流程
    
    【连接管理】
    维护两种数据库连接：
    - MySQL连接：用于结构化数据存储和查询
    - Milvus连接：用于向量数据存储和相似度搜索
    """
    
    def __init__(self, config: KnowledgeRAGSettings):
        """
        初始化数据库存储器 - 建立双数据库连接
        
        【初始化流程】
        1. 保存配置信息（数据库连接参数、表结构定义等）
        2. 初始化数据库客户端对象
        3. 建立MySQL和Milvus连接
        4. 验证连接状态
        
        【配置说明】
        config包含：
        - database: MySQL连接配置（host, port, user, password等）
        - milvus: Milvus连接配置（host, port, alias等）
        - 其他系统配置参数
        
        Args:
            config: KnowledgeRAG配置对象，包含所有数据库连接和系统配置
            
        Raises:
            Exception: 数据库连接失败时抛出异常，确保系统不会在不完整状态下运行
        """
        self.config = config
        self.schema_config = RAGSchemaConfig()  # 表结构配置管理器
        
        # 数据库连接状态管理
        self.mysql_conn = None      # MySQL原生连接对象
        self.mysql_client = None    # MySQL客户端封装（提供便捷方法）
        self.milvus_client = None   # Milvus客户端封装
        
        # 立即建立数据库连接（失败会抛出异常）
        self._connect_databases()
    
    def _connect_databases(self):
        """
        连接到MySQL和Milvus数据库
        
        【连接策略】
        1. 先连接MySQL：关系型数据库，存储结构化数据
        2. 再连接Milvus：向量数据库，存储向量数据
        3. 启用autocommit：简化事务管理（适合批量插入场景）
        4. 使用UTF8MB4字符集：支持emoji和特殊字符
        
        【错误处理】
        - 任何连接失败都会抛出异常
        - 确保系统不会在不完整状态下运行
        - 提供详细的错误日志便于调试
        
        【连接参数说明】
        MySQL参数：
        - autocommit=True：自动提交，避免事务管理复杂性
        - charset=utf8mb4：支持完整的Unicode字符集
        
        Milvus参数：
        - alias：连接别名，便于管理多个连接
        - 使用默认连接池配置
        """
        try:
            # 第一步：连接MySQL关系型数据库
            # 用于存储文档和chunk的结构化信息（文本、元数据等）
            self.mysql_conn = mysql.connector.connect(
                host=self.config.database.host,           # 数据库服务器地址
                port=self.config.database.port,           # MySQL端口（默认3306）
                user=self.config.database.user,           # 用户名
                password=self.config.database.password,   # 密码
                database=self.config.database.database,   # 目标数据库名
                charset=self.config.database.charset,     # 字符集（推荐utf8mb4）
                autocommit=True  # 自动提交，简化事务管理（适合批量插入）
            )
            # 创建MySQL客户端封装，提供便捷的数据库操作方法
            self.mysql_client = MySQLClient(self.mysql_conn)
            logger.info("已连接到MySQL数据库")
            
            # 第二步：连接Milvus向量数据库
            # 检查PyMilvus是否可用
            if not MILVUS_AVAILABLE:
                raise ImportError("PyMilvus 未安装，无法使用 Milvus 功能")
            
            # 建立Milvus连接
            # 用于存储文档和chunk的向量表示（embedding）
            connections.connect(
                alias=self.config.milvus.alias,    # 连接别名，便于管理
                host=self.config.milvus.host,      # Milvus服务地址
                port=self.config.milvus.port       # Milvus端口（默认19530）
            )
            
            # 创建Milvus客户端封装，提供便捷的向量数据库操作方法
            self.milvus_client = MilvusClient(self.config.milvus.alias)
            logger.info("已连接到Milvus向量数据库")
            
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            # 连接失败立即抛出异常，避免系统在不完整状态下运行
            raise
    
    def create_document_tables_and_collections(self):
        """
        创建文档级别的MySQL表和Milvus集合
        =================================
        
        【创建内容】
        为文档级数据创建完整的存储结构：
        1. rag_documents表：存储文档的基本信息和内容
        2. documents_vectors集合：存储文档摘要和关键词的向量表示
        3. documents_insights_vectors集合：存储文档核心观点的向量表示
        
        【表结构说明】
        rag_documents表包含：
        - 文档基本信息：file_id, file_name, summary等
        - 结构化内容：insights（JSON格式）, key_words（JSON格式）
        - 原始内容：doc_markdown_content（完整文档内容）
        - 处理状态：processing_status（跟踪处理进度）
        
        【向量集合说明】
        documents_vectors集合：
        - 存储文档摘要向量（summary_embedding）
        - 存储关键词向量（key_words_embedding）
        - 维度：1536（OpenAI text-embedding-ada-002模型）
        
        documents_insights_vectors集合：
        - 存储每个insight的独立向量
        - 支持更精细的语义搜索
        - 维度：1536
        
        【设计优势】
        - 统一存储：所有文档信息集中管理
        - 向量分离：不同类型向量独立存储，便于优化
        - 索引支持：为高效查询创建必要索引
        """
        try:
            # 第一步：创建rag_documents MySQL表
            # 这是文档数据的核心存储表，包含所有文档的基本信息和内容
            doc_schema = self.schema_config.get_document_mysql_schema()
            self._create_mysql_table(doc_schema["table_name"], doc_schema)
            
            # 第二步：创建documents向量集合
            # 存储文档摘要和关键词的向量表示，用于语义相似度搜索
            doc_vector_schema = self.schema_config.get_document_milvus_schema(1536)
            collection_name = "documents_vectors"
            self.milvus_client.create_collection(collection_name, doc_vector_schema)
            
            # 第三步：创建documents insights向量集合
            # 存储文档核心观点的向量表示，支持更精细的语义搜索
            doc_insights_schema = self.schema_config.get_document_insights_milvus_schema(1536)
            insights_collection_name = "documents_insights_vectors"
            self.milvus_client.create_collection(insights_collection_name, doc_insights_schema)
            
            logger.info("文档级别的表和集合创建完成")
            
        except Exception as e:
            logger.error(f"创建文档级别表和集合失败: {e}")
            raise
    
    def create_chunk_tables_and_collections(self, source_ids: List[str]):
        """
        为每个source_id创建chunk级别的MySQL表和Milvus集合
        ==============================================
        
        【核心思想】
        采用"分而治之"的策略，为每个文档（source_id）创建独立的存储结构：
        - 避免单表过大，提高查询性能
        - 支持增量添加新文档，无需预定义所有表
        - 便于数据管理和维护
        
        【命名规则】
        MySQL表：chunks_{source_id}（如：chunks_paper_001）
        Milvus集合：chunks_vectors_{source_id}（如：chunks_vectors_paper_001）
        Insights集合：chunks_insights_vectors_{source_id}
        
        【创建内容】
        为每个source_id创建：
        1. MySQL表：存储chunk的结构化信息（文本、元数据等）
        2. Milvus向量集合：存储chunk的向量表示
        3. Milvus insights集合：存储chunk观点的向量表示
        
        【性能考虑】
        - 并行创建：多个表/集合可以并行创建
        - 进度跟踪：使用tqdm显示创建进度
        - 错误隔离：单个source_id失败不影响其他
        
        Args:
            source_ids: 所有唯一的source_id列表
                - 每个source_id代表一个独立的文档
                - 会为每个source_id创建独立的存储结构
                - 支持动态扩展，无需预定义所有文档
        
        【设计优势】
        - 可扩展性：新文档可以独立添加，不影响现有数据
        - 查询效率：小表查询比大表查询更快
        - 维护便利：可以独立备份、删除特定文档的数据
        - 资源隔离：不同文档的数据相互独立
        """
        try:
            # 获取chunk表结构和向量集合配置
            chunk_schema = self.schema_config.get_chunk_schema()
            chunk_insights_schema = self.schema_config.get_chunk_insights_milvus_schema(1536)
            
            # 为每个source_id创建独立的存储结构
            for source_id in tqdm(source_ids, desc="创建chunk表和集合"):
                # 第一步：为每个source_id创建MySQL表
                # 表名格式：chunks_{source_id}，如chunks_paper_001
                table_name = self.schema_config.get_chunk_table_name(source_id)
                chunk_mysql_schema = chunk_schema["mysql"]
                chunk_mysql_schema["table_name"] = table_name
                self._create_mysql_table(table_name, chunk_mysql_schema)
                
                # 第二步：为每个source_id创建Milvus向量集合
                # 集合名格式：chunks_vectors_{source_id}
                collection_name = self.schema_config.get_chunk_collection_name(source_id)
                chunk_milvus_schema = chunk_schema["milvus"]
                chunk_milvus_schema["collection_name"] = collection_name
                self.milvus_client.create_collection(collection_name, chunk_milvus_schema)
                
                # 第三步：为每个source_id创建insights向量集合
                # 集合名格式：chunks_insights_vectors_{source_id}
                insights_collection_name = self.schema_config.get_chunk_insights_collection_name(source_id)
                chunk_insights_schema["collection_name"] = insights_collection_name
                self.milvus_client.create_collection(insights_collection_name, chunk_insights_schema)
            
            logger.info(f"为{len(source_ids)}个source_id创建了chunk表和集合")
            
        except Exception as e:
            logger.error(f"创建chunk表和集合失败: {e}")
            raise
    
    def _create_mysql_table(self, table_name: str, schema: Dict):
        """
        创建MySQL表 - 动态表结构生成
        ===========================
        
        【核心功能】
        根据配置的schema动态生成并执行CREATE TABLE语句：
        1. 解析schema配置，生成完整的表结构
        2. 执行CREATE TABLE IF NOT EXISTS语句
        3. 创建必要的索引以支持高效查询
        4. 处理约束冲突和重复索引
        
        【Schema配置格式】
        schema字典包含：
        - columns: 列定义列表，每个列包含name, type, 约束等
        - indexes: 索引定义列表，支持普通索引、唯一索引、全文索引
        - 其他表级配置
        
        【索引策略】
        - 自动跳过重复索引：避免"索引已存在"错误
        - 智能约束处理：UNIQUE约束和UNIQUE索引的冲突处理
        - 全文索引支持：为文本搜索字段创建FULLTEXT索引
        
        【错误处理】
        - 使用IF NOT EXISTS：避免表已存在错误
        - 索引创建失败时记录警告但不中断流程
        - 提供详细的错误日志便于调试
        
        Args:
            table_name: 要创建的表名
                - 支持动态表名（如chunks_{source_id}）
                - 使用反引号包围，避免关键字冲突
                
            schema: 表结构配置字典
                - 包含完整的列定义和索引配置
                - 支持复杂的约束和索引类型
                - 由RAGSchemaConfig提供标准化配置
        """
        try:
            # 第一步：解析schema配置，生成列定义SQL
            columns_sql = []
            for col in schema["columns"]:
                # 基础列定义：列名 + 数据类型
                col_sql = f"`{col['name']}` {col['type']}"
                
                # 添加约束条件
                if col.get('not_null'):
                    col_sql += " NOT NULL"  # 非空约束
                if col.get('auto_increment'):
                    col_sql += " AUTO_INCREMENT"  # 自增主键
                if col.get('primary_key'):
                    col_sql += " PRIMARY KEY"  # 主键约束
                if col.get('unique'):
                    col_sql += " UNIQUE"  # 唯一约束
                if col.get('default'):
                    col_sql += f" DEFAULT {col['default']}"  # 默认值
                if col.get('comment'):
                    col_sql += f" COMMENT '{col['comment']}'"  # 列注释
                
                columns_sql.append(col_sql)
            
            # 第二步：生成完整的CREATE TABLE语句
            create_sql = f"""
            CREATE TABLE IF NOT EXISTS `{table_name}` (
                {', '.join(columns_sql)}
            ) ENGINE=InnoDB DEFAULT CHARSET={self.config.database.charset} 
            COLLATE={self.config.database.collation}
            """
            # 注意：
            # - IF NOT EXISTS：避免表已存在错误
            # - InnoDB引擎：支持事务和外键
            # - 使用配置的字符集和排序规则
            
            logger.info(f"准备执行SQL: {create_sql}")
            self.mysql_client.execute(create_sql)
            
            # 第三步：创建索引（如果schema中定义了索引）
            if "indexes" in schema:
                # 收集已有UNIQUE约束的字段，避免重复创建UNIQUE索引
                unique_columns = set()
                for col in schema["columns"]:
                    if col.get('unique'):
                        unique_columns.add(col['name'])
                
                # 遍历所有索引配置
                for index in schema["indexes"]:
                    # 智能跳过：如果字段已有UNIQUE约束，跳过对应的UNIQUE索引
                    if (index.get("type") == "UNIQUE" and 
                        len(index["columns"]) == 1 and 
                        index["columns"][0] in unique_columns):
                        logger.info(f"跳过索引 {index['name']}，字段已有UNIQUE约束")
                        continue
                    
                    # 创建索引
                    self._create_mysql_index(table_name, index)
            
            logger.info(f"MySQL表 {table_name} 创建完成")
            
        except Exception as e:
            logger.error(f"创建MySQL表 {table_name} 失败: {e}")
            raise
    
    def _create_mysql_index(self, table_name: str, index_config: Dict):
        """
        创建MySQL索引
        
        Args:
            table_name: 表名
            index_config: 索引配置
        """
        try:
            index_name = index_config["name"]
            columns = index_config["columns"]
            index_type = index_config["type"]
            
            if index_type == "FULLTEXT":
                index_sql = f"CREATE FULLTEXT INDEX `{index_name}` ON `{table_name}` ({', '.join([f'`{col}`' for col in columns])})"
            elif index_type == "UNIQUE":
                # 对于VARCHAR字段限制索引长度
                col_specs = []
                for col in columns:
                    if col == "file_name":
                        col_specs.append(f"`{col}`(255)")
                    else:
                        col_specs.append(f"`{col}`")
                index_sql = f"CREATE UNIQUE INDEX `{index_name}` ON `{table_name}` ({', '.join(col_specs)})"
            else:  # INDEX
                # 对于VARCHAR字段限制索引长度
                col_specs = []
                for col in columns:
                    if col == "file_name":
                        col_specs.append(f"`{col}`(255)")
                    else:
                        col_specs.append(f"`{col}`")
                index_sql = f"CREATE INDEX `{index_name}` ON `{table_name}` ({', '.join(col_specs)})"
            
            self.mysql_client.execute(index_sql)
            
        except mysql.connector.Error as e:
            if e.errno == 1061:  # Duplicate key name
                logger.warning(f"索引 {index_name} 已存在，跳过创建")
            else:
                logger.error(f"创建索引 {index_name} 失败: {e}")
                raise
    
    def save_documents_to_db(self, docs_data: List[Dict]):
        """
        保存文档数据到数据库 - 三阶段存储策略
        ===================================
        
        【存储策略】
        采用"先MySQL后Milvus"的三阶段存储策略：
        1. 准备阶段：将数据转换为适合存储的格式
        2. MySQL阶段：插入结构化数据，获取自增ID
        3. Milvus阶段：使用MySQL ID插入向量数据
        
        【数据流向】
        JSON数据 → MySQL结构化存储 → 获取ID → Milvus向量存储
        
        【存储内容】
        MySQL (rag_documents表)：
        - 文档基本信息：file_id, file_name, summary
        - 结构化内容：insights, key_words（JSON格式）
        - 原始内容：doc_markdown_content
        - 处理状态：processing_status
        
        Milvus (documents_vectors集合)：
        - 文档摘要向量：summary_embedding
        - 关键词向量：key_words_embedding
        - 元数据：file_name等
        
        Milvus (documents_insights_vectors集合)：
        - 每个insight的独立向量
        - insight文本和索引信息
        
        【技术特点】
        - INSERT IGNORE：避免重复插入
        - 批量操作：提高插入效率
        - 进度跟踪：实时显示处理进度
        - 错误隔离：单个文档失败不影响其他
        
        Args:
            docs_data: 文档数据列表
                - 每个文档包含完整的文本内容和向量表示
                - 来自step_1的处理结果
                - 包含insights和key_words的向量化结果
        """
        try:
            # 初始化存储容器
            mysql_records = []      # MySQL结构化数据
            vector_records = []     # 文档向量数据
            insights_records = []   # insights向量数据
            
            # 第一阶段：数据准备和格式转换
            for doc in tqdm(docs_data, desc="准备文档数据"):
                doc_id = None  # 将在插入MySQL后获取
                
                # 准备MySQL记录 - 结构化数据存储
                mysql_record = {
                    'file_id': doc['file_id'],                    # 文档唯一标识
                    'file_name': doc['file_name'],                # 文件名
                    'summary': doc['summary'],                    # 文档摘要
                    'insights': json.dumps(doc['insights'], ensure_ascii=False),      # 核心观点（JSON格式）
                    'key_words': json.dumps(doc['key_words'], ensure_ascii=False),    # 关键词（JSON格式）
                    'doc_markdown_content': doc['doc_markdown_content'],              # 原始文档内容
                    'processing_status': 'completed'              # 处理状态
                }
                mysql_records.append((mysql_record, doc))  # 保留原始doc引用，用于后续向量处理
            
            # 第二阶段：MySQL数据插入
            for mysql_record, original_doc in tqdm(mysql_records, desc="插入文档到MySQL"):
                # 使用INSERT IGNORE避免重复插入
                insert_sql = """
                INSERT IGNORE INTO rag_documents (file_id, file_name, summary, insights, key_words, 
                                         doc_markdown_content, processing_status)
                VALUES (%(file_id)s, %(file_name)s, %(summary)s, %(insights)s, %(key_words)s,
                        %(doc_markdown_content)s, %(processing_status)s)
                """
                
                # 插入MySQL记录并获取自增ID
                doc_id = self.mysql_client.execute(insert_sql, mysql_record, fetch_lastrowid=True)
                
                # 第三阶段：准备向量数据（使用MySQL ID）
                # 准备文档向量记录
                vector_record = {
                    'id': doc_id,                                                      # MySQL自增ID
                    'file_id': original_doc['file_id'],                               # 文档ID
                    'summary_embedding': original_doc['summary_embedding_vector'],    # 摘要向量
                    'key_words_embedding': original_doc['key_words_embedding_vector'], # 关键词向量
                    'metadata': json.dumps({'file_name': original_doc['file_name']}, ensure_ascii=False)  # 元数据
                }
                vector_records.append(vector_record)
                
                # 准备insights向量记录（每个insight一个向量）
                for idx, (insight_text, insight_vector) in enumerate(zip(
                    original_doc['insights'], original_doc['insights_embedding_vector']
                )):
                    insights_record = {
                        'doc_id': doc_id,                                             # 关联的文档ID
                        'file_id': original_doc['file_id'],                           # 文档ID
                        'insight_index': idx,                                         # insight索引
                        'insight_text': insight_text,                                 # insight文本
                        'insight_embedding': insight_vector,                          # insight向量
                        'metadata': json.dumps({'doc_file_name': original_doc['file_name']}, ensure_ascii=False)  # 元数据
                    }
                    insights_records.append(insights_record)
            
            # 第四阶段：批量插入向量数据到Milvus
            if vector_records:
                self.milvus_client.batch_insert("documents_vectors", vector_records)
                logger.info(f"插入了{len(vector_records)}条文档向量记录")
            
            if insights_records:
                self.milvus_client.batch_insert("documents_insights_vectors", insights_records)
                logger.info(f"插入了{len(insights_records)}条文档insights向量记录")
            
            logger.info(f"成功保存{len(docs_data)}个文档到数据库")
            
        except Exception as e:
            logger.error(f"保存文档数据失败: {e}")
            raise
    
    def save_chunks_to_db(self, chunks_data: List[Dict]):
        """
        保存chunk数据到数据库 - 分组存储策略
        =================================
        
        【存储策略】
        采用"分而治之"的分组存储策略：
        1. 按source_id分组：将chunks按所属文档分组
        2. 并行处理：每个source_id独立处理，提高效率
        3. 错误隔离：单个source_id失败不影响其他
        
        【数据组织】
        chunks_data → 按source_id分组 → 每个source_id独立存储
        
        【存储结构】
        每个source_id对应：
        - MySQL表：chunks_{source_id}
        - Milvus集合：chunks_vectors_{source_id}
        - Milvus集合：chunks_insights_vectors_{source_id}
        
        【性能优势】
        - 小表查询：避免单表过大导致的性能问题
        - 并行处理：多个source_id可以并行处理
        - 资源隔离：不同文档的数据相互独立
        - 增量扩展：新文档可以独立添加
        
        【错误处理】
        - 分组处理：单个source_id失败不影响其他
        - 进度跟踪：显示每个source_id的处理进度
        - 统计报告：提供详细的处理结果统计
        
        Args:
            chunks_data: chunk数据列表
                - 每个chunk包含source_id标识所属文档
                - 包含完整的文本内容和向量表示
                - 来自step_1的处理结果
        """
        try:
            # 第一步：按source_id分组chunk数据
            # 使用字典结构，key为source_id，value为该文档的所有chunks
            chunks_by_source = {}
            for chunk in chunks_data:
                source_id = chunk['source_id']
                if source_id not in chunks_by_source:
                    chunks_by_source[source_id] = []
                chunks_by_source[source_id].append(chunk)
            
            # 第二步：为每个source_id独立处理chunk数据
            # 使用tqdm显示处理进度，便于监控长时间运行的任务
            for source_id, chunks in tqdm(chunks_by_source.items(), desc="保存chunks到数据库"):
                self._save_chunks_by_source_id(source_id, chunks)
            
            # 第三步：统计处理结果
            total_chunks = sum(len(chunks) for chunks in chunks_by_source.values())
            logger.info(f"成功保存{total_chunks}个chunks到数据库")
            
        except Exception as e:
            logger.error(f"保存chunk数据失败: {e}")
            raise
    
    def _save_chunks_by_source_id(self, source_id: str, chunks: List[Dict]):
        """
        保存指定source_id的chunk数据
        
        Args:
            source_id: 文档的source_id
            chunks: 该source_id下的chunk列表
        """
        try:
            table_name = self.schema_config.get_chunk_table_name(source_id)
            collection_name = self.schema_config.get_chunk_collection_name(source_id)
            insights_collection_name = self.schema_config.get_chunk_insights_collection_name(source_id)
            
            mysql_records = []
            vector_records = []
            insights_records = []
            
            for chunk in chunks:
                # 准备MySQL记录
                mysql_record = {
                    'source_id': chunk['source_id'],
                    'chunk_id': chunk['chunk_id'],
                    'summary': chunk['summary'],
                    'insights': json.dumps(chunk['insights'], ensure_ascii=False),
                    'key_words': json.dumps(chunk['key_words'], ensure_ascii=False),
                    'chunk_markdown_content': chunk['chunk_markdown_content'],
                    'metadata': json.dumps({'processed': True}, ensure_ascii=False)
                }
                mysql_records.append((mysql_record, chunk))
            
            # 批量插入MySQL记录
            if mysql_records:
                insert_sql = f"""
                INSERT IGNORE INTO {table_name} (source_id, chunk_id, summary, insights, key_words,
                                         chunk_markdown_content, metadata)
                VALUES (%(source_id)s, %(chunk_id)s, %(summary)s, %(insights)s, %(key_words)s,
                        %(chunk_markdown_content)s, %(metadata)s)
                """
                
                # 逐条插入，更安全地处理特殊字符
                success_count = 0
                for mysql_record, original_chunk in mysql_records:
                    try:
                        single_insert_sql = f"""
                        INSERT IGNORE INTO {table_name} (source_id, chunk_id, summary, insights, key_words,
                                                 chunk_markdown_content, metadata)
                        VALUES (%(source_id)s, %(chunk_id)s, %(summary)s, %(insights)s, %(key_words)s,
                                %(chunk_markdown_content)s, %(metadata)s)
                        """
                        self.mysql_client.execute(single_insert_sql, mysql_record)
                        success_count += 1
                    except Exception as e:
                        logger.warning(f"跳过chunk {mysql_record['chunk_id']}，插入失败: {e}")
                        continue
                
                logger.info(f"成功插入 {success_count}/{len(mysql_records)} 个chunks到 {table_name}")
                
                # 获取刚插入的记录ID
                id_query = f"SELECT id, chunk_id FROM {table_name} WHERE source_id = %s ORDER BY id DESC LIMIT {len(chunks)}"
                inserted_records = self.mysql_client.fetch_all(id_query, (source_id,))
                
                # 创建chunk_id到db_id的映射
                chunk_id_to_db_id = {record['chunk_id']: record['id'] for record in inserted_records}
                
                # 准备向量记录（基于批量插入后的ID映射）
                for _, original_chunk in mysql_records:
                    chunk_db_id = chunk_id_to_db_id.get(original_chunk['chunk_id'])
                    if chunk_db_id:
                        vector_record = {
                            'id': chunk_db_id,
                            'source_id': original_chunk['source_id'],
                            'chunk_id': original_chunk['chunk_id'],
                            'summary_embedding': original_chunk['summary_embedding_vector'],
                            'key_words_embedding': original_chunk['key_words_embedding_vector'],
                            'metadata': json.dumps({'source_id': original_chunk['source_id']}, ensure_ascii=False)
                        }
                        vector_records.append(vector_record)
                        
                        # 准备insights向量记录
                        for idx, (insight_text, insight_vector) in enumerate(zip(
                            original_chunk['insights'], original_chunk['insights_embedding_vector']
                        )):
                            insights_record = {
                                'chunk_id': original_chunk['chunk_id'],
                                'source_id': original_chunk['source_id'],
                                'insight_index': idx,
                                'insight_text': insight_text,
                                'insight_embedding': insight_vector,
                                'metadata': json.dumps({'chunk_id': original_chunk['chunk_id']}, ensure_ascii=False)
                            }
                            insights_records.append(insights_record)
            
            # 批量插入向量数据
            if vector_records:
                self.milvus_client.batch_insert(collection_name, vector_records)
            
            if insights_records:
                self.milvus_client.batch_insert(insights_collection_name, insights_records)
            
            logger.info(f"source_id {source_id}: 保存了{len(chunks)}个chunks")
            
        except Exception as e:
            logger.error(f"保存source_id {source_id}的chunks失败: {e}")
            raise
    
    def close_connections(self):
        """关闭数据库连接"""
        try:
            if self.mysql_conn:
                self.mysql_conn.close()
            if MILVUS_AVAILABLE and hasattr(self, 'milvus_client'):
                connections.disconnect(self.config.milvus.alias)
            logger.info("数据库连接已关闭")
        except Exception as e:
            logger.error(f"关闭数据库连接失败: {e}")


def main():
    """
    主函数 - 执行完整的数据库存储流程
    ================================
    
    【执行流程】
    这是整个知识库构建流程的第二步，负责将处理好的数据持久化存储：
    1. 环境准备：配置日志、加载配置、建立数据库连接
    2. 数据加载：从JSON文件读取文档和chunk数据
    3. 结构创建：动态创建MySQL表和Milvus集合
    4. 数据存储：批量插入文档和chunk数据
    5. 资源清理：关闭数据库连接
    
    【数据来源】
    输入文件：
    - paper_set_1_docs.json：文档级数据（摘要、关键词、向量等）
    - paper_set_1_chunks.json：chunk级数据（文档片段、向量等）
    
    【存储结果】
    输出到数据库：
    - MySQL：结构化数据（文本、元数据等）
    - Milvus：向量数据（embedding向量）
    
    【错误处理】
    - 文件检查：确保输入文件存在
    - 异常捕获：捕获并记录所有错误
    - 资源清理：确保数据库连接正确关闭
    - 进度跟踪：实时显示处理进度
    
    【使用场景】
    - 知识库初始化：首次构建知识库
    - 增量更新：添加新的文档数据
    - 数据迁移：从其他系统迁移数据
    """
    
    # 第一步：环境准备
    # 设置日志格式，便于调试和监控
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # 第二步：加载配置
        # 从环境变量加载数据库连接配置
        config = KnowledgeRAGSettings.from_env()
        logger.info("配置加载完成")
        
        # 第三步：初始化数据库存储器
        # 建立MySQL和Milvus连接
        saver = DataToDBSaver(config)
        
        # 第四步：定义数据文件路径
        # 指向step_1处理结果的JSON文件
        docs_file = project_root / "experiments_docs_processed" / "paper_set_1_docs.json"
        chunks_file = project_root / "experiments_docs_processed" / "paper_set_1_chunks.json"
        
        # 第五步：检查输入文件是否存在
        # 确保数据源文件存在，避免后续处理失败
        if not docs_file.exists():
            raise FileNotFoundError(f"文档数据文件不存在: {docs_file}")
        if not chunks_file.exists():
            raise FileNotFoundError(f"Chunk数据文件不存在: {chunks_file}")
        
        # 第六步：加载文档数据
        logger.info("加载文档数据...")
        with open(docs_file, 'r', encoding='utf-8') as f:
            docs_data = json.load(f)
        logger.info(f"加载了{len(docs_data)}个文档")
        
        # 第七步：加载chunk数据
        logger.info("加载chunk数据...")
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        logger.info(f"加载了{len(chunks_data)}个chunks")
        
        # 第八步：分析数据结构
        # 提取所有唯一的source_id，用于创建对应的表和集合
        source_ids = list(set(chunk['source_id'] for chunk in chunks_data))
        logger.info(f"发现{len(source_ids)}个唯一的source_id")
        
        # 第九步：创建数据库结构
        # 创建文档级别的表和集合（统一存储）
        logger.info("创建文档级别的表和集合...")
        saver.create_document_tables_and_collections()
        
        # 创建chunk级别的表和集合（按source_id分组）
        logger.info("创建chunk级别的表和集合...")
        saver.create_chunk_tables_and_collections(source_ids)
        
        # 第十步：存储数据
        # 保存文档数据到数据库
        logger.info("保存文档数据...")
        saver.save_documents_to_db(docs_data)
        
        # 保存chunk数据到数据库
        logger.info("保存chunk数据...")
        saver.save_chunks_to_db(chunks_data)
        
        # 第十一步：完成处理
        logger.info("数据存储完成！")
        
    except Exception as e:
        # 错误处理：记录详细错误信息
        logger.error(f"数据存储过程失败: {e}")
        raise
    
    finally:
        # 资源清理：确保数据库连接正确关闭
        if 'saver' in locals():
            saver.close_connections()



