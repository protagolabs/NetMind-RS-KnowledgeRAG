"""
KnowledgeRAG MySQL客户端工具
作者: XYZ-Algorithm-Team
用途: 提供MySQL数据库的连接和操作接口
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, date
import mysql.connector
from mysql.connector import Error
from mysql.connector.pooling import MySQLConnectionPool
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@dataclass
class ChunkIn:
    """Chunk输入数据类"""
    seq_no: int
    chunk_uid: str
    text: str
    section_path: Optional[str] = None
    page_no: Optional[int] = None
    token_count: int = 0

@dataclass
class ChunkOut:
    """Chunk输出数据类"""
    id: int
    version_id: int
    chunk_uid: str
    seq_no: int
    section_path: Optional[str]
    page_no: Optional[int]
    text: str
    token_count: int
    created_at: datetime

@dataclass
class DocumentInfo:
    """文档信息数据类"""
    id: int
    user_id: int
    doc_uuid: str
    title: str
    mime_type: Optional[str]
    created_at: datetime
    latest_version_id: Optional[int]

@dataclass
class DocumentVersionInfo:
    """文档版本信息数据类"""
    id: int
    document_id: int
    version_label: str
    source_uri: str
    checksum: str
    effective_date: Optional[date]
    uploaded_at: datetime
    parsed_status: str

class MySQLClient:
    """MySQL客户端"""
    
    def __init__(self, host: str, port: int, user: str, password: str, database: str,
                 pool_name: str = "knowledge_rag_pool", pool_size: int = 10):
        """
        初始化MySQL客户端
        
        Args:
            host: MySQL主机地址
            port: MySQL端口
            user: MySQL用户名
            password: MySQL密码
            database: 数据库名
            pool_name: 连接池名称
            pool_size: 连接池大小
        """
        self.config = {
            'host': host,
            'port': port,
            'user': user,
            'password': password,
            'database': database,
            'charset': 'utf8mb4',
            'collation': 'utf8mb4_unicode_ci',
            'autocommit': False,
            'pool_name': pool_name,
            'pool_size': pool_size,
            'pool_reset_session': True,
            'sql_mode': 'STRICT_TRANS_TABLES,NO_ENGINE_SUBSTITUTION',
            'connect_timeout': 10,
            'autocommit': True,
            'connection_timeout': 20,
            'auth_plugin': 'mysql_native_password'
        }
        
        self.pool: Optional[MySQLConnectionPool] = None
        self._initialize_pool()
    
    def _initialize_pool(self):
        """初始化连接池"""
        try:
            self.pool = MySQLConnectionPool(**self.config)
            logger.info(f"MySQL连接池初始化成功，大小: {self.config['pool_size']}")
        except Error as e:
            logger.error(f"MySQL连接池初始化失败: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """获取数据库连接上下文管理器"""
        connection = None
        try:
            connection = self.pool.get_connection()
            yield connection
        except Error as e:
            logger.error(f"获取MySQL连接失败: {e}")
            if connection:
                connection.rollback()
            raise
        finally:
            if connection:
                connection.close()
    
    def create_document(self, user_id: int, title: str, mime_type: Optional[str] = None) -> int:
        """
        创建文档记录
        
        Args:
            user_id: 用户ID
            title: 文档标题
            mime_type: MIME类型
            
        Returns:
            文档ID
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # 生成文档UUID
                import uuid
                doc_uuid = str(uuid.uuid4())
                
                # 插入文档记录
                insert_sql = """
                INSERT INTO documents (user_id, doc_uuid, title, mime_type, created_at)
                VALUES (%s, %s, %s, %s, %s)
                """
                
                cursor.execute(insert_sql, (user_id, doc_uuid, title, mime_type, datetime.now()))
                doc_id = cursor.lastrowid
                
                conn.commit()
                cursor.close()
                
                logger.info(f"创建文档成功: doc_id={doc_id}, doc_uuid={doc_uuid}")
                return doc_id
                
        except Error as e:
            logger.error(f"创建文档失败: {e}")
            raise
    
    def create_version(self, doc_id: int, source_uri: str, version_label: str, 
                      checksum: str, effective_date: Optional[date] = None) -> int:
        """
        创建文档版本记录
        
        Args:
            doc_id: 文档ID
            source_uri: 源URI
            version_label: 版本标签
            checksum: 文件校验和
            effective_date: 生效日期
            
        Returns:
            版本ID
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # 插入版本记录
                insert_sql = """
                INSERT INTO document_versions (document_id, version_label, source_uri, checksum, 
                                             effective_date, uploaded_at, parsed_status)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """
                
                cursor.execute(insert_sql, (doc_id, version_label, source_uri, checksum,
                                          effective_date, datetime.now(), 'pending'))
                version_id = cursor.lastrowid
                
                # 更新文档的最新版本ID
                update_sql = """
                UPDATE documents SET latest_version_id = %s WHERE id = %s
                """
                cursor.execute(update_sql, (version_id, doc_id))
                
                conn.commit()
                cursor.close()
                
                logger.info(f"创建版本成功: version_id={version_id}, version_label={version_label}")
                return version_id
                
        except Error as e:
            logger.error(f"创建版本失败: {e}")
            raise
    
    def bulk_insert_chunks(self, version_id: int, chunk_records: List[ChunkIn]) -> List[int]:
        """
        批量插入chunk记录
        
        Args:
            version_id: 版本ID
            chunk_records: chunk记录列表
            
        Returns:
            插入的chunk ID列表
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # 准备批量插入数据
                insert_sql = """
                INSERT INTO chunks (version_id, chunk_uid, seq_no, section_path, page_no, text, token_count)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """
                
                data = []
                for chunk in chunk_records:
                    data.append((
                        version_id,
                        chunk.chunk_uid,
                        chunk.seq_no,
                        chunk.section_path,
                        chunk.page_no,
                        chunk.text,
                        chunk.token_count
                    ))
                
                # 批量插入
                cursor.executemany(insert_sql, data)
                
                # 获取插入的ID范围
                first_id = cursor.lastrowid
                chunk_ids = list(range(first_id, first_id + len(chunk_records)))
                
                conn.commit()
                cursor.close()
                
                logger.info(f"批量插入chunk成功: {len(chunk_records)} 条记录")
                return chunk_ids
                
        except Error as e:
            logger.error(f"批量插入chunk失败: {e}")
            raise
    
    def link_embedding(self, chunk_id: int, vector_ref: str, model_name: str, dim: int) -> int:
        """
        关联embedding记录
        
        Args:
            chunk_id: chunk ID
            vector_ref: 向量引用
            model_name: 模型名称
            dim: 向量维度
            
        Returns:
            embedding ID
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # 插入embedding记录
                insert_sql = """
                INSERT INTO embeddings (chunk_id, model_name, dim, vector_ref, created_at)
                VALUES (%s, %s, %s, %s, %s)
                """
                
                cursor.execute(insert_sql, (chunk_id, model_name, dim, vector_ref, datetime.now()))
                embedding_id = cursor.lastrowid
                
                conn.commit()
                cursor.close()
                
                logger.info(f"关联embedding成功: embedding_id={embedding_id}")
                return embedding_id
                
        except Error as e:
            logger.error(f"关联embedding失败: {e}")
            raise
    
    def get_chunks(self, version_id: int, limit: Optional[int] = None, offset: int = 0) -> List[ChunkOut]:
        """
        获取版本的chunk列表
        
        Args:
            version_id: 版本ID
            limit: 限制数量
            offset: 偏移量
            
        Returns:
            chunk列表
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(dictionary=True)
                
                # 构建查询SQL
                select_sql = """
                SELECT id, version_id, chunk_uid, seq_no, section_path, page_no, text, token_count, created_at
                FROM chunks
                WHERE version_id = %s
                ORDER BY seq_no
                """
                
                params = [version_id]
                
                if limit:
                    select_sql += " LIMIT %s"
                    params.append(limit)
                
                if offset:
                    select_sql += " OFFSET %s"
                    params.append(offset)
                
                cursor.execute(select_sql, params)
                rows = cursor.fetchall()
                
                # 转换为数据类
                chunks = []
                for row in rows:
                    chunks.append(ChunkOut(
                        id=row['id'],
                        version_id=row['version_id'],
                        chunk_uid=row['chunk_uid'],
                        seq_no=row['seq_no'],
                        section_path=row['section_path'],
                        page_no=row['page_no'],
                        text=row['text'],
                        token_count=row['token_count'],
                        created_at=row['created_at']
                    ))
                
                cursor.close()
                
                logger.info(f"获取chunks成功: {len(chunks)} 条记录")
                return chunks
                
        except Error as e:
            logger.error(f"获取chunks失败: {e}")
            raise
    
    def resolve_latest_version(self, doc_uuid: str) -> Optional[DocumentVersionInfo]:
        """
        解析文档的最新版本
        
        Args:
            doc_uuid: 文档UUID
            
        Returns:
            最新版本信息
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(dictionary=True)
                
                # 查询最新版本
                select_sql = """
                SELECT dv.id, dv.document_id, dv.version_label, dv.source_uri, dv.checksum,
                       dv.effective_date, dv.uploaded_at, dv.parsed_status
                FROM document_versions dv
                JOIN documents d ON dv.document_id = d.id
                WHERE d.doc_uuid = %s
                ORDER BY dv.uploaded_at DESC
                LIMIT 1
                """
                
                cursor.execute(select_sql, (doc_uuid,))
                row = cursor.fetchone()
                
                cursor.close()
                
                if row:
                    version_info = DocumentVersionInfo(
                        id=row['id'],
                        document_id=row['document_id'],
                        version_label=row['version_label'],
                        source_uri=row['source_uri'],
                        checksum=row['checksum'],
                        effective_date=row['effective_date'],
                        uploaded_at=row['uploaded_at'],
                        parsed_status=row['parsed_status']
                    )
                    logger.info(f"解析最新版本成功: {version_info.version_label}")
                    return version_info
                else:
                    logger.warning(f"未找到文档版本: {doc_uuid}")
                    return None
                
        except Error as e:
            logger.error(f"解析最新版本失败: {e}")
            raise
    
    def fetch_metadata_for_chunks(self, chunk_ids: List[int]) -> List[Dict[str, Any]]:
        """
        获取chunk的元数据
        
        Args:
            chunk_ids: chunk ID列表
            
        Returns:
            chunk元数据列表
        """
        try:
            if not chunk_ids:
                return []
            
            with self.get_connection() as conn:
                cursor = conn.cursor(dictionary=True)
                
                # 构建查询SQL
                placeholders = ','.join(['%s'] * len(chunk_ids))
                select_sql = f"""
                SELECT c.id, c.chunk_uid, c.seq_no, c.section_path, c.page_no, c.text, c.token_count,
                       dv.version_label, dv.source_uri, dv.effective_date,
                       d.doc_uuid, d.title, d.mime_type,
                       u.name as user_name
                FROM chunks c
                JOIN document_versions dv ON c.version_id = dv.id
                JOIN documents d ON dv.document_id = d.id
                JOIN users u ON d.user_id = u.id
                WHERE c.id IN ({placeholders})
                ORDER BY c.seq_no
                """
                
                cursor.execute(select_sql, chunk_ids)
                rows = cursor.fetchall()
                
                cursor.close()
                
                logger.info(f"获取chunk元数据成功: {len(rows)} 条记录")
                return rows
                
        except Error as e:
            logger.error(f"获取chunk元数据失败: {e}")
            raise
    
    def get_document_info(self, doc_uuid: str) -> Optional[DocumentInfo]:
        """
        获取文档信息
        
        Args:
            doc_uuid: 文档UUID
            
        Returns:
            文档信息
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(dictionary=True)
                
                select_sql = """
                SELECT id, user_id, doc_uuid, title, mime_type, created_at, latest_version_id
                FROM documents
                WHERE doc_uuid = %s
                """
                
                cursor.execute(select_sql, (doc_uuid,))
                row = cursor.fetchone()
                
                cursor.close()
                
                if row:
                    return DocumentInfo(
                        id=row['id'],
                        user_id=row['user_id'],
                        doc_uuid=row['doc_uuid'],
                        title=row['title'],
                        mime_type=row['mime_type'],
                        created_at=row['created_at'],
                        latest_version_id=row['latest_version_id']
                    )
                else:
                    return None
                
        except Error as e:
            logger.error(f"获取文档信息失败: {e}")
            raise
    
    def update_parsed_status(self, version_id: int, status: str) -> bool:
        """
        更新版本解析状态
        
        Args:
            version_id: 版本ID
            status: 状态 ('pending', 'ok', 'error')
            
        Returns:
            更新成功返回True
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                update_sql = """
                UPDATE document_versions 
                SET parsed_status = %s 
                WHERE id = %s
                """
                
                cursor.execute(update_sql, (status, version_id))
                affected_rows = cursor.rowcount
                
                conn.commit()
                cursor.close()
                
                logger.info(f"更新解析状态成功: version_id={version_id}, status={status}")
                return affected_rows > 0
                
        except Error as e:
            logger.error(f"更新解析状态失败: {e}")
            raise
    
    def get_user_documents(self, user_id: int) -> List[DocumentInfo]:
        """
        获取用户的所有文档
        
        Args:
            user_id: 用户ID
            
        Returns:
            文档信息列表
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(dictionary=True)
                
                select_sql = """
                SELECT id, user_id, doc_uuid, title, mime_type, created_at, latest_version_id
                FROM documents
                WHERE user_id = %s
                ORDER BY created_at DESC
                """
                
                cursor.execute(select_sql, (user_id,))
                rows = cursor.fetchall()
                
                cursor.close()
                
                documents = []
                for row in rows:
                    documents.append(DocumentInfo(
                        id=row['id'],
                        user_id=row['user_id'],
                        doc_uuid=row['doc_uuid'],
                        title=row['title'],
                        mime_type=row['mime_type'],
                        created_at=row['created_at'],
                        latest_version_id=row['latest_version_id']
                    ))
                
                logger.info(f"获取用户文档成功: {len(documents)} 个文档")
                return documents
                
        except Error as e:
            logger.error(f"获取用户文档失败: {e}")
            raise
    
    def create_chunks(self, version_id: int, chunks: List[ChunkIn]) -> bool:
        """
        批量创建文本块记录
        
        Args:
            version_id: 版本ID
            chunks: 文本块列表
            
        Returns:
            创建是否成功
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # 批量插入文本块
                insert_sql = """
                INSERT INTO chunks (version_id, chunk_uid, seq_no, section_path, 
                                  page_no, text, token_count, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                chunk_data = []
                for chunk in chunks:
                    chunk_data.append((
                        version_id,
                        chunk.chunk_uid,
                        chunk.seq_no,
                        chunk.section_path,
                        chunk.page_no,
                        chunk.text,
                        chunk.token_count,
                        datetime.now()
                    ))
                
                cursor.executemany(insert_sql, chunk_data)
                conn.commit()
                cursor.close()
                
                logger.info(f"批量创建文本块成功: {len(chunks)} 个")
                return True
                
        except Error as e:
            logger.error(f"批量创建文本块失败: {e}")
            return False
    
    def list_tables(self) -> List[str]:
        """
        获取数据库中的所有表名
        
        Returns:
            表名列表
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SHOW TABLES")
                tables = [row[0] for row in cursor.fetchall()]
                cursor.close()
                return tables
        except Error as e:
            logger.error(f"获取表列表失败: {e}")
            return []
    
    def close(self):
        """关闭连接池"""
        if self.pool:
            # 连接池会自动管理连接，这里不需要特别操作
            logger.info("MySQL连接池已关闭")

# 全局客户端实例
_mysql_client = None

def get_mysql_client() -> MySQLClient:
    """
    获取全局MySQL客户端实例
    
    Returns:
        MySQL客户端实例
    """
    global _mysql_client
    if _mysql_client is None:
        config = {
            'host': os.getenv('MYSQL_HOST', '127.0.0.1'),
            'port': int(os.getenv('MYSQL_PORT', 3306)),
            'user': os.getenv('MYSQL_USER', 'root'),
            'password': os.getenv('MYSQL_PASSWORD', 'devpass'),
            'database': os.getenv('MYSQL_DB', 'knowledge_rag')
        }
        _mysql_client = MySQLClient(**config)
    return _mysql_client 