"""
ChunkDataManager - Chunk级数据管理器

这个模块提供了便捷的chunk级数据操作接口，让用户可以轻松地：
1. 向特定文档添加chunks
2. 批量处理chunk数据
3. 管理chunk的MySQL和Milvus数据

作者: XYZ-Algorithm-Team
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from .rag_document_manager import RAGDocumentManager
from .database_clients import MySQLClient, MilvusClient

logger = logging.getLogger(__name__)

class ChunkDataManager:
    """
    Chunk数据管理器
    
    提供便捷的chunk级数据操作接口，用户可以：
    - 向文档添加单个或批量chunks
    - 管理chunk的MySQL和Milvus数据
    - 查询和删除chunk数据
    
    使用示例:
        chunk_manager = ChunkDataManager()
        
        # 添加单个chunk
        chunk_id = chunk_manager.add_chunk_to_document(
            doc_id=123,
            chunk_text="这是chunk内容...",
            chunk_embedding=[0.1, 0.2, ...]
        )
        
        # 批量添加chunks
        chunk_ids = chunk_manager.add_chunks_to_document(
            doc_id=123,
            chunks_data=[
                {
                    "chunk_text": "第一个chunk",
                    "chunk_embedding": [0.1, 0.2, ...]
                },
                # ... 更多chunks
            ]
        )
    """
    
    def __init__(self, experiment_name: str = None):
        """
        初始化Chunk数据管理器
        
        Args:
            experiment_name: 实验名称
        """
        self.rag_manager = RAGDocumentManager(experiment_name)
        self.mysql_client = self.rag_manager.mysql_client
        self.milvus_client = self.rag_manager.milvus_client
        self.db_name = self.rag_manager.db_name
        
        logger.info("ChunkDataManager 初始化完成")
    
    def add_chunk_to_document(
        self,
        doc_id: int,
        chunk_text: str,
        chunk_embedding: List[float] = None,
        chunk_index: int = None,
        chunk_title: str = None,
        keywords: str = None,
        title_embedding: List[float] = None,
        metadata: Dict = None,
        token_count: int = None
    ) -> int:
        """
        向文档添加单个chunk
        
        Args:
            doc_id: 文档ID
            chunk_text: chunk文本内容
            chunk_embedding: chunk文本的向量
            chunk_index: chunk在文档中的序号（如果不提供会自动计算）
            chunk_title: chunk标题
            keywords: chunk关键词
            title_embedding: chunk标题的向量
            metadata: chunk元数据
            token_count: token数量
        
        Returns:
            int: chunk ID
        """
        
        # 获取文档信息
        doc_info = self.rag_manager.get_document_info(doc_id)
        if not doc_info:
            raise ValueError(f"文档 {doc_id} 不存在")
        
        # 获取chunk级数据库信息
        chunk_mysql_table = doc_info['chunk_mysql_table']
        chunk_milvus_collection = doc_info['chunk_milvus_collection']
        
        if not chunk_mysql_table or not chunk_milvus_collection:
            raise ValueError(f"文档 {doc_id} 的chunk级数据库不存在")
        
        try:
            # 1. 如果没有提供chunk_index，自动计算
            if chunk_index is None:
                chunk_index = self._get_next_chunk_index(chunk_mysql_table)
            
            # 2. 插入到MySQL
            mysql_data = {
                "chunk_index": chunk_index,
                "chunk_text": chunk_text,
                "chunk_title": chunk_title,
                "keywords": keywords,
                "metadata": json.dumps(metadata or {}, ensure_ascii=False),
                "token_count": token_count
            }
            
            chunk_id = self._insert_chunk_to_mysql(chunk_mysql_table, mysql_data)
            logger.info(f"Chunk插入MySQL成功，ID: {chunk_id}")
            
            # 3. 插入到Milvus（如果提供了向量）
            if chunk_embedding:
                vector_data = {
                    "id": chunk_id,
                    "chunk_embedding": chunk_embedding,
                    "title_embedding": title_embedding or [0.0] * len(chunk_embedding),
                    "metadata": metadata or {}
                }
                
                milvus_id = self.milvus_client.insert(chunk_milvus_collection, vector_data)
                logger.info(f"Chunk向量插入Milvus成功，ID: {milvus_id}")
            
            # 4. 更新文档的chunk计数
            self._update_document_chunk_count(doc_id)
            
            logger.info(f"Chunk添加完成，文档: {doc_id}, Chunk ID: {chunk_id}")
            return chunk_id
            
        except Exception as e:
            logger.error(f"添加chunk失败: {e}")
            # 尝试清理可能已插入的数据
            try:
                if 'chunk_id' in locals():
                    self._cleanup_failed_chunk_insertion(chunk_mysql_table, chunk_milvus_collection, chunk_id)
            except:
                pass
            raise
    
    def add_chunks_to_document(
        self,
        doc_id: int,
        chunks_data: List[Dict]
    ) -> List[int]:
        """
        批量向文档添加chunks
        
        Args:
            doc_id: 文档ID
            chunks_data: chunk数据列表
                格式示例: [
                    {
                        "chunk_text": "第一个chunk内容",
                        "chunk_embedding": [0.1, 0.2, ...],
                        "chunk_title": "第一章",
                        "keywords": "关键词1,关键词2",
                        "title_embedding": [0.3, 0.4, ...],
                        "metadata": {"source": "page_1"},
                        "token_count": 150
                    },
                    # ... 更多chunks
                ]
        
        Returns:
            List[int]: chunk ID列表
        """
        
        if not chunks_data:
            return []
        
        # 获取文档信息
        doc_info = self.rag_manager.get_document_info(doc_id)
        if not doc_info:
            raise ValueError(f"文档 {doc_id} 不存在")
        
        chunk_mysql_table = doc_info['chunk_mysql_table']
        chunk_milvus_collection = doc_info['chunk_milvus_collection']
        
        logger.info(f"开始批量添加 {len(chunks_data)} 个chunks到文档 {doc_id}")
        
        chunk_ids = []
        successful_count = 0
        
        try:
            # 获取起始chunk_index
            start_chunk_index = self._get_next_chunk_index(chunk_mysql_table)
            
            # 批量处理
            for i, chunk_data in enumerate(chunks_data):
                try:
                    # 设置chunk_index
                    if "chunk_index" not in chunk_data:
                        chunk_data["chunk_index"] = start_chunk_index + i
                    
                    # 添加单个chunk
                    chunk_id = self.add_chunk_to_document(
                        doc_id=doc_id,
                        chunk_text=chunk_data["chunk_text"],
                        chunk_embedding=chunk_data.get("chunk_embedding"),
                        chunk_index=chunk_data["chunk_index"],
                        chunk_title=chunk_data.get("chunk_title"),
                        keywords=chunk_data.get("keywords"),
                        title_embedding=chunk_data.get("title_embedding"),
                        metadata=chunk_data.get("metadata"),
                        token_count=chunk_data.get("token_count")
                    )
                    
                    chunk_ids.append(chunk_id)
                    successful_count += 1
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"已处理 {i + 1}/{len(chunks_data)} 个chunks")
                        
                except Exception as e:
                    logger.error(f"处理第 {i+1} 个chunk失败: {e}")
                    chunk_ids.append(None)  # 占位符，表示失败
            
            logger.info(f"批量添加完成: {successful_count}/{len(chunks_data)} 个chunks成功")
            return chunk_ids
            
        except Exception as e:
            logger.error(f"批量添加chunks失败: {e}")
            raise
    
    def get_document_chunks(
        self,
        doc_id: int,
        chunk_indices: List[int] = None,
        limit: int = 100,
        offset: int = 0,
        include_vectors: bool = False
    ) -> List[Dict]:
        """
        获取文档的chunks
        
        Args:
            doc_id: 文档ID
            chunk_indices: 指定的chunk序号列表，如果不提供则返回所有chunks
            limit: 返回数量限制
            offset: 偏移量
            include_vectors: 是否包含向量数据
        
        Returns:
            List[Dict]: chunk列表
        """
        
        # 获取文档信息
        doc_info = self.rag_manager.get_document_info(doc_id)
        if not doc_info:
            raise ValueError(f"文档 {doc_id} 不存在")
        
        chunk_mysql_table = doc_info['chunk_mysql_table']
        chunk_milvus_collection = doc_info['chunk_milvus_collection']
        
        # 构建查询SQL
        where_conditions = []
        params = []
        
        if chunk_indices:
            placeholders = ','.join(['%s'] * len(chunk_indices))
            where_conditions.append(f"chunk_index IN ({placeholders})")
            params.extend(chunk_indices)
        
        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
        
        query_sql = f"""
        SELECT * FROM `{self.db_name}`.`{chunk_mysql_table}`
        WHERE {where_clause}
        ORDER BY chunk_index ASC
        LIMIT %s OFFSET %s
        """
        params.extend([limit, offset])
        
        try:
            chunks = self.mysql_client.fetch_all(query_sql, params)
            
            # 处理结果
            for chunk in chunks:
                if chunk.get('metadata'):
                    chunk['metadata'] = json.loads(chunk['metadata'])
                chunk['document_id'] = doc_id
            
            # 如果需要包含向量数据
            if include_vectors and chunks:
                chunk_ids = [chunk['id'] for chunk in chunks]
                vector_data = self._get_chunks_vectors(chunk_milvus_collection, chunk_ids)
                
                # 合并向量数据
                vector_dict = {v['id']: v for v in vector_data}
                for chunk in chunks:
                    if chunk['id'] in vector_dict:
                        chunk['vector_data'] = vector_dict[chunk['id']]
            
            logger.info(f"获取文档 {doc_id} 的 {len(chunks)} 个chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"获取chunks失败: {e}")
            return []
    
    def delete_chunk(self, doc_id: int, chunk_id: int) -> bool:
        """
        删除指定的chunk
        
        Args:
            doc_id: 文档ID
            chunk_id: chunk ID
        
        Returns:
            bool: 删除是否成功
        """
        
        # 获取文档信息
        doc_info = self.rag_manager.get_document_info(doc_id)
        if not doc_info:
            logger.warning(f"文档 {doc_id} 不存在")
            return False
        
        chunk_mysql_table = doc_info['chunk_mysql_table']
        chunk_milvus_collection = doc_info['chunk_milvus_collection']
        
        try:
            # 1. 删除MySQL记录
            delete_mysql_sql = f"""
            DELETE FROM `{self.db_name}`.`{chunk_mysql_table}` 
            WHERE id = %s
            """
            mysql_affected = self.mysql_client.execute(delete_mysql_sql, (chunk_id,))
            
            # 2. 删除Milvus记录
            milvus_success = self.milvus_client.delete(chunk_milvus_collection, chunk_id)
            
            # 3. 更新文档chunk计数
            if mysql_affected > 0:
                self._update_document_chunk_count(doc_id)
            
            success = mysql_affected > 0
            logger.info(f"删除chunk {'成功' if success else '失败'}: 文档 {doc_id}, Chunk {chunk_id}")
            return success
            
        except Exception as e:
            logger.error(f"删除chunk失败: {e}")
            return False
    
    def delete_document_chunks(self, doc_id: int, chunk_indices: List[int] = None) -> int:
        """
        删除文档的chunks
        
        Args:
            doc_id: 文档ID
            chunk_indices: 要删除的chunk序号列表，如果不提供则删除所有chunks
        
        Returns:
            int: 删除的chunk数量
        """
        
        # 获取文档信息
        doc_info = self.rag_manager.get_document_info(doc_id)
        if not doc_info:
            logger.warning(f"文档 {doc_id} 不存在")
            return 0
        
        chunk_mysql_table = doc_info['chunk_mysql_table']
        chunk_milvus_collection = doc_info['chunk_milvus_collection']
        
        try:
            # 构建删除条件
            where_conditions = []
            params = []
            
            if chunk_indices:
                placeholders = ','.join(['%s'] * len(chunk_indices))
                where_conditions.append(f"chunk_index IN ({placeholders})")
                params.extend(chunk_indices)
            
            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
            
            # 1. 获取要删除的chunk IDs
            query_sql = f"""
            SELECT id FROM `{self.db_name}`.`{chunk_mysql_table}` 
            WHERE {where_clause}
            """
            
            chunk_records = self.mysql_client.fetch_all(query_sql, params)
            chunk_ids = [record['id'] for record in chunk_records]
            
            if not chunk_ids:
                logger.info(f"文档 {doc_id} 没有找到要删除的chunks")
                return 0
            
            # 2. 删除MySQL记录
            delete_mysql_sql = f"""
            DELETE FROM `{self.db_name}`.`{chunk_mysql_table}` 
            WHERE {where_clause}
            """
            mysql_deleted = self.mysql_client.execute(delete_mysql_sql, params)
            
            # 3. 批量删除Milvus记录
            milvus_deleted = 0
            for chunk_id in chunk_ids:
                if self.milvus_client.delete(chunk_milvus_collection, chunk_id):
                    milvus_deleted += 1
            
            # 4. 更新文档chunk计数
            if mysql_deleted > 0:
                self._update_document_chunk_count(doc_id)
            
            logger.info(f"删除chunks完成: 文档 {doc_id}, MySQL: {mysql_deleted}, Milvus: {milvus_deleted}")
            return mysql_deleted
            
        except Exception as e:
            logger.error(f"删除chunks失败: {e}")
            return 0
    
    def update_chunk(
        self,
        doc_id: int,
        chunk_id: int,
        updates: Dict
    ) -> bool:
        """
        更新chunk数据
        
        Args:
            doc_id: 文档ID
            chunk_id: chunk ID
            updates: 更新数据
                格式示例: {
                    "chunk_text": "新的chunk内容",
                    "keywords": "新关键词",
                    "chunk_embedding": [0.1, 0.2, ...],  # 向量更新
                    "metadata": {"updated": True}
                }
        
        Returns:
            bool: 更新是否成功
        """
        
        # 获取文档信息
        doc_info = self.rag_manager.get_document_info(doc_id)
        if not doc_info:
            logger.warning(f"文档 {doc_id} 不存在")
            return False
        
        chunk_mysql_table = doc_info['chunk_mysql_table']
        chunk_milvus_collection = doc_info['chunk_milvus_collection']
        
        try:
            mysql_success = True
            milvus_success = True
            
            # 分离MySQL更新和Milvus更新
            mysql_updates = {}
            milvus_updates = {}
            
            for key, value in updates.items():
                if key.endswith('_embedding'):
                    milvus_updates[key] = value
                elif key in ['metadata'] and isinstance(value, dict):
                    # metadata可能需要同时更新MySQL和Milvus
                    mysql_updates[key] = json.dumps(value, ensure_ascii=False)
                    milvus_updates[key] = value
                else:
                    mysql_updates[key] = value
            
            # 1. 更新MySQL
            if mysql_updates:
                set_clauses = [f"`{field}` = %s" for field in mysql_updates.keys()]
                params = list(mysql_updates.values()) + [chunk_id]
                
                update_mysql_sql = f"""
                UPDATE `{self.db_name}`.`{chunk_mysql_table}`
                SET {', '.join(set_clauses)}
                WHERE id = %s
                """
                
                affected_rows = self.mysql_client.execute(update_mysql_sql, params)
                mysql_success = affected_rows > 0
            
            # 2. 更新Milvus
            if milvus_updates:
                milvus_success = self.milvus_client.update(chunk_milvus_collection, chunk_id, milvus_updates)
            
            success = mysql_success and milvus_success
            logger.info(f"更新chunk {'成功' if success else '失败'}: 文档 {doc_id}, Chunk {chunk_id}")
            return success
            
        except Exception as e:
            logger.error(f"更新chunk失败: {e}")
            return False
    
    def get_document_chunk_statistics(self, doc_id: int) -> Dict:
        """
        获取文档的chunk统计信息
        
        Args:
            doc_id: 文档ID
        
        Returns:
            Dict: 统计信息
        """
        
        # 获取文档信息
        doc_info = self.rag_manager.get_document_info(doc_id)
        if not doc_info:
            return {}
        
        chunk_mysql_table = doc_info['chunk_mysql_table']
        
        try:
            stats_sql = f"""
            SELECT 
                COUNT(*) as total_chunks,
                AVG(token_count) as avg_token_count,
                MAX(token_count) as max_token_count,
                MIN(token_count) as min_token_count,
                SUM(token_count) as total_tokens
            FROM `{self.db_name}`.`{chunk_mysql_table}`
            """
            
            stats = self.mysql_client.fetch_one(stats_sql)
            
            # 添加文档信息
            stats['document_id'] = doc_id
            stats['document_title'] = doc_info['title']
            stats['chunk_mysql_table'] = chunk_mysql_table
            stats['chunk_milvus_collection'] = doc_info['chunk_milvus_collection']
            
            logger.info(f"获取文档 {doc_id} 统计信息完成")
            return stats
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}
    
    # ======================================
    # 私有辅助方法
    # ======================================
    
    def _get_next_chunk_index(self, table_name: str) -> int:
        """获取下一个chunk序号"""
        query_sql = f"""
        SELECT COALESCE(MAX(chunk_index), 0) + 1 as next_index 
        FROM `{self.db_name}`.`{table_name}`
        """
        
        result = self.mysql_client.fetch_one(query_sql)
        return result['next_index'] if result else 1
    
    def _insert_chunk_to_mysql(self, table_name: str, data: Dict) -> int:
        """插入chunk到MySQL并返回ID"""
        columns = [k for k, v in data.items() if v is not None]
        values = [v for k, v in data.items() if v is not None]
        
        placeholders = ', '.join(['%s'] * len(columns))
        column_names = ', '.join([f'`{col}`' for col in columns])
        
        insert_sql = f"""
        INSERT INTO `{self.db_name}`.`{table_name}` 
        ({column_names}) VALUES ({placeholders})
        """
        
        return self.mysql_client.execute(insert_sql, values, fetch_lastrowid=True)
    
    def _get_chunks_vectors(self, collection_name: str, chunk_ids: List[int]) -> List[Dict]:
        """获取多个chunks的向量数据"""
        try:
            vectors = []
            for chunk_id in chunk_ids:
                vector_data = self.milvus_client.get_by_id(collection_name, chunk_id)
                if vector_data:
                    vectors.append(vector_data)
            return vectors
        except Exception as e:
            logger.error(f"获取向量数据失败: {e}")
            return []
    
    def _update_document_chunk_count(self, doc_id: int):
        """更新文档的chunk计数"""
        try:
            # 获取实际的chunk数量
            doc_info = self.rag_manager.get_document_info(doc_id)
            if not doc_info:
                return
            
            count_sql = f"""
            SELECT COUNT(*) as count FROM `{self.db_name}`.`{doc_info['chunk_mysql_table']}`
            """
            
            result = self.mysql_client.fetch_one(count_sql)
            actual_count = result['count'] if result else 0
            
            # 更新documents表中的chunk_count
            update_sql = f"""
            UPDATE `{self.db_name}`.`documents` 
            SET chunk_count = %s, updated_at = CURRENT_TIMESTAMP 
            WHERE id = %s
            """
            
            self.mysql_client.execute(update_sql, (actual_count, doc_id))
            logger.debug(f"文档 {doc_id} chunk计数已更新: {actual_count}")
            
        except Exception as e:
            logger.error(f"更新chunk计数失败: {e}")
    
    def _cleanup_failed_chunk_insertion(self, mysql_table: str, milvus_collection: str, chunk_id: int):
        """清理失败的chunk插入"""
        try:
            # 删除可能已插入的MySQL记录
            delete_mysql_sql = f"DELETE FROM `{self.db_name}`.`{mysql_table}` WHERE id = %s"
            self.mysql_client.execute(delete_mysql_sql, (chunk_id,))
            
            # 删除可能已插入的Milvus记录
            self.milvus_client.delete(milvus_collection, chunk_id)
            
        except Exception as e:
            logger.error(f"清理失败的chunk插入时出错: {e}")