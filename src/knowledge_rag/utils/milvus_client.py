"""
KnowledgeRAG Milvus客户端工具
作者: XYZ-Algorithm-Team
用途: 提供Milvus向量数据库的连接和操作接口
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from datetime import datetime

# 动态导入Milvus相关模块
try:
    from pymilvus import (
        connections, Collection, CollectionSchema, FieldSchema, DataType,
        utility, db, MilvusException
    )
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    logging.warning("Milvus客户端依赖未安装，请安装 pymilvus")

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """搜索结果数据类"""
    embedding_id: int
    user_id: int
    doc_uuid: str
    version_label: str
    chunk_uid: str
    distance: float
    timestamp: int

@dataclass
class EmbeddingData:
    """嵌入数据类"""
    embedding_id: int
    user_id: int
    doc_uuid: str
    version_label: str
    chunk_uid: str
    vector: List[float]
    timestamp: Optional[int] = None

class MilvusClient:
    """Milvus客户端"""
    
    def __init__(self, host: str = "localhost", port: int = 19530, 
                 collection_name: str = "rag_embeddings_v1", 
                 vector_dim: int = 1536, alias: str = "default"):
        """
        初始化Milvus客户端
        
        Args:
            host: Milvus服务器地址
            port: Milvus服务器端口
            collection_name: 集合名称
            vector_dim: 向量维度
            alias: 连接别名
        """
        if not MILVUS_AVAILABLE:
            raise ImportError("Milvus客户端依赖未安装")
        
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.alias = alias
        self.collection: Optional[Collection] = None
        
        self._connect()
        self._initialize_collection()
    
    def _connect(self):
        """连接到Milvus服务器"""
        try:
            # 检查连接是否已存在
            if connections.has_connection(self.alias):
                connections.disconnect(self.alias)
            
            # 创建连接
            connections.connect(
                alias=self.alias,
                host=self.host,
                port=self.port
            )
            
            logger.info(f"成功连接到Milvus服务器: {self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"连接Milvus服务器失败: {e}")
            raise
    
    def _create_collection_schema(self) -> CollectionSchema:
        """创建集合Schema"""
        fields = [
            FieldSchema(name="embedding_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="user_id", dtype=DataType.INT64),
            FieldSchema(name="doc_uuid", dtype=DataType.VARCHAR, max_length=36),
            FieldSchema(name="version_label", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="chunk_uid", dtype=DataType.VARCHAR, max_length=36),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim),
            FieldSchema(name="ts", dtype=DataType.INT64)
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="KnowledgeRAG embeddings collection",
            enable_dynamic_field=True
        )
        
        return schema
    
    def _initialize_collection(self):
        """初始化集合"""
        try:
            # 检查集合是否存在
            if utility.has_collection(self.collection_name, using=self.alias):
                logger.info(f"集合已存在: {self.collection_name}")
                self.collection = Collection(self.collection_name, using=self.alias)
            else:
                # 创建集合
                schema = self._create_collection_schema()
                self.collection = Collection(
                    name=self.collection_name,
                    schema=schema,
                    using=self.alias
                )
                logger.info(f"创建集合成功: {self.collection_name}")
            
            # 创建索引
            self._create_index()
            
            # 加载集合到内存
            self.collection.load()
            logger.info(f"集合加载成功: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"初始化集合失败: {e}")
            raise
    
    def _create_index(self):
        """创建向量索引"""
        try:
            # 检查索引是否已存在
            if self.collection.has_index():
                logger.info("索引已存在")
                return
            
            # 创建HNSW索引
            index_params = {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {
                    "M": 16,
                    "efConstruction": 200
                }
            }
            
            self.collection.create_index(
                field_name="vector",
                index_params=index_params
            )
            
            logger.info("向量索引创建成功")
            
        except Exception as e:
            logger.error(f"创建索引失败: {e}")
            raise
    
    def create_collection_if_not_exists(self) -> bool:
        """
        如果集合不存在则创建
        
        Returns:
            创建成功返回True
        """
        try:
            if not utility.has_collection(self.collection_name, using=self.alias):
                self._initialize_collection()
                return True
            return False
            
        except Exception as e:
            logger.error(f"创建集合失败: {e}")
            raise
    
    def upsert_embedding(self, embedding_id: int, user_id: int, doc_uuid: str, 
                        version_label: str, chunk_uid: str, vector: List[float]) -> bool:
        """
        插入或更新嵌入向量
        
        Args:
            embedding_id: 嵌入ID
            user_id: 用户ID
            doc_uuid: 文档UUID
            version_label: 版本标签
            chunk_uid: 块UUID
            vector: 向量数据
            
        Returns:
            操作成功返回True
        """
        try:
            # 准备数据
            data = [
                [embedding_id],
                [user_id],
                [doc_uuid],
                [version_label],
                [chunk_uid],
                [vector],
                [int(datetime.now().timestamp() * 1000)]  # 毫秒时间戳
            ]
            
            # 插入数据
            mr = self.collection.upsert(data)
            
            # 检查插入结果
            if mr.insert_count > 0:
                logger.debug(f"向量插入成功: embedding_id={embedding_id}")
                return True
            else:
                logger.warning(f"向量插入失败: embedding_id={embedding_id}")
                return False
                
        except Exception as e:
            logger.error(f"向量插入失败: {e}")
            raise
    
    def batch_upsert_embeddings(self, embeddings: List[EmbeddingData]) -> bool:
        """
        批量插入嵌入向量
        
        Args:
            embeddings: 嵌入数据列表
            
        Returns:
            操作成功返回True
        """
        try:
            if not embeddings:
                return True
            
            # 准备批量数据
            embedding_ids = []
            user_ids = []
            doc_uuids = []
            version_labels = []
            chunk_uids = []
            vectors = []
            timestamps = []
            
            current_ts = int(datetime.now().timestamp() * 1000)
            
            for emb in embeddings:
                embedding_ids.append(emb.embedding_id)
                user_ids.append(emb.user_id)
                doc_uuids.append(emb.doc_uuid)
                version_labels.append(emb.version_label)
                chunk_uids.append(emb.chunk_uid)
                vectors.append(emb.vector)
                timestamps.append(emb.timestamp or current_ts)
            
            # 批量插入
            data = [
                embedding_ids,
                user_ids,
                doc_uuids,
                version_labels,
                chunk_uids,
                vectors,
                timestamps
            ]
            
            mr = self.collection.upsert(data)
            
            if mr.insert_count > 0:
                logger.info(f"批量向量插入成功: {len(embeddings)} 条记录")
                return True
            else:
                logger.warning(f"批量向量插入失败")
                return False
                
        except Exception as e:
            logger.error(f"批量向量插入失败: {e}")
            raise
    
    def search(self, query_vector: List[float], top_k: int = 10, 
               user_id: Optional[int] = None, doc_uuid: Optional[str] = None,
               version_label: Optional[str] = None, ts_range: Optional[Tuple[int, int]] = None) -> List[SearchResult]:
        """
        搜索相似向量
        
        Args:
            query_vector: 查询向量
            top_k: 返回top k结果
            user_id: 用户ID过滤
            doc_uuid: 文档UUID过滤
            version_label: 版本标签过滤
            ts_range: 时间戳范围过滤 (start_ts, end_ts)
            
        Returns:
            搜索结果列表
        """
        try:
            # 构建搜索参数
            search_params = {
                "metric_type": "COSINE",
                "params": {
                    "ef": 200
                }
            }
            
            # 构建过滤表达式
            filter_expressions = []
            
            if user_id is not None:
                filter_expressions.append(f"user_id == {user_id}")
            
            if doc_uuid is not None:
                filter_expressions.append(f'doc_uuid == "{doc_uuid}"')
            
            if version_label is not None:
                filter_expressions.append(f'version_label == "{version_label}"')
            
            if ts_range is not None:
                start_ts, end_ts = ts_range
                filter_expressions.append(f"ts >= {start_ts} and ts <= {end_ts}")
            
            # 组合过滤条件
            filter_expr = " and ".join(filter_expressions) if filter_expressions else None
            
            # 执行搜索
            results = self.collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                expr=filter_expr,
                output_fields=["embedding_id", "user_id", "doc_uuid", "version_label", "chunk_uid", "ts"]
            )
            
            # 处理搜索结果
            search_results = []
            for hits in results:
                for hit in hits:
                    search_results.append(SearchResult(
                        embedding_id=hit.entity.get("embedding_id"),
                        user_id=hit.entity.get("user_id"),
                        doc_uuid=hit.entity.get("doc_uuid"),
                        version_label=hit.entity.get("version_label"),
                        chunk_uid=hit.entity.get("chunk_uid"),
                        distance=hit.distance,
                        timestamp=hit.entity.get("ts")
                    ))
            
            logger.info(f"搜索完成: 返回 {len(search_results)} 条结果")
            return search_results
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            raise
    
    def delete_embeddings(self, chunk_uid_list: List[str]) -> bool:
        """
        删除指定chunk的嵌入向量
        
        Args:
            chunk_uid_list: 块UUID列表
            
        Returns:
            删除成功返回True
        """
        try:
            if not chunk_uid_list:
                return True
            
            # 构建删除表达式
            chunk_uids_str = '", "'.join(chunk_uid_list)
            expr = f'chunk_uid in ["{chunk_uids_str}"]'
            
            # 执行删除
            mr = self.collection.delete(expr)
            
            if mr.delete_count > 0:
                logger.info(f"删除向量成功: {mr.delete_count} 条记录")
                return True
            else:
                logger.warning(f"未找到要删除的向量")
                return False
                
        except Exception as e:
            logger.error(f"删除向量失败: {e}")
            raise
    
    def delete_by_user(self, user_id: int) -> bool:
        """
        删除指定用户的所有嵌入向量
        
        Args:
            user_id: 用户ID
            
        Returns:
            删除成功返回True
        """
        try:
            expr = f"user_id == {user_id}"
            mr = self.collection.delete(expr)
            
            if mr.delete_count > 0:
                logger.info(f"删除用户向量成功: user_id={user_id}, 删除数量={mr.delete_count}")
                return True
            else:
                logger.warning(f"未找到用户的向量: user_id={user_id}")
                return False
                
        except Exception as e:
            logger.error(f"删除用户向量失败: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取集合统计信息
        
        Returns:
            统计信息字典
        """
        try:
            stats = self.collection.get_stats()
            
            # 解析统计信息
            stats_dict = {}
            for stat in stats:
                stats_dict[stat.name] = stat.value
            
            # 获取索引信息
            indexes = self.collection.indexes
            index_info = []
            for index in indexes:
                index_info.append({
                    "field_name": index.field_name,
                    "index_type": index.index_type,
                    "metric_type": index.metric_type,
                    "params": index.params
                })
            
            result = {
                "collection_name": self.collection_name,
                "stats": stats_dict,
                "indexes": index_info,
                "schema": {
                    "fields": [
                        {
                            "name": field.name,
                            "type": field.dtype.name,
                            "is_primary": field.is_primary
                        }
                        for field in self.collection.schema.fields
                    ]
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"获取集合统计失败: {e}")
            raise
    
    def flush(self):
        """刷新集合，确保数据持久化"""
        try:
            self.collection.flush()
            logger.info("集合刷新成功")
        except Exception as e:
            logger.error(f"集合刷新失败: {e}")
            raise
    
    def compact(self):
        """压缩集合，优化存储空间"""
        try:
            self.collection.compact()
            logger.info("集合压缩成功")
        except Exception as e:
            logger.error(f"集合压缩失败: {e}")
            raise
    
    def drop_collection(self):
        """删除集合"""
        try:
            if utility.has_collection(self.collection_name, using=self.alias):
                utility.drop_collection(self.collection_name, using=self.alias)
                logger.info(f"集合删除成功: {self.collection_name}")
            else:
                logger.warning(f"集合不存在: {self.collection_name}")
        except Exception as e:
            logger.error(f"删除集合失败: {e}")
            raise
    
    def close(self):
        """关闭连接"""
        try:
            if self.collection:
                self.collection.release()
            
            if connections.has_connection(self.alias):
                connections.disconnect(self.alias)
            
            logger.info("Milvus连接已关闭")
            
        except Exception as e:
            logger.error(f"关闭连接失败: {e}")

# 全局客户端实例
_milvus_client = None

def get_milvus_client(host: str = None, port: int = None, 
                     collection_name: str = None, vector_dim: int = None) -> MilvusClient:
    """
    获取全局Milvus客户端实例
    
    Args:
        host: Milvus服务器地址
        port: Milvus服务器端口
        collection_name: 集合名称
        vector_dim: 向量维度
        
    Returns:
        Milvus客户端实例
    """
    global _milvus_client
    if _milvus_client is None:
        # 从配置文件获取维度
        if vector_dim is None:
            from ..config import get_embedding_settings
            embedding_settings = get_embedding_settings()
            vector_dim = embedding_settings.dimension
        
        config = {
            'host': host or os.getenv('MILVUS_HOST', 'localhost'),
            'port': port or int(os.getenv('MILVUS_PORT', 19530)),
            'collection_name': collection_name or os.getenv('MILVUS_COLLECTION', 'rag_embeddings_v1'),
            'vector_dim': vector_dim
        }
        _milvus_client = MilvusClient(**config)
    return _milvus_client 