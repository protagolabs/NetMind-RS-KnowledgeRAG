"""
数据库客户端封装模块

提供统一的数据库操作接口：
- MySQLClient: MySQL 数据库操作
- MilvusClient: Milvus 向量数据库操作  
- ObjectStoreClient: 本地对象存储操作

作者: XYZ-Algorithm-Team
"""

import os
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

# MySQL 客户端
import mysql.connector
from mysql.connector import Error

# Milvus 客户端
try:
    from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    logging.warning("PyMilvus 未安装，Milvus 功能不可用")

logger = logging.getLogger(__name__)

class MySQLClient:
    """
    MySQL 数据库客户端封装
    
    提供常用的数据库操作方法，支持：
    - 执行 SQL 语句
    - 查询单条/多条记录
    - 事务管理
    - 连接池管理
    
    使用示例:
        mysql_client = MySQLClient(connection)
        
        # 执行插入并获取自增 ID
        new_id = mysql_client.execute(
            "INSERT INTO users (name, email) VALUES (%s, %s)",
            ("张三", "zhang@example.com"),
            fetch_lastrowid=True
        )
        
        # 查询单条记录
        user = mysql_client.fetch_one("SELECT * FROM users WHERE id = %s", (new_id,))
        
        # 查询多条记录
        users = mysql_client.fetch_all("SELECT * FROM users WHERE status = %s", ("active",))
    """
    
    def __init__(self, connection: mysql.connector.MySQLConnection):
        """
        初始化 MySQL 客户端
        
        Args:
            connection: MySQL 连接对象
        """
        self.connection = connection
        
        if not self.connection or not self.connection.is_connected():
            raise ValueError("MySQL 连接无效或已断开")
        
        logger.info("MySQLClient 初始化完成")
    
    def execute(
        self, 
        sql: str, 
        params: Tuple = None, 
        fetch_lastrowid: bool = False,
        commit: bool = True
    ) -> Union[int, None]:
        """
        执行 SQL 语句
        
        Args:
            sql: SQL 语句
            params: 参数元组
            fetch_lastrowid: 是否返回自增 ID
            commit: 是否自动提交事务
        
        Returns:
            int: 影响的行数，或者自增 ID（当 fetch_lastrowid=True 时）
            
        Raises:
            mysql.connector.Error: 当 SQL 执行失败时
        """
        cursor = None
        try:
            cursor = self.connection.cursor()
            cursor.execute(sql, params or ())
            
            if commit:
                self.connection.commit()
            
            if fetch_lastrowid:
                return cursor.lastrowid
            else:
                return cursor.rowcount
                
        except Error as e:
            if commit:
                self.connection.rollback()
            logger.error(f"SQL 执行失败: {sql}, 错误: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
    
    def execute_many(
        self, 
        sql: str, 
        params_list: List[Tuple],
        commit: bool = True
    ) -> int:
        """
        批量执行 SQL 语句
        
        Args:
            sql: SQL 语句
            params_list: 参数列表
            commit: 是否自动提交事务
        
        Returns:
            int: 影响的总行数
        """
        cursor = None
        try:
            cursor = self.connection.cursor()
            cursor.executemany(sql, params_list)
            
            if commit:
                self.connection.commit()
            
            return cursor.rowcount
            
        except Error as e:
            if commit:
                self.connection.rollback()
            logger.error(f"批量 SQL 执行失败: {sql}, 错误: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
    
    def fetch_one(self, sql: str, params: Tuple = None) -> Optional[Dict]:
        """
        查询单条记录
        
        Args:
            sql: 查询 SQL
            params: 参数元组
        
        Returns:
            Dict: 记录字典，如果没有结果则返回 None
        """
        cursor = None
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(sql, params or ())
            result = cursor.fetchone()
            return result
            
        except Error as e:
            logger.error(f"查询失败: {sql}, 错误: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
    
    def fetch_all(self, sql: str, params: Tuple = None) -> List[Dict]:
        """
        查询多条记录
        
        Args:
            sql: 查询 SQL
            params: 参数元组
        
        Returns:
            List[Dict]: 记录列表
        """
        cursor = None
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(sql, params or ())
            results = cursor.fetchall()
            return results or []
            
        except Error as e:
            logger.error(f"查询失败: {sql}, 错误: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
    
    def fetch_page(
        self, 
        sql: str, 
        params: Tuple = None, 
        page: int = 1, 
        page_size: int = 20
    ) -> Tuple[List[Dict], int]:
        """
        分页查询
        
        Args:
            sql: 基础查询 SQL（不包含 LIMIT）
            params: 参数元组
            page: 页码（从 1 开始）
            page_size: 每页大小
        
        Returns:
            Tuple[List[Dict], int]: (记录列表, 总记录数)
        """
        # 查询总记录数
        count_sql = f"SELECT COUNT(*) as total FROM ({sql}) as count_query"
        total_result = self.fetch_one(count_sql, params)
        total_count = total_result['total'] if total_result else 0
        
        # 分页查询
        offset = (page - 1) * page_size
        paginated_sql = f"{sql} LIMIT {page_size} OFFSET {offset}"
        records = self.fetch_all(paginated_sql, params)
        
        return records, total_count
    
    def begin_transaction(self):
        """开始事务"""
        self.connection.start_transaction()
    
    def commit(self):
        """提交事务"""
        self.connection.commit()
    
    def rollback(self):
        """回滚事务"""
        self.connection.rollback()

class MilvusClient:
    """
    Milvus 向量数据库客户端封装
    
    提供向量数据库的常用操作：
    - 创建/删除集合
    - 插入/更新/删除向量数据
    - 向量相似度搜索
    - 索引管理
    
    使用示例:
        milvus_client = MilvusClient("default")
        
        # 创建集合
        milvus_client.create_collection("my_collection", schema_config)
        
        # 插入向量数据
        vector_id = milvus_client.insert("my_collection", {
            "id": 1,
            "vector": [0.1, 0.2, 0.3, ...],
            "metadata": {"title": "文档标题"}
        })
        
        # 向量搜索
        results = milvus_client.search(
            "my_collection",
            query_vectors=[[0.1, 0.2, ...]],
            vector_field="vector",
            top_k=10
        )
    """
    
    def __init__(self, connection_alias: str = "default"):
        """
        初始化 Milvus 客户端
        
        Args:
            connection_alias: 连接别名
        """
        if not MILVUS_AVAILABLE:
            raise ImportError("PyMilvus 未安装，无法使用 Milvus 功能")
        
        self.connection_alias = connection_alias
        
        # 检查连接是否有效
        try:
            connections.get_connection(connection_alias)
            logger.info("MilvusClient 初始化完成")
        except Exception as e:
            logger.error(f"Milvus 连接无效: {e}")
            raise
    
    def create_collection(self, collection_name: str, schema_config: Dict) -> bool:
        """
        创建集合
        
        Args:
            collection_name: 集合名称
            schema_config: 集合结构配置
                格式示例: {
                    "fields": [
                        {"name": "id", "type": "INT64", "is_primary": True, "auto_id": False},
                        {"name": "vector", "type": "FLOAT_VECTOR", "dim": 768},
                        {"name": "metadata", "type": "JSON"}
                    ],
                    "indexes": [
                        {"field": "vector", "type": "IVF_FLAT", "metric": "L2", "params": {"nlist": 1024}}
                    ]
                }
        
        Returns:
            bool: 创建是否成功
        """
        try:
            # 检查集合是否已存在
            if utility.has_collection(collection_name, using=self.connection_alias):
                logger.warning(f"集合 {collection_name} 已存在")
                return True
            
            # 构建字段定义
            fields = []
            for field_config in schema_config.get("fields", []):
                field_name = field_config["name"]
                field_type = getattr(DataType, field_config["type"])
                
                # 处理向量字段的维度
                if field_config["type"] == "FLOAT_VECTOR":
                    field = FieldSchema(
                        name=field_name,
                        dtype=field_type,
                        dim=field_config["dim"],
                        description=field_config.get("comment", "")
                    )
                else:
                    field = FieldSchema(
                        name=field_name,
                        dtype=field_type,
                        is_primary=field_config.get("is_primary", False),
                        auto_id=field_config.get("auto_id", False),
                        description=field_config.get("comment", "")
                    )
                
                fields.append(field)
            
            # 创建集合架构
            schema = CollectionSchema(
                fields=fields,
                description=f"Collection for {collection_name}"
            )
            
            # 创建集合
            collection = Collection(
                name=collection_name,
                schema=schema,
                using=self.connection_alias
            )
            
            # 创建索引
            for index_config in schema_config.get("indexes", []):
                field_name = index_config["field"]
                index_type = index_config["type"]
                metric_type = index_config.get("metric", "L2")
                index_params = index_config.get("params", {})
                
                collection.create_index(
                    field_name=field_name,
                    index_params={
                        "index_type": index_type,
                        "metric_type": metric_type,
                        "params": index_params
                    }
                )
            
            # 加载集合
            collection.load()
            
            logger.info(f"集合 {collection_name} 创建成功")
            return True
            
        except Exception as e:
            logger.error(f"创建集合失败: {e}")
            return False
    
    def drop_collection(self, collection_name: str) -> bool:
        """
        删除集合
        
        Args:
            collection_name: 集合名称
        
        Returns:
            bool: 删除是否成功
        """
        try:
            if utility.has_collection(collection_name, using=self.connection_alias):
                utility.drop_collection(collection_name, using=self.connection_alias)
                logger.info(f"集合 {collection_name} 删除成功")
                return True
            else:
                logger.warning(f"集合 {collection_name} 不存在")
                return True
                
        except Exception as e:
            logger.error(f"删除集合失败: {e}")
            return False
    
    def insert(self, collection_name: str, data: Dict) -> int:
        """
        插入单条向量数据
        
        Args:
            collection_name: 集合名称
            data: 数据字典
                示例: {
                    "id": 123,
                    "vector": [0.1, 0.2, 0.3, ...],
                    "metadata": {"title": "文档标题"}
                }
        
        Returns:
            int: 插入记录的 ID
        """
        try:
            collection = Collection(collection_name, using=self.connection_alias)
            
            # 将字典转换为列表格式
            entities = []
            for field in collection.schema.fields:
                field_name = field.name
                if field_name in data:
                    entities.append([data[field_name]])
                else:
                    # 如果是自增字段，可以跳过
                    if field.auto_id:
                        continue
                    else:
                        raise ValueError(f"缺少必需字段: {field_name}")
            
            # 插入数据
            insert_result = collection.insert(entities)
            
            # 获取插入的 ID
            if hasattr(insert_result, 'primary_keys') and insert_result.primary_keys:
                inserted_id = insert_result.primary_keys[0]
            else:
                # 如果没有返回 ID，使用提供的 ID
                inserted_id = data.get("id")
            
            logger.info(f"向量数据插入成功，ID: {inserted_id}")
            return inserted_id
            
        except Exception as e:
            logger.error(f"向量数据插入失败: {e}")
            raise
    
    def batch_insert(self, collection_name: str, data_list: List[Dict]) -> List[int]:
        """
        批量插入向量数据
        
        Args:
            collection_name: 集合名称
            data_list: 数据列表
        
        Returns:
            List[int]: 插入记录的 ID 列表
        """
        try:
            collection = Collection(collection_name, using=self.connection_alias)
            
            # 按字段组织数据
            entities = {}
            for field in collection.schema.fields:
                field_name = field.name
                if not field.auto_id:  # 跳过自增字段
                    entities[field_name] = []
            
            # 填充数据
            for data in data_list:
                for field_name in entities.keys():
                    if field_name in data:
                        entities[field_name].append(data[field_name])
                    else:
                        raise ValueError(f"记录缺少必需字段: {field_name}")
            
            # 转换为列表格式
            entity_list = list(entities.values())
            
            # 批量插入
            insert_result = collection.insert(entity_list)
            
            # 获取插入的 ID 列表
            if hasattr(insert_result, 'primary_keys'):
                inserted_ids = insert_result.primary_keys
            else:
                inserted_ids = [data.get("id") for data in data_list]
            
            logger.info(f"批量插入完成，插入 {len(inserted_ids)} 条记录")
            return inserted_ids
            
        except Exception as e:
            logger.error(f"批量插入失败: {e}")
            raise
    
    def search(
        self, 
        collection_name: str, 
        query_vectors: List[List[float]],
        vector_field: str = "vector",
        top_k: int = 10,
        filters: str = None,
        output_fields: List[str] = None
    ) -> List[Dict]:
        """
        向量相似度搜索
        
        Args:
            collection_name: 集合名称
            query_vectors: 查询向量列表
            vector_field: 向量字段名
            top_k: 返回最相似的 K 个结果
            filters: 过滤条件（Milvus 表达式）
            output_fields: 输出字段列表
        
        Returns:
            List[Dict]: 搜索结果列表
        """
        try:
            collection = Collection(collection_name, using=self.connection_alias)
            
            # 设置搜索参数
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10}
            }
            
            # 执行搜索
            search_results = collection.search(
                data=query_vectors,
                anns_field=vector_field,
                param=search_params,
                limit=top_k,
                expr=filters,
                output_fields=output_fields or ["*"]
            )
            
            # 处理搜索结果
            processed_results = []
            for hits in search_results:
                for hit in hits:
                    result = {
                        "id": hit.id,
                        "distance": hit.distance,
                        "entity": hit.entity
                    }
                    processed_results.append(result)
            
            logger.info(f"向量搜索完成，找到 {len(processed_results)} 条结果")
            return processed_results
            
        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            return []
    
    def get_by_id(self, collection_name: str, record_id: int) -> Optional[Dict]:
        """
        根据 ID 获取记录
        
        Args:
            collection_name: 集合名称
            record_id: 记录 ID
        
        Returns:
            Dict: 记录数据，如果不存在则返回 None
        """
        try:
            collection = Collection(collection_name, using=self.connection_alias)
            
            # 查询记录
            query_result = collection.query(
                expr=f"id == {record_id}",
                output_fields=["*"]
            )
            
            if query_result:
                return query_result[0]
            else:
                return None
                
        except Exception as e:
            logger.error(f"获取记录失败: {e}")
            return None
    
    def update(self, collection_name: str, record_id: int, updates: Dict) -> bool:
        """
        更新记录（Milvus 2.3+ 支持）
        
        Args:
            collection_name: 集合名称
            record_id: 记录 ID
            updates: 更新数据
        
        Returns:
            bool: 更新是否成功
        """
        try:
            # 注意：Milvus 的更新操作比较复杂，通常建议删除后重新插入
            logger.warning("Milvus 更新操作较为复杂，建议使用删除+插入的方式")
            return False
            
        except Exception as e:
            logger.error(f"更新记录失败: {e}")
            return False
    
    def delete(self, collection_name: str, record_id: int) -> bool:
        """
        删除记录
        
        Args:
            collection_name: 集合名称
            record_id: 记录 ID
        
        Returns:
            bool: 删除是否成功
        """
        try:
            collection = Collection(collection_name, using=self.connection_alias)
            
            # 删除记录
            delete_result = collection.delete(expr=f"id == {record_id}")
            
            success = delete_result.delete_count > 0
            if success:
                logger.info(f"记录删除成功，ID: {record_id}")
            else:
                logger.warning(f"记录不存在或删除失败，ID: {record_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"删除记录失败: {e}")
            return False

class ObjectStoreClient:
    """
    本地对象存储客户端
    
    提供文件存储的基本操作：
    - 上传/下载文件
    - 创建/删除目录
    - 列出文件
    - 文件管理
    
    使用示例:
        store_client = ObjectStoreClient(Path("./data/object_store"))
        
        # 上传文件
        file_path = store_client.upload_file(
            "docs/papers", 
            "paper1.pdf", 
            pdf_content,
            "research"
        )
        
        # 下载文件
        content = store_client.download_file(file_path)
        
        # 列出文件
        files = store_client.list_files("docs/papers")
    """
    
    def __init__(self, base_path: Path):
        """
        初始化对象存储客户端
        
        Args:
            base_path: 存储根目录
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ObjectStoreClient 初始化完成，根目录: {self.base_path}")
    
    def create_directory(self, dir_path: str) -> bool:
        """
        创建目录
        
        Args:
            dir_path: 目录路径（相对于根目录）
        
        Returns:
            bool: 创建是否成功
        """
        try:
            full_path = self.base_path / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"目录创建成功: {full_path}")
            return True
            
        except Exception as e:
            logger.error(f"目录创建失败: {e}")
            return False
    
    def upload_file(
        self, 
        dir_path: str, 
        file_name: str, 
        file_content: bytes, 
        subfolder: str = None
    ) -> str:
        """
        上传文件
        
        Args:
            dir_path: 目录路径
            file_name: 文件名
            file_content: 文件内容（二进制）
            subfolder: 子文件夹名称
        
        Returns:
            str: 文件存储路径
        """
        try:
            # 构建完整路径
            if subfolder:
                full_dir = self.base_path / dir_path / subfolder
            else:
                full_dir = self.base_path / dir_path
            
            # 确保目录存在
            full_dir.mkdir(parents=True, exist_ok=True)
            
            # 写入文件
            file_path = full_dir / file_name
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            # 返回相对路径
            relative_path = str(file_path.relative_to(self.base_path))
            logger.info(f"文件上传成功: {relative_path}")
            return relative_path
            
        except Exception as e:
            logger.error(f"文件上传失败: {e}")
            raise
    
    def download_file(self, file_path: str) -> bytes:
        """
        下载文件
        
        Args:
            file_path: 文件路径（相对于根目录）
        
        Returns:
            bytes: 文件内容
        """
        try:
            full_path = self.base_path / file_path
            
            if not full_path.exists():
                raise FileNotFoundError(f"文件不存在: {file_path}")
            
            with open(full_path, 'rb') as f:
                content = f.read()
            
            logger.info(f"文件下载成功: {file_path}")
            return content
            
        except Exception as e:
            logger.error(f"文件下载失败: {e}")
            raise
    
    def list_files(self, dir_path: str = "") -> List[str]:
        """
        列出目录中的所有文件
        
        Args:
            dir_path: 目录路径（相对于根目录），空字符串表示根目录
        
        Returns:
            List[str]: 文件路径列表（相对于根目录）
        """
        try:
            if dir_path:
                full_dir = self.base_path / dir_path
            else:
                full_dir = self.base_path
            
            if not full_dir.exists():
                return []
            
            files = []
            for file_path in full_dir.rglob('*'):
                if file_path.is_file():
                    relative_path = str(file_path.relative_to(self.base_path))
                    files.append(relative_path)
            
            logger.info(f"列出文件完成，共 {len(files)} 个文件")
            return files
            
        except Exception as e:
            logger.error(f"列出文件失败: {e}")
            return []
    
    def delete_file(self, file_path: str) -> bool:
        """
        删除文件
        
        Args:
            file_path: 文件路径（相对于根目录）
        
        Returns:
            bool: 删除是否成功
        """
        try:
            full_path = self.base_path / file_path
            
            if full_path.exists() and full_path.is_file():
                full_path.unlink()
                logger.info(f"文件删除成功: {file_path}")
                return True
            else:
                logger.warning(f"文件不存在: {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"文件删除失败: {e}")
            return False
    
    def delete_directory(self, dir_path: str) -> bool:
        """
        删除目录及其所有内容
        
        Args:
            dir_path: 目录路径（相对于根目录）
        
        Returns:
            bool: 删除是否成功
        """
        try:
            full_path = self.base_path / dir_path
            
            if full_path.exists() and full_path.is_dir():
                shutil.rmtree(full_path)
                logger.info(f"目录删除成功: {dir_path}")
                return True
            else:
                logger.warning(f"目录不存在: {dir_path}")
                return False
                
        except Exception as e:
            logger.error(f"目录删除失败: {e}")
            return False
    
    def get_file_info(self, file_path: str) -> Optional[Dict]:
        """
        获取文件信息
        
        Args:
            file_path: 文件路径（相对于根目录）
        
        Returns:
            Dict: 文件信息字典，包含大小、修改时间等
        """
        try:
            full_path = self.base_path / file_path
            
            if not full_path.exists():
                return None
            
            stat = full_path.stat()
            
            return {
                "path": file_path,
                "size": stat.st_size,
                "created_time": stat.st_ctime,
                "modified_time": stat.st_mtime,
                "is_file": full_path.is_file(),
                "is_directory": full_path.is_dir()
            }
            
        except Exception as e:
            logger.error(f"获取文件信息失败: {e}")
            return None