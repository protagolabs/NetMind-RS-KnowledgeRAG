"""
KnowledgeRAG 本地S3模拟工具
作者: XYZ-Algorithm-Team
用途: 模拟S3对象存储，提供本地文件系统的对象存储接口
"""

import os
import hashlib
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any, BinaryIO
from datetime import datetime
import json
import logging
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class ObjectMetadata:
    """对象元数据"""
    key: str
    size: int
    last_modified: datetime
    etag: str
    content_type: Optional[str] = None
    user_metadata: Optional[Dict[str, str]] = None

class S3LocalClient:
    """本地S3模拟客户端"""
    
    def __init__(self, base_path: str = "./data/object_store"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"初始化本地S3客户端，基础路径: {self.base_path}")
    
    def _get_object_path(self, user_id: int, doc_uuid: str, version_label: str, filename: str) -> Path:
        """获取对象的完整路径"""
        return self.base_path / f"user_{user_id}" / doc_uuid / f"v{version_label}" / filename
    
    def _ensure_user_directory(self, user_id: int) -> Path:
        """确保用户目录存在"""
        user_dir = self.base_path / f"user_{user_id}"
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir
    
    def _calculate_etag(self, file_path: Path) -> str:
        """计算文件的ETag (SHA256)"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _generate_uri(self, user_id: int, doc_uuid: str, version_label: str, filename: str) -> str:
        """生成对象URI"""
        return f"s3://local/user_{user_id}/{doc_uuid}/v{version_label}/{filename}"
    
    def put_object(self, user_id: int, doc_uuid: str, version_label: str, 
                   filename: str, file_stream: BinaryIO, 
                   content_type: Optional[str] = None,
                   metadata: Optional[Dict[str, str]] = None) -> str:
        """
        上传对象到本地存储
        
        Args:
            user_id: 用户ID
            doc_uuid: 文档UUID
            version_label: 版本标签
            filename: 文件名
            file_stream: 文件流
            content_type: 内容类型
            metadata: 用户元数据
            
        Returns:
            对象URI
        """
        try:
            # 确保用户目录存在
            self._ensure_user_directory(user_id)
            
            # 获取对象路径
            object_path = self._get_object_path(user_id, doc_uuid, version_label, filename)
            object_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 写入文件
            with open(object_path, "wb") as f:
                file_stream.seek(0)
                f.write(file_stream.read())
            
            # 保存元数据
            if content_type or metadata:
                metadata_path = object_path.with_suffix(object_path.suffix + ".meta")
                meta_info = {
                    "content_type": content_type,
                    "user_metadata": metadata or {},
                    "created_at": datetime.now().isoformat()
                }
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(meta_info, f, ensure_ascii=False, indent=2)
            
            # 生成URI
            uri = self._generate_uri(user_id, doc_uuid, version_label, filename)
            logger.info(f"对象上传成功: {uri}")
            return uri
            
        except Exception as e:
            logger.error(f"上传对象失败: {e}")
            raise
    
    def get_object(self, source_uri: str) -> bytes:
        """
        从本地存储获取对象
        
        Args:
            source_uri: 对象URI
            
        Returns:
            对象字节数据
        """
        try:
            # 解析URI
            if not source_uri.startswith("s3://local/"):
                raise ValueError(f"无效的URI格式: {source_uri}")
            
            # 提取路径
            relative_path = source_uri.replace("s3://local/", "")
            object_path = self.base_path / relative_path
            
            if not object_path.exists():
                raise FileNotFoundError(f"对象不存在: {source_uri}")
            
            # 读取文件
            with open(object_path, "rb") as f:
                data = f.read()
            
            logger.debug(f"读取对象成功: {source_uri}, 大小: {len(data)} bytes")
            return data
            
        except Exception as e:
            logger.error(f"读取对象失败: {e}")
            raise
    
    def get_object_metadata(self, source_uri: str) -> ObjectMetadata:
        """
        获取对象元数据
        
        Args:
            source_uri: 对象URI
            
        Returns:
            对象元数据
        """
        try:
            # 解析URI
            if not source_uri.startswith("s3://local/"):
                raise ValueError(f"无效的URI格式: {source_uri}")
            
            # 提取路径
            relative_path = source_uri.replace("s3://local/", "")
            object_path = self.base_path / relative_path
            
            if not object_path.exists():
                raise FileNotFoundError(f"对象不存在: {source_uri}")
            
            # 获取文件信息
            stat = object_path.stat()
            etag = self._calculate_etag(object_path)
            
            # 读取元数据文件
            metadata_path = object_path.with_suffix(object_path.suffix + ".meta")
            content_type = None
            user_metadata = None
            
            if metadata_path.exists():
                with open(metadata_path, "r", encoding="utf-8") as f:
                    meta_info = json.load(f)
                    content_type = meta_info.get("content_type")
                    user_metadata = meta_info.get("user_metadata")
            
            return ObjectMetadata(
                key=relative_path,
                size=stat.st_size,
                last_modified=datetime.fromtimestamp(stat.st_mtime),
                etag=etag,
                content_type=content_type,
                user_metadata=user_metadata
            )
            
        except Exception as e:
            logger.error(f"获取对象元数据失败: {e}")
            raise
    
    def list_user_docs(self, user_id: int) -> List[str]:
        """
        列出用户的所有文档URI
        
        Args:
            user_id: 用户ID
            
        Returns:
            URI列表
        """
        try:
            user_dir = self.base_path / f"user_{user_id}"
            
            if not user_dir.exists():
                return []
            
            uris = []
            for doc_dir in user_dir.iterdir():
                if doc_dir.is_dir():
                    for version_dir in doc_dir.iterdir():
                        if version_dir.is_dir():
                            for file_path in version_dir.iterdir():
                                if file_path.is_file() and not file_path.name.endswith(".meta"):
                                    uri = self._generate_uri(
                                        user_id, 
                                        doc_dir.name, 
                                        version_dir.name[1:],  # 去掉 'v' 前缀
                                        file_path.name
                                    )
                                    uris.append(uri)
            
            logger.info(f"用户 {user_id} 共有 {len(uris)} 个对象")
            return uris
            
        except Exception as e:
            logger.error(f"列出用户文档失败: {e}")
            raise
    
    def list_objects(self, user_id: int, doc_uuid: Optional[str] = None, 
                    version_label: Optional[str] = None) -> List[ObjectMetadata]:
        """
        列出对象及其元数据
        
        Args:
            user_id: 用户ID
            doc_uuid: 文档UUID（可选）
            version_label: 版本标签（可选）
            
        Returns:
            对象元数据列表
        """
        try:
            user_dir = self.base_path / f"user_{user_id}"
            
            if not user_dir.exists():
                return []
            
            objects = []
            
            # 确定搜索范围
            if doc_uuid:
                doc_dirs = [user_dir / doc_uuid] if (user_dir / doc_uuid).exists() else []
            else:
                doc_dirs = [d for d in user_dir.iterdir() if d.is_dir()]
            
            for doc_dir in doc_dirs:
                if version_label:
                    version_dirs = [doc_dir / f"v{version_label}"] if (doc_dir / f"v{version_label}").exists() else []
                else:
                    version_dirs = [d for d in doc_dir.iterdir() if d.is_dir()]
                
                for version_dir in version_dirs:
                    for file_path in version_dir.iterdir():
                        if file_path.is_file() and not file_path.name.endswith(".meta"):
                            uri = self._generate_uri(
                                user_id, 
                                doc_dir.name, 
                                version_dir.name[1:],  # 去掉 'v' 前缀
                                file_path.name
                            )
                            metadata = self.get_object_metadata(uri)
                            objects.append(metadata)
            
            return objects
            
        except Exception as e:
            logger.error(f"列出对象失败: {e}")
            raise
    
    def delete_object(self, source_uri: str) -> bool:
        """
        删除对象
        
        Args:
            source_uri: 对象URI
            
        Returns:
            删除成功返回True
        """
        try:
            # 解析URI
            if not source_uri.startswith("s3://local/"):
                raise ValueError(f"无效的URI格式: {source_uri}")
            
            # 提取路径
            relative_path = source_uri.replace("s3://local/", "")
            object_path = self.base_path / relative_path
            
            if not object_path.exists():
                logger.warning(f"对象不存在: {source_uri}")
                return False
            
            # 删除文件
            object_path.unlink()
            
            # 删除元数据文件
            metadata_path = object_path.with_suffix(object_path.suffix + ".meta")
            if metadata_path.exists():
                metadata_path.unlink()
            
            logger.info(f"对象删除成功: {source_uri}")
            return True
            
        except Exception as e:
            logger.error(f"删除对象失败: {e}")
            raise
    
    def generate_local_url(self, source_uri: str) -> str:
        """
        生成本地访问URL（调试用）
        
        Args:
            source_uri: 对象URI
            
        Returns:
            本地文件路径
        """
        try:
            # 解析URI
            if not source_uri.startswith("s3://local/"):
                raise ValueError(f"无效的URI格式: {source_uri}")
            
            # 提取路径
            relative_path = source_uri.replace("s3://local/", "")
            object_path = self.base_path / relative_path
            
            return str(object_path.absolute())
            
        except Exception as e:
            logger.error(f"生成本地URL失败: {e}")
            raise
    
    def check_user_permission(self, user_id: int, source_uri: str) -> bool:
        """
        检查用户权限
        
        Args:
            user_id: 用户ID
            source_uri: 对象URI
            
        Returns:
            有权限返回True
        """
        try:
            # 解析URI
            if not source_uri.startswith("s3://local/"):
                return False
            
            # 检查是否属于该用户
            expected_prefix = f"s3://local/user_{user_id}/"
            return source_uri.startswith(expected_prefix)
            
        except Exception as e:
            logger.error(f"检查权限失败: {e}")
            return False
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        获取存储统计信息
        
        Returns:
            统计信息字典
        """
        try:
            stats = {
                "total_users": 0,
                "total_documents": 0,
                "total_objects": 0,
                "total_size": 0,
                "users": {}
            }
            
            for user_dir in self.base_path.iterdir():
                if user_dir.is_dir() and user_dir.name.startswith("user_"):
                    user_id = user_dir.name.replace("user_", "")
                    stats["total_users"] += 1
                    
                    user_stats = {
                        "documents": 0,
                        "objects": 0,
                        "size": 0
                    }
                    
                    for doc_dir in user_dir.iterdir():
                        if doc_dir.is_dir():
                            user_stats["documents"] += 1
                            stats["total_documents"] += 1
                            
                            for version_dir in doc_dir.iterdir():
                                if version_dir.is_dir():
                                    for file_path in version_dir.iterdir():
                                        if file_path.is_file() and not file_path.name.endswith(".meta"):
                                            user_stats["objects"] += 1
                                            stats["total_objects"] += 1
                                            
                                            file_size = file_path.stat().st_size
                                            user_stats["size"] += file_size
                                            stats["total_size"] += file_size
                    
                    stats["users"][user_id] = user_stats
            
            return stats
            
        except Exception as e:
            logger.error(f"获取存储统计失败: {e}")
            raise

# 全局客户端实例
_s3_client = None

def get_s3_client(base_path: str = "./data/object_store") -> S3LocalClient:
    """
    获取全局S3客户端实例
    
    Args:
        base_path: 存储基础路径
        
    Returns:
        S3客户端实例
    """
    global _s3_client
    if _s3_client is None:
        _s3_client = S3LocalClient(base_path)
    return _s3_client 