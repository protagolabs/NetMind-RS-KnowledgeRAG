"""
KnowledgeRAG 配置管理模块
========================

本模块提供了KnowledgeRAG系统的统一配置管理功能，支持多种配置来源
和环境适配，确保系统在不同环境下的灵活部署和运行。

核心功能：
1. 环境变量管理：自动加载.env文件和系统环境变量
2. 配置分类：按功能模块组织配置（数据库、向量库、存储等）
3. 配置验证：确保配置参数的有效性和完整性
4. 单例模式：提供全局统一的配置访问接口
5. 环境适配：支持开发、测试、生产等多种环境

配置分类：
- DatabaseSettings: MySQL数据库连接和配置
- MilvusSettings: Milvus向量数据库配置
- ObjectStoreSettings: 对象存储（本地文件系统）配置
- EmbeddingSettings: 向量嵌入模型配置
- KnowledgeRAGSettings: 系统主配置，整合所有子配置

技术特点：
- 基于dataclass的类型安全配置
- 支持环境变量覆盖默认值
- 自动配置验证和错误提示
- 敏感信息保护（密码隐藏）
- 灵活的配置重载机制

使用场景：
- 系统初始化和配置加载
- 多环境部署配置管理
- 数据库连接池配置
- 向量模型参数调优
- 存储路径和权限管理

环境变量支持：
- OPENAI_API_KEY: OpenAI API密钥
- MYSQL_*: MySQL数据库配置
- MILVUS_*: Milvus向量库配置
- EMBEDDING_*: 嵌入模型配置
- 其他系统级配置

作者: NetMind-RS-KnowledgeRAG Team
"""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass, field

# 自动加载 .env 文件
try:
    from dotenv import load_dotenv
    env_file = Path(__file__).parent.parent.parent / '.env'
    if env_file.exists():
        load_dotenv(env_file)
        logging.info(f"已加载配置文件: {env_file}")
except ImportError:
    logging.warning("python-dotenv 未安装，将使用系统环境变量")

logger = logging.getLogger(__name__)

MODEL_NAME = "gpt-5-2025-08-07"
MODEL_NAME_CUR = "gpt-4.1-2025-04-14"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MATCHING_MODEL = os.getenv("MATCHING_MODEL", MODEL_NAME_CUR)
INTENT_RECOGNITION_MODEL_NAME = os.getenv("INTENT_RECOGNITION_MODEL_NAME", MODEL_NAME_CUR)
QUERY_REWRITE_MODEL_NAME = os.getenv("QUERY_REWRITE_MODEL_NAME", MODEL_NAME_CUR)
GENERATION_MODEL_NAME = os.getenv("GENERATION_MODEL_NAME", MODEL_NAME_CUR)

@dataclass
class DatabaseSettings:
    """MySQL数据库配置管理类。
    
    管理MySQL数据库的连接参数和性能配置，支持从环境变量
    自动加载配置，确保不同环境下的灵活部署。
    
    Attributes:
        host: 数据库服务器地址，默认本地
        port: 数据库端口，MySQL默认3306
        user: 数据库用户名
        password: 数据库密码（生产环境建议使用环境变量）
        database: 目标数据库名称
        charset: 字符集，推荐utf8mb4支持emoji
        collation: 排序规则，影响文本比较和排序
        pool_size: 连接池大小，影响并发性能
        
    环境变量映射：
        MYSQL_HOST -> host
        MYSQL_PORT -> port
        MYSQL_USER -> user
        MYSQL_PASSWORD -> password
        MYSQL_DB -> database
        MYSQL_CHARSET -> charset
        MYSQL_COLLATION -> collation
        MYSQL_POOL_SIZE -> pool_size
        
    用途：
        - RAG文档和chunk的结构化数据存储
        - 全文搜索索引支持
        - 元数据和状态信息管理
        - 系统配置和日志存储
    """
    host: str = "127.0.0.1"
    port: int = 3306
    user: str = "root"
    password: str = "devpass"
    database: str = "knowledge_rag"
    charset: str = "utf8mb4"
    collation: str = "utf8mb4_unicode_ci"
    pool_size: int = 10
    
    @classmethod
    def from_env(cls) -> 'DatabaseSettings':
        """从环境变量创建数据库配置。
        
        优先使用环境变量，如果环境变量不存在则使用默认值。
        这种设计支持容器化部署和多环境配置。
        
        Returns:
            DatabaseSettings: 配置好的数据库设置实例
            
        环境变量优先级：
            环境变量 > 默认值
            
        示例：
            # 通过环境变量配置
            export MYSQL_HOST=db.example.com
            export MYSQL_PASSWORD=secure_password
            
            # 代码中使用
            db_config = DatabaseSettings.from_env()
        """
        return cls(
            host=os.getenv('MYSQL_HOST', '127.0.0.1'),
            port=int(os.getenv('MYSQL_PORT', '3306')),
            user=os.getenv('MYSQL_USER', 'root'),
            password=os.getenv('MYSQL_PASSWORD', 'devpass'),
            database=os.getenv('MYSQL_DB', 'knowledge_rag'),
            charset=os.getenv('MYSQL_CHARSET', 'utf8mb4'),
            collation=os.getenv('MYSQL_COLLATION', 'utf8mb4_unicode_ci'),
            pool_size=int(os.getenv('MYSQL_POOL_SIZE', '10'))
        )

@dataclass
class MilvusSettings:
    """Milvus向量数据库配置管理类。
    
    管理Milvus向量数据库的连接参数和集合配置，专门用于
    存储和检索文档的向量表示。
    
    Attributes:
        host: Milvus服务器地址
        port: Milvus端口，默认19530
        collection_name: 默认集合名称（实际会创建多个集合）
        alias: 连接别名，用于管理多个连接
        
    环境变量映射：
        MILVUS_HOST -> host
        MILVUS_PORT -> port
        MILVUS_COLLECTION -> collection_name
        MILVUS_ALIAS -> alias
        
    用途：
        - 存储文档和chunk的向量表示
        - 支持高效的语义相似度搜索
        - 向量索引管理和优化
        - 大规模向量数据的持久化存储
        
    集合设计：
        - documents_vectors: 文档级向量
        - documents_insights_vectors: 文档洞察向量
        - chunks_vectors_{source_id}: chunk级向量
        - chunks_insights_vectors_{source_id}: chunk洞察向量
    """
    host: str = "127.0.0.1"
    port: int = 19530
    collection_name: str = "rag_embeddings_v1"
    alias: str = "default"
    
    @classmethod
    def from_env(cls) -> 'MilvusSettings':
        """从环境变量创建Milvus配置。
        
        支持从环境变量加载Milvus连接参数，便于容器化部署
        和云环境配置。
        
        Returns:
            MilvusSettings: 配置好的Milvus设置实例
            
        部署建议：
            - 开发环境：使用本地Milvus实例
            - 生产环境：使用集群模式或云服务
            - 测试环境：可以使用内存模式
            
        示例：
            # 环境变量配置
            export MILVUS_HOST=milvus.example.com
            export MILVUS_PORT=19530
            
            # 代码使用
            milvus_config = MilvusSettings.from_env()
        """
        return cls(
            host=os.getenv('MILVUS_HOST', '127.0.0.1'),
            port=int(os.getenv('MILVUS_PORT', '19530')),
            collection_name=os.getenv('MILVUS_COLLECTION', 'rag_embeddings_v1'),
            alias=os.getenv('MILVUS_ALIAS', 'default')
        )

@dataclass
class ObjectStoreSettings:
    """对象存储配置管理类。
    
    管理文档、处理结果和实验数据的存储配置。当前版本支持
    本地文件系统存储，未来可扩展支持云存储服务。
    
    Attributes:
        type: 存储类型，当前只支持"local"
        base_path: 存储根目录路径
        experiments_dir: 实验数据子目录名称
        auto_create_dirs: 是否自动创建目录
        max_file_size: 最大文件大小限制（字节）
        
    目录结构：
        base_path/
        ├── experiments/          # 实验数据
        │   ├── paper_set_1/     # 数据集1
        │   └── paper_set_2/     # 数据集2
        ├── processed/           # 处理结果
        └── temp/               # 临时文件
        
    环境变量映射：
        OBJECT_STORE_TYPE -> type
        LOCAL_OBJECT_STORE_PATH -> base_path
        LOCAL_OBJECT_STORE_EXPERIMENTS_DIR -> experiments_dir
        LOCAL_OBJECT_STORE_AUTO_CREATE_DIRS -> auto_create_dirs
        LOCAL_OBJECT_STORE_MAX_FILE_SIZE -> max_file_size
        
    用途：
        - 原始文档存储
        - 处理结果缓存
        - 实验数据管理
        - 临时文件处理
        
    扩展性：
        未来可支持AWS S3、阿里云OSS等云存储服务
    """
    type: str = "local"  # 目前只支持local
    base_path: str = "./data/local_object_store"
    experiments_dir: str = "experiments"
    
    # 本地对象存储配置
    auto_create_dirs: bool = True
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    
    @classmethod
    def from_env(cls) -> 'ObjectStoreSettings':
        """从环境变量创建对象存储配置。
        
        支持通过环境变量灵活配置存储路径和参数，适应不同
        的部署环境和存储需求。
        
        Returns:
            ObjectStoreSettings: 配置好的对象存储设置实例
            
        配置建议：
            - 开发环境：使用项目内相对路径
            - 生产环境：使用绝对路径，确保权限正确
            - 容器环境：挂载外部存储卷
            
        示例：
            # 环境变量配置
            export LOCAL_OBJECT_STORE_PATH=/data/knowledge_rag
            export LOCAL_OBJECT_STORE_MAX_FILE_SIZE=209715200  # 200MB
            
            # 代码使用
            store_config = ObjectStoreSettings.from_env()
        """
        return cls(
            type=os.getenv('OBJECT_STORE_TYPE', 'local'),
            base_path=os.getenv('LOCAL_OBJECT_STORE_PATH', './data/local_object_store'),
            experiments_dir=os.getenv('LOCAL_OBJECT_STORE_EXPERIMENTS_DIR', 'experiments'),
            auto_create_dirs=os.getenv('LOCAL_OBJECT_STORE_AUTO_CREATE_DIRS', 'True').lower() == 'true',
            max_file_size=int(os.getenv('LOCAL_OBJECT_STORE_MAX_FILE_SIZE', '104857600'))  # 100MB
        )

@dataclass
class EmbeddingSettings:
    """嵌入模型配置"""
    model_name: str = "text-embedding-3-small"
    dimension: int = 1536  # 默认向量维度
    batch_size: int = 32
    max_seq_length: int = 512
    
    # 模型路径配置
    model_path: Optional[str] = None
    device: str = "auto"  # auto, cpu, cuda
    
    @classmethod
    def from_env(cls) -> 'EmbeddingSettings':
        """从环境变量创建嵌入模型配置"""
        return cls(
            model_name=os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small'),
            dimension=int(os.getenv('EMBEDDING_DIMENSION', '1536')),
            batch_size=int(os.getenv('EMBEDDING_BATCH_SIZE', '32')),
            max_seq_length=int(os.getenv('EMBEDDING_MAX_SEQ_LENGTH', '512')),
            model_path=os.getenv('EMBEDDING_MODEL_PATH'),
            device=os.getenv('EMBEDDING_DEVICE', 'auto')
        )

@dataclass
class KnowledgeRAGSettings:
    """KnowledgeRAG主配置"""
    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    milvus: MilvusSettings = field(default_factory=MilvusSettings)
    object_store: ObjectStoreSettings = field(default_factory=ObjectStoreSettings)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    
    # 环境配置
    environment: str = "development"  # development, staging, production
    debug: bool = True
    
    @classmethod
    def from_env(cls) -> 'KnowledgeRAGSettings':
        """从环境变量创建完整配置"""
        return cls(
            database=DatabaseSettings.from_env(),
            milvus=MilvusSettings.from_env(),
            object_store=ObjectStoreSettings.from_env(),
            embedding=EmbeddingSettings.from_env(),
            environment=os.getenv('ENVIRONMENT', 'development'),
            debug=os.getenv('DEBUG', 'True').lower() == 'true'
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        import dataclasses
        return dataclasses.asdict(self)
    
    def validate(self) -> bool:
        """验证配置有效性"""
        errors = []
        
        # 验证数据库配置
        if not self.database.host:
            errors.append("Database host is required")
        
        if not (1 <= self.database.port <= 65535):
            errors.append("Database port must be between 1 and 65535")
        
        # 验证Milvus配置
        if not self.milvus.host:
            errors.append("Milvus host is required")
        
        if not (1 <= self.milvus.port <= 65535):
            errors.append("Milvus port must be between 1 and 65535")
        
        # 验证嵌入模型配置
        if self.embedding.dimension <= 0:
            errors.append("Embedding dimension must be positive")
        
        # 验证对象存储配置
        if self.object_store.type != "local":
            errors.append("Only 'local' object store type is supported")
        
        # 验证本地对象存储路径
        base_path = Path(self.object_store.base_path)
        try:
            if self.object_store.auto_create_dirs:
                base_path.mkdir(parents=True, exist_ok=True)
                # 创建实验目录
                experiments_path = base_path / self.object_store.experiments_dir
                experiments_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create object store directory: {e}")
        
        # 验证文件大小配置
        if self.object_store.max_file_size <= 0:
            errors.append("Max file size must be positive")
        
        if errors:
            logger.error(f"Configuration validation failed: {errors}")
            return False
        
        return True
    
    def get_connection_string(self) -> str:
        """获取数据库连接字符串"""
        return (f"mysql://{self.database.user}:{self.database.password}@"
                f"{self.database.host}:{self.database.port}/{self.database.database}")
    
    def get_milvus_uri(self) -> str:
        """获取Milvus连接URI"""
        return f"http://{self.milvus.host}:{self.milvus.port}"
    
    def get_object_store_config(self) -> Dict[str, Any]:
        """获取对象存储配置"""
        if self.object_store.type == "local":
            return {
                "type": "local",
                "base_path": self.object_store.base_path,
                "experiments_dir": self.object_store.experiments_dir,
                "auto_create_dirs": self.object_store.auto_create_dirs,
                "max_file_size": self.object_store.max_file_size
            }
        else:
            raise ValueError(f"Unsupported object store type: {self.object_store.type}. Only 'local' is supported.")

# 全局配置实例
_settings: Optional[KnowledgeRAGSettings] = None

def get_settings() -> KnowledgeRAGSettings:
    """获取全局配置实例"""
    global _settings
    if _settings is None:
        _settings = KnowledgeRAGSettings.from_env()
        
        # 验证配置
        if not _settings.validate():
            raise ValueError("Invalid configuration")
        
        logger.info(f"Configuration loaded: environment={_settings.environment}, debug={_settings.debug}")
    
    return _settings

def reload_settings():
    """重新加载配置"""
    global _settings
    _settings = None
    return get_settings()

# 快捷访问函数
def get_db_settings() -> DatabaseSettings:
    """获取数据库配置"""
    return get_settings().database

def get_milvus_settings() -> MilvusSettings:
    """获取Milvus配置"""
    return get_settings().milvus

def get_object_store_settings() -> ObjectStoreSettings:
    """获取对象存储配置"""
    return get_settings().object_store

def get_embedding_settings() -> EmbeddingSettings:
    """获取嵌入模型配置"""
    return get_settings().embedding

def is_debug() -> bool:
    """是否为调试模式"""
    return get_settings().debug

def is_production() -> bool:
    """是否为生产环境"""
    return get_settings().environment == "production"

def print_config():
    """打印配置信息（隐藏敏感信息）"""
    settings = get_settings()
    config_dict = settings.to_dict()
    
    # 隐藏敏感信息
    if 'database' in config_dict:
        config_dict['database']['password'] = "***"
    
    import json
    print(json.dumps(config_dict, indent=2, ensure_ascii=False))

 