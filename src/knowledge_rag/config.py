"""
KnowledgeRAG 配置管理模块
作者: XYZ-Algorithm-Team
用途: 集中管理配置参数，读取环境变量，提供配置单例
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

@dataclass
class DatabaseSettings:
    """数据库配置"""
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
        """从环境变量创建数据库配置"""
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
    """Milvus配置"""
    host: str = "127.0.0.1"
    port: int = 19530
    collection_name: str = "rag_embeddings_v1"
    alias: str = "default"
    
    @classmethod
    def from_env(cls) -> 'MilvusSettings':
        """从环境变量创建Milvus配置"""
        return cls(
            host=os.getenv('MILVUS_HOST', '127.0.0.1'),
            port=int(os.getenv('MILVUS_PORT', '19530')),
            collection_name=os.getenv('MILVUS_COLLECTION', 'rag_embeddings_v1'),
            alias=os.getenv('MILVUS_ALIAS', 'default')
        )

@dataclass
class ObjectStoreSettings:
    """对象存储配置"""
    type: str = "local"  # 目前只支持local
    base_path: str = "./data/local_object_store"
    experiments_dir: str = "experiments"
    
    # 本地对象存储配置
    auto_create_dirs: bool = True
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    
    @classmethod
    def from_env(cls) -> 'ObjectStoreSettings':
        """从环境变量创建对象存储配置"""
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
class TokenBudgetSettings:
    """Token预算配置"""
    max_context_tokens: int = 2048
    chunk_max_tokens: int = 350
    top_k_raw: int = 20
    top_m_rerank: int = 5
    compression_ratio: float = 0.5
    
    @classmethod
    def from_env(cls) -> 'TokenBudgetSettings':
        """从环境变量创建Token预算配置"""
        return cls(
            max_context_tokens=int(os.getenv('MAX_CONTEXT_TOKENS', '2048')),
            chunk_max_tokens=int(os.getenv('CHUNK_MAX_TOKENS', '350')),
            top_k_raw=int(os.getenv('TOP_K_RAW', '20')),
            top_m_rerank=int(os.getenv('TOP_M_RERANK', '5')),
            compression_ratio=float(os.getenv('COMPRESSION_RATIO', '0.5'))
        )

@dataclass
class LoggingSettings:
    """日志配置"""
    level: str = "INFO"
    dir: str = "./logs"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    
    @classmethod
    def from_env(cls) -> 'LoggingSettings':
        """从环境变量创建日志配置"""
        return cls(
            level=os.getenv('LOG_LEVEL', 'INFO'),
            dir=os.getenv('LOG_DIR', './logs'),
            max_file_size=int(os.getenv('LOG_MAX_FILE_SIZE', '10485760')),
            backup_count=int(os.getenv('LOG_BACKUP_COUNT', '5'))
        )

@dataclass
class RetrievalSettings:
    """检索配置"""
    vector_similarity_threshold: float = 0.7
    enable_rerank: bool = True
    rerank_model: str = "bge-reranker-base"
    enable_compression: bool = True
    
    @classmethod
    def from_env(cls) -> 'RetrievalSettings':
        """从环境变量创建检索配置"""
        return cls(
            vector_similarity_threshold=float(os.getenv('VECTOR_SIMILARITY_THRESHOLD', '0.7')),
            enable_rerank=os.getenv('ENABLE_RERANK', 'True').lower() == 'true',
            rerank_model=os.getenv('RERANK_MODEL', 'bge-reranker-base'),
            enable_compression=os.getenv('ENABLE_COMPRESSION', 'True').lower() == 'true'
        )

@dataclass
class SecuritySettings:
    """安全配置"""
    enable_user_isolation: bool = True
    enable_audit_log: bool = True
    session_timeout: int = 3600  # 1小时
    max_query_rate: int = 100  # 每分钟最大查询次数
    
    @classmethod
    def from_env(cls) -> 'SecuritySettings':
        """从环境变量创建安全配置"""
        return cls(
            enable_user_isolation=os.getenv('ENABLE_USER_ISOLATION', 'True').lower() == 'true',
            enable_audit_log=os.getenv('ENABLE_AUDIT_LOG', 'True').lower() == 'true',
            session_timeout=int(os.getenv('SESSION_TIMEOUT', '3600')),
            max_query_rate=int(os.getenv('MAX_QUERY_RATE', '100'))
        )

@dataclass
class KnowledgeRAGSettings:
    """KnowledgeRAG主配置"""
    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    milvus: MilvusSettings = field(default_factory=MilvusSettings)
    object_store: ObjectStoreSettings = field(default_factory=ObjectStoreSettings)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    token_budget: TokenBudgetSettings = field(default_factory=TokenBudgetSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    security: SecuritySettings = field(default_factory=SecuritySettings)
    
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
            token_budget=TokenBudgetSettings.from_env(),
            logging=LoggingSettings.from_env(),
            retrieval=RetrievalSettings.from_env(),
            security=SecuritySettings.from_env(),
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
        
        # 验证Token预算配置
        if self.token_budget.max_context_tokens <= 0:
            errors.append("Max context tokens must be positive")
        
        if self.token_budget.top_k_raw <= 0:
            errors.append("Top K raw must be positive")
        
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
        
        # 验证日志配置
        log_dir = Path(self.logging.dir)
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create log directory: {e}")
        
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

def get_token_budget_settings() -> TokenBudgetSettings:
    """获取Token预算配置"""
    return get_settings().token_budget

def get_logging_settings() -> LoggingSettings:
    """获取日志配置"""
    return get_settings().logging

def get_retrieval_settings() -> RetrievalSettings:
    """获取检索配置"""
    return get_settings().retrieval

def get_security_settings() -> SecuritySettings:
    """获取安全配置"""
    return get_settings().security

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

if __name__ == "__main__":
    # 测试配置
    print("KnowledgeRAG Configuration Test")
    print("=" * 50)
    
    try:
        settings = get_settings()
        print(f"✓ Configuration loaded successfully")
        print(f"  - Environment: {settings.environment}")
        print(f"  - Debug: {settings.debug}")
        print(f"  - Database: {settings.database.host}:{settings.database.port}")
        print(f"  - Milvus: {settings.milvus.host}:{settings.milvus.port}")
        print(f"  - Object Store: {settings.object_store.type}")
        print(f"  - Embedding Model: {settings.embedding.model_name}")
        print(f"  - Log Level: {settings.logging.level}")
        
        print("\n" + "=" * 50)
        print("Full Configuration:")
        print_config()
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        import traceback
        traceback.print_exc() 