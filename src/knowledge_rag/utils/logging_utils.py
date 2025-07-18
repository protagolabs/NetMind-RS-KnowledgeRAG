"""
KnowledgeRAG 统一日志工具
作者: XYZ-Algorithm-Team
用途: 提供统一的结构化日志记录功能
"""

import os
import json
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
from functools import wraps
from pathlib import Path
import traceback
from enum import Enum

class LogLevel(Enum):
    """日志级别"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class LogCategory(Enum):
    """日志类别"""
    SYSTEM = "system"
    QUERY = "query"
    RETRIEVAL = "retrieval"
    INGESTION = "ingestion"
    EMBEDDING = "embedding"
    DATABASE = "database"
    PERFORMANCE = "performance"
    SECURITY = "security"
    USER = "user"

@dataclass
class QueryLog:
    """查询日志数据类"""
    query_id: str
    user_id: int
    query_text: str
    query_type: str
    timestamp: float
    duration_ms: float
    top_k: int
    results_count: int
    token_count: int
    model_name: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@dataclass
class RetrievalLog:
    """检索日志数据类"""
    query_id: str
    user_id: int
    phase: str  # 'vector_search', 'rerank', 'filter', 'compress'
    timestamp: float
    duration_ms: float
    input_count: int
    output_count: int
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@dataclass
class IngestionLog:
    """摄取日志数据类"""
    job_id: str
    user_id: int
    document_id: int
    version_id: int
    phase: str  # 'upload', 'parse', 'chunk', 'embed', 'complete'
    timestamp: float
    duration_ms: float
    input_size: int
    output_count: int
    status: str  # 'success', 'error', 'warning'
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class PerformanceLog:
    """性能日志数据类"""
    operation: str
    timestamp: float
    duration_ms: float
    cpu_usage: Optional[float] = None
    memory_usage: Optional[float] = None
    disk_usage: Optional[float] = None
    network_io: Optional[Dict[str, float]] = None
    database_queries: Optional[int] = None
    cache_hits: Optional[int] = None
    cache_misses: Optional[int] = None

class StructuredLogger:
    """结构化日志记录器"""
    
    def __init__(self, name: str = "knowledge_rag", 
                 log_dir: str = "./logs",
                 log_level: str = "INFO",
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5):
        """
        初始化结构化日志记录器
        
        Args:
            name: 日志记录器名称
            log_dir: 日志目录
            log_level: 日志级别
            max_file_size: 单个日志文件最大大小
            backup_count: 备份文件数量
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建日志记录器
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # 清除现有的处理器
        self.logger.handlers.clear()
        
        # 设置日志格式
        self.formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 创建不同类型的日志处理器
        self._setup_handlers(max_file_size, backup_count)
        
        # 为所有处理器设置格式
        for handler in self.logger.handlers:
            handler.setFormatter(self.formatter)
    
    def _setup_handlers(self, max_file_size: int, backup_count: int):
        """设置日志处理器"""
        from logging.handlers import RotatingFileHandler
        
        # 主日志文件
        main_handler = RotatingFileHandler(
            self.log_dir / f"{self.name}.log",
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        
        # 错误日志文件
        error_handler = RotatingFileHandler(
            self.log_dir / f"{self.name}_error.log",
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        
        # 性能日志文件
        perf_handler = RotatingFileHandler(
            self.log_dir / f"{self.name}_performance.log",
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        
        # 控制台输出
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 添加处理器
        self.logger.addHandler(main_handler)
        self.logger.addHandler(error_handler)
        self.logger.addHandler(console_handler)
        
        # 单独保存性能日志处理器
        self.perf_handler = perf_handler
        self.perf_handler.setFormatter(self.formatter)
    
    def log_structured(self, level: LogLevel, category: LogCategory, 
                      message: str, data: Optional[Dict[str, Any]] = None):
        """
        记录结构化日志
        
        Args:
            level: 日志级别
            category: 日志类别
            message: 日志消息
            data: 附加数据
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level.value,
            "category": category.value,
            "message": message,
            "data": data or {}
        }
        
        log_message = json.dumps(log_entry, ensure_ascii=False)
        
        # 根据级别记录日志
        if level == LogLevel.DEBUG:
            self.logger.debug(log_message)
        elif level == LogLevel.INFO:
            self.logger.info(log_message)
        elif level == LogLevel.WARNING:
            self.logger.warning(log_message)
        elif level == LogLevel.ERROR:
            self.logger.error(log_message)
        elif level == LogLevel.CRITICAL:
            self.logger.critical(log_message)
    
    def log_query(self, query_log: QueryLog):
        """记录查询日志"""
        self.log_structured(
            LogLevel.INFO,
            LogCategory.QUERY,
            f"Query processed: {query_log.query_id}",
            asdict(query_log)
        )
    
    def log_retrieval(self, retrieval_log: RetrievalLog):
        """记录检索日志"""
        level = LogLevel.ERROR if retrieval_log.error else LogLevel.INFO
        self.log_structured(
            level,
            LogCategory.RETRIEVAL,
            f"Retrieval {retrieval_log.phase}: {retrieval_log.query_id}",
            asdict(retrieval_log)
        )
    
    def log_ingestion(self, ingestion_log: IngestionLog):
        """记录摄取日志"""
        level = LogLevel.ERROR if ingestion_log.error else LogLevel.INFO
        self.log_structured(
            level,
            LogCategory.INGESTION,
            f"Ingestion {ingestion_log.phase}: {ingestion_log.job_id}",
            asdict(ingestion_log)
        )
    
    def log_performance(self, perf_log: PerformanceLog):
        """记录性能日志"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": LogLevel.INFO.value,
            "category": LogCategory.PERFORMANCE.value,
            "message": f"Performance: {perf_log.operation}",
            "data": asdict(perf_log)
        }
        
        log_message = json.dumps(log_entry, ensure_ascii=False)
        
        # 同时记录到主日志和性能日志
        self.logger.info(log_message)
        self.perf_handler.emit(logging.LogRecord(
            name=self.name,
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=log_message,
            args=(),
            exc_info=None
        ))
    
    def log_error(self, category: LogCategory, message: str, 
                 error: Exception, context: Optional[Dict[str, Any]] = None):
        """记录错误日志"""
        error_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "context": context or {}
        }
        
        self.log_structured(
            LogLevel.ERROR,
            category,
            message,
            error_data
        )
    
    def log_security(self, event: str, user_id: Optional[int] = None,
                    ip_address: Optional[str] = None, 
                    metadata: Optional[Dict[str, Any]] = None):
        """记录安全日志"""
        security_data = {
            "event": event,
            "user_id": user_id,
            "ip_address": ip_address,
            "metadata": metadata or {}
        }
        
        self.log_structured(
            LogLevel.WARNING,
            LogCategory.SECURITY,
            f"Security event: {event}",
            security_data
        )

class PerformanceTimer:
    """性能计时器上下文管理器"""
    
    def __init__(self, logger: StructuredLogger, operation: str,
                 metadata: Optional[Dict[str, Any]] = None):
        self.logger = logger
        self.operation = operation
        self.metadata = metadata or {}
        self.start_time = None
        self.duration_ms = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration_ms = (time.time() - self.start_time) * 1000
        
        perf_log = PerformanceLog(
            operation=self.operation,
            timestamp=self.start_time,
            duration_ms=self.duration_ms,
            **self.metadata
        )
        
        self.logger.log_performance(perf_log)

def log_performance(operation: str):
    """性能日志装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with PerformanceTimer(get_logger(), f"{func.__name__}_{operation}"):
                return func(*args, **kwargs)
        return wrapper
    return decorator

def log_errors(category: LogCategory = LogCategory.SYSTEM):
    """错误日志装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger = get_logger()
                logger.log_error(
                    category,
                    f"Error in {func.__name__}",
                    e,
                    {"args": str(args), "kwargs": str(kwargs)}
                )
                raise
        return wrapper
    return decorator

class TokenCounter:
    """Token计数器"""
    
    def __init__(self):
        self.counts = {}
    
    def add_tokens(self, operation: str, model: str, 
                  input_tokens: int, output_tokens: int = 0):
        """添加token计数"""
        key = f"{operation}:{model}"
        if key not in self.counts:
            self.counts[key] = {"input": 0, "output": 0, "total": 0}
        
        self.counts[key]["input"] += input_tokens
        self.counts[key]["output"] += output_tokens
        self.counts[key]["total"] += input_tokens + output_tokens
    
    def get_summary(self) -> Dict[str, Any]:
        """获取token使用摘要"""
        total_input = sum(count["input"] for count in self.counts.values())
        total_output = sum(count["output"] for count in self.counts.values())
        
        return {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "by_operation": self.counts
        }
    
    def reset(self):
        """重置计数器"""
        self.counts.clear()

# 全局日志记录器实例
_logger = None
_token_counter = TokenCounter()

def get_logger() -> StructuredLogger:
    """获取全局日志记录器实例"""
    global _logger
    if _logger is None:
        log_config = {
            'name': 'knowledge_rag',
            'log_dir': os.getenv('LOG_DIR', './logs'),
            'log_level': os.getenv('LOG_LEVEL', 'INFO'),
            'max_file_size': int(os.getenv('LOG_MAX_FILE_SIZE', '10485760')),  # 10MB
            'backup_count': int(os.getenv('LOG_BACKUP_COUNT', '5'))
        }
        _logger = StructuredLogger(**log_config)
    return _logger

def get_token_counter() -> TokenCounter:
    """获取全局token计数器"""
    return _token_counter

def log_query_start(query_id: str, user_id: int, query_text: str, 
                   query_type: str = "semantic") -> float:
    """记录查询开始"""
    timestamp = time.time()
    
    logger = get_logger()
    logger.log_structured(
        LogLevel.INFO,
        LogCategory.QUERY,
        f"Query started: {query_id}",
        {
            "query_id": query_id,
            "user_id": user_id,
            "query_text": query_text[:100] + "..." if len(query_text) > 100 else query_text,
            "query_type": query_type,
            "timestamp": timestamp
        }
    )
    
    return timestamp

def log_query_end(query_id: str, user_id: int, query_text: str,
                 start_time: float, results_count: int, 
                 token_count: int, model_name: Optional[str] = None,
                 filters: Optional[Dict[str, Any]] = None,
                 error: Optional[str] = None):
    """记录查询结束"""
    duration_ms = (time.time() - start_time) * 1000
    
    query_log = QueryLog(
        query_id=query_id,
        user_id=user_id,
        query_text=query_text,
        query_type="semantic",
        timestamp=start_time,
        duration_ms=duration_ms,
        top_k=results_count,
        results_count=results_count,
        token_count=token_count,
        model_name=model_name,
        filters=filters,
        error=error
    )
    
    logger = get_logger()
    logger.log_query(query_log)

def log_retrieval_phase(query_id: str, user_id: int, phase: str,
                       start_time: float, input_count: int, output_count: int,
                       metadata: Optional[Dict[str, Any]] = None,
                       error: Optional[str] = None):
    """记录检索阶段"""
    duration_ms = (time.time() - start_time) * 1000
    
    retrieval_log = RetrievalLog(
        query_id=query_id,
        user_id=user_id,
        phase=phase,
        timestamp=start_time,
        duration_ms=duration_ms,
        input_count=input_count,
        output_count=output_count,
        metadata=metadata,
        error=error
    )
    
    logger = get_logger()
    logger.log_retrieval(retrieval_log)

def create_query_id() -> str:
    """创建查询ID"""
    import uuid
    return str(uuid.uuid4())[:8]

def create_job_id() -> str:
    """创建作业ID"""
    import uuid
    return str(uuid.uuid4())[:8] 