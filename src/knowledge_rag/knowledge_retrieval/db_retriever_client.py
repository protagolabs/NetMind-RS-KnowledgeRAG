"""
KnowledgeRAG Retrieval Client
=============================

作者: Bin Liang
日期: 2025-01-27
描述: KnowledgeRAG API 服务的客户端封装

主要功能:
1. 封装所有 KnowledgeRAG API 端点
2. 提供类型安全的请求和响应处理
3. 统一的错误处理和重试机制
4. 支持异步和同步调用

技术栈:
- httpx: 现代异步 HTTP 客户端
- Pydantic: 数据验证和序列化
- 类型提示: 完整的类型安全支持
"""

import asyncio
import json
from typing import List, Dict, Any, Optional, Union
from urllib.parse import urljoin

import httpx
from loguru import logger
from pydantic import BaseModel, Field
from pydantic import ValidationError as PydanticValidationError


# ===== 异常定义 =====

class KnowledgeRAGClientError(Exception):
    """KnowledgeRAG 客户端基础异常"""
    pass


class APIConnectionError(KnowledgeRAGClientError):
    """API 连接错误"""
    pass


class APIResponseError(KnowledgeRAGClientError):
    """API 响应错误"""
    def __init__(self, status_code: int, message: str, details: Optional[Dict] = None):
        self.status_code = status_code
        self.message = message
        self.details = details or {}
        super().__init__(f"API Error {status_code}: {message}")


class KnowledgeRAGValidationError(KnowledgeRAGClientError):
    """数据验证错误"""
    pass


# ===== 请求模型 =====

class UploadRequest(BaseModel):
    """文档上传请求模型"""
    markdown_documents: List[str] = Field(..., description="Markdown 文档内容列表")
    file_names: List[str] = Field(..., description="文件名列表")


class EmbeddingRequest(BaseModel):
    """嵌入请求模型"""
    doc_analysis_list: List[dict] = Field(..., description="文档分析结果列表")
    chunk_analysis_list: List[dict] = Field(..., description="chunk 分析结果列表")


class SaveToDBRequest(BaseModel):
    """保存到数据库请求模型"""
    doc_json_file_path: Optional[str] = Field(None, description="文档 JSON 文件路径")
    chunk_json_file_path: Optional[str] = Field(None, description="chunk JSON 文件路径")
    doc_embedding_list: Optional[List[dict]] = Field(None, description="文档嵌入列表")
    chunk_embedding_list: Optional[List[dict]] = Field(None, description="chunk 嵌入列表")


class DocRetrievalRequest(BaseModel):
    """文档检索请求模型"""
    query: str = Field(..., description="查询文本", min_length=1)


class DocRetrievalByDataSetRequest(BaseModel):
    """按数据集类型检索文档请求模型"""
    data_set_type: str = Field(..., description="数据集类型")
    query: str = Field(..., description="查询文本", min_length=1)


class ChunkRetrievalRequest(BaseModel):
    """chunk 检索请求模型"""
    query: str = Field(..., description="查询文本", min_length=1)
    doc_ids: List[str] = Field(..., description="文档 ID 列表")
    each_doc_chunk_number: int = Field(default=10, description="每个文档返回的 chunk 数量", ge=1, le=50)


class ChunkDecisionRequest(BaseModel):
    """chunk 决策请求模型"""
    query_text: str = Field(..., description="查询文本", min_length=1)
    chunks: List[dict] = Field(..., description="chunk 列表")


# ===== 响应模型 =====

class BaseResponse(BaseModel):
    """基础响应模型"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")


class HealthResponse(BaseModel):
    """健康检查响应模型"""
    message: str
    version: str
    status: str
    retriever_status: str


class DetailedHealthResponse(BaseModel):
    """详细健康检查响应模型"""
    status: str
    retriever: Dict[str, bool]
    config: Dict[str, Any]


class UploadResponse(BaseResponse):
    """文档上传响应模型"""
    doc_analysis_list: List[dict] = Field(..., description="文档分析结果")
    chunk_analysis_list: List[dict] = Field(..., description="chunk 分析结果")
    total_docs: int = Field(..., description="处理的文档总数")
    total_chunks: int = Field(..., description="处理的 chunk 总数")


class EmbeddingResponse(BaseResponse):
    """嵌入响应模型"""
    doc_embedding_list: List[dict] = Field(..., description="文档嵌入列表")
    chunk_embedding_list: List[dict] = Field(..., description="chunk 嵌入列表")


class SaveToDBResponse(BaseResponse):
    """保存到数据库响应模型"""
    pass


class RetrievalResponse(BaseResponse):
    """检索响应模型"""
    results: List[Any] = Field(..., description="检索结果")
    total_results: int = Field(..., description="结果总数")


# ===== 主要客户端类 =====

class KnowledgeRAGClient:
    """
    KnowledgeRAG API 客户端
    
    提供对 KnowledgeRAG API 服务的完整封装，支持：
    - 文档上传和处理
    - 嵌入生成
    - 数据库保存
    - 各种检索功能
    - 健康检查
    """
    
    def __init__(
        self, 
        base_url: str = "http://localhost:8955",
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        初始化客户端
        
        Args:
            base_url: API 服务器基础 URL
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
            retry_delay: 重试间隔（秒）
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # 创建异步 HTTP 客户端
        self._async_client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            follow_redirects=True
        )
        
        # 创建同步 HTTP 客户端
        self._sync_client = httpx.Client(
            timeout=httpx.Timeout(timeout),
            follow_redirects=True
        )
        
        logger.info(f"KnowledgeRAG 客户端初始化完成: {base_url}")
    
    def __del__(self):
        """清理资源"""
        try:
            if hasattr(self, '_sync_client'):
                self._sync_client.close()
        except:
            pass
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()
    
    async def close(self):
        """关闭异步客户端"""
        if hasattr(self, '_async_client'):
            await self._async_client.aclose()
    
    def _build_url(self, endpoint: str) -> str:
        """构建完整的 API URL"""
        return urljoin(self.base_url + '/', endpoint.lstrip('/'))
    
    async def _make_request_async(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict] = None,
        json_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        发送异步 HTTP 请求
        
        Args:
            method: HTTP 方法
            endpoint: API 端点
            data: 表单数据
            json_data: JSON 数据
            
        Returns:
            API 响应数据
            
        Raises:
            APIConnectionError: 连接错误
            APIResponseError: 响应错误
        """
        url = self._build_url(endpoint)
        
        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"发送请求: {method} {url} (尝试 {attempt + 1})")
                
                response = await self._async_client.request(
                    method=method,
                    url=url,
                    data=data,
                    json=json_data
                )
                
                # 检查响应状态
                if response.status_code == 200:
                    try:
                        return response.json()
                    except json.JSONDecodeError as e:
                        raise APIResponseError(
                            response.status_code, 
                            f"响应解析失败: {e}",
                            {"raw_response": response.text}
                        )
                else:
                    # 尝试解析错误响应
                    try:
                        error_data = response.json()
                        raise APIResponseError(
                            response.status_code,
                            error_data.get('message', 'Unknown error'),
                            error_data
                        )
                    except json.JSONDecodeError:
                        raise APIResponseError(
                            response.status_code,
                            f"HTTP {response.status_code}: {response.text}",
                            {"raw_response": response.text}
                        )
                        
            except httpx.RequestError as e:
                if attempt == self.max_retries:
                    raise APIConnectionError(f"连接失败: {e}")
                
                logger.warning(f"请求失败，{self.retry_delay}秒后重试: {e}")
                await asyncio.sleep(self.retry_delay)
                continue
            
            except APIResponseError:
                # API 错误不重试
                raise
        
        # 如果所有重试都失败，抛出连接错误
        raise APIConnectionError(f"所有重试都失败，无法连接到 {url}")
    
    def _make_request_sync(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict] = None,
        json_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        发送同步 HTTP 请求
        
        Args:
            method: HTTP 方法
            endpoint: API 端点
            data: 表单数据
            json_data: JSON 数据
            
        Returns:
            API 响应数据
            
        Raises:
            APIConnectionError: 连接错误
            APIResponseError: 响应错误
        """
        url = self._build_url(endpoint)
        
        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"发送同步请求: {method} {url} (尝试 {attempt + 1})")
                
                response = self._sync_client.request(
                    method=method,
                    url=url,
                    data=data,
                    json=json_data
                )
                
                # 检查响应状态
                if response.status_code == 200:
                    try:
                        return response.json()
                    except json.JSONDecodeError as e:
                        raise APIResponseError(
                            response.status_code, 
                            f"响应解析失败: {e}",
                            {"raw_response": response.text}
                        )
                else:
                    # 尝试解析错误响应
                    try:
                        error_data = response.json()
                        raise APIResponseError(
                            response.status_code,
                            error_data.get('message', 'Unknown error'),
                            error_data
                        )
                    except json.JSONDecodeError:
                        raise APIResponseError(
                            response.status_code,
                            f"HTTP {response.status_code}: {response.text}",
                            {"raw_response": response.text}
                        )
                        
            except httpx.RequestError as e:
                if attempt == self.max_retries:
                    raise APIConnectionError(f"连接失败: {e}")
                
                logger.warning(f"请求失败，{self.retry_delay}秒后重试: {e}")
                import time
                time.sleep(self.retry_delay)
                continue
            
            except APIResponseError:
                # API 错误不重试
                raise
        
        # 如果所有重试都失败，抛出连接错误
        raise APIConnectionError(f"所有重试都失败，无法连接到 {url}")
    
    # ===== 健康检查方法 =====
    
    async def health_check_async(self) -> HealthResponse:
        """异步健康检查"""
        try:
            response_data = await self._make_request_async("GET", "/")
            return HealthResponse(**response_data)
        except PydanticValidationError as e:
            raise KnowledgeRAGValidationError(f"健康检查响应验证失败: {e}")
    
    def health_check(self) -> HealthResponse:
        """同步健康检查"""
        try:
            response_data = self._make_request_sync("GET", "/")
            return HealthResponse(**response_data)
        except PydanticValidationError as e:
            raise KnowledgeRAGValidationError(f"健康检查响应验证失败: {e}")
    
    async def detailed_health_check_async(self) -> DetailedHealthResponse:
        """异步详细健康检查"""
        try:
            response_data = await self._make_request_async("GET", "/health")
            return DetailedHealthResponse(**response_data)
        except PydanticValidationError as e:
            raise KnowledgeRAGValidationError(f"详细健康检查响应验证失败: {e}")
    
    def detailed_health_check(self) -> DetailedHealthResponse:
        """同步详细健康检查"""
        try:
            response_data = self._make_request_sync("GET", "/health")
            return DetailedHealthResponse(**response_data)
        except PydanticValidationError as e:
            raise KnowledgeRAGValidationError(f"详细健康检查响应验证失败: {e}")
    
    # ===== 文档处理方法 =====
    
    async def upload_documents_async(
        self, 
        markdown_documents: List[str], 
        file_names: List[str]
    ) -> UploadResponse:
        """
        异步上传并处理 Markdown 文档
        
        Args:
            markdown_documents: Markdown 文档内容列表
            file_names: 对应的文件名列表
            
        Returns:
            文档上传响应
        """
        try:
            request = UploadRequest(
                markdown_documents=markdown_documents,
                file_names=file_names
            )
            response_data = await self._make_request_async(
                "POST", 
                "/upload", 
                json_data=request.dict()
            )
            return UploadResponse(**response_data)
        except PydanticValidationError as e:
            raise KnowledgeRAGValidationError(f"文档上传请求验证失败: {e}")
    
    def upload_documents(
        self, 
        markdown_documents: List[str], 
        file_names: List[str]
    ) -> UploadResponse:
        """
        同步上传并处理 Markdown 文档
        
        Args:
            markdown_documents: Markdown 文档内容列表
            file_names: 对应的文件名列表
            
        Returns:
            文档上传响应
        """
        try:
            request = UploadRequest(
                markdown_documents=markdown_documents,
                file_names=file_names
            )
            response_data = self._make_request_sync(
                "POST", 
                "/upload", 
                json_data=request.dict()
            )
            return UploadResponse(**response_data)
        except PydanticValidationError as e:
            raise KnowledgeRAGValidationError(f"文档上传请求验证失败: {e}")
    
    async def generate_embeddings_async(
        self, 
        doc_analysis_list: List[dict], 
        chunk_analysis_list: List[dict]
    ) -> EmbeddingResponse:
        """
        异步生成文档和 chunk 嵌入
        
        Args:
            doc_analysis_list: 文档分析结果列表
            chunk_analysis_list: chunk 分析结果列表
            
        Returns:
            嵌入生成响应
        """
        try:
            request = EmbeddingRequest(
                doc_analysis_list=doc_analysis_list,
                chunk_analysis_list=chunk_analysis_list
            )
            response_data = await self._make_request_async(
                "POST", 
                "/embedding", 
                json_data=request.dict()
            )
            return EmbeddingResponse(**response_data)
        except PydanticValidationError as e:
            raise KnowledgeRAGValidationError(f"嵌入生成请求验证失败: {e}")
    
    def generate_embeddings(
        self, 
        doc_analysis_list: List[dict], 
        chunk_analysis_list: List[dict]
    ) -> EmbeddingResponse:
        """
        同步生成文档和 chunk 嵌入
        
        Args:
            doc_analysis_list: 文档分析结果列表
            chunk_analysis_list: chunk 分析结果列表
            
        Returns:
            嵌入生成响应
        """
        try:
            request = EmbeddingRequest(
                doc_analysis_list=doc_analysis_list,
                chunk_analysis_list=chunk_analysis_list
            )
            response_data = self._make_request_sync(
                "POST", 
                "/embedding", 
                json_data=request.dict()
            )
            return EmbeddingResponse(**response_data)
        except PydanticValidationError as e:
            raise KnowledgeRAGValidationError(f"嵌入生成请求验证失败: {e}")
    
    async def save_to_database_async(
        self,
        doc_json_file_path: Optional[str] = None,
        chunk_json_file_path: Optional[str] = None,
        doc_embedding_list: Optional[List[dict]] = None,
        chunk_embedding_list: Optional[List[dict]] = None
    ) -> SaveToDBResponse:
        """
        异步保存数据到数据库
        
        Args:
            doc_json_file_path: 文档 JSON 文件路径
            chunk_json_file_path: chunk JSON 文件路径
            doc_embedding_list: 文档嵌入列表
            chunk_embedding_list: chunk 嵌入列表
            
        Returns:
            保存响应
        """
        try:
            request = SaveToDBRequest(
                doc_json_file_path=doc_json_file_path,
                chunk_json_file_path=chunk_json_file_path,
                doc_embedding_list=doc_embedding_list,
                chunk_embedding_list=chunk_embedding_list
            )
            response_data = await self._make_request_async(
                "POST", 
                "/save-to-db", 
                json_data=request.dict()
            )
            return SaveToDBResponse(**response_data)
        except PydanticValidationError as e:
            raise KnowledgeRAGValidationError(f"保存数据库请求验证失败: {e}")
    
    def save_to_database(
        self,
        doc_json_file_path: Optional[str] = None,
        chunk_json_file_path: Optional[str] = None,
        doc_embedding_list: Optional[List[dict]] = None,
        chunk_embedding_list: Optional[List[dict]] = None
    ) -> SaveToDBResponse:
        """
        同步保存数据到数据库
        
        Args:
            doc_json_file_path: 文档 JSON 文件路径
            chunk_json_file_path: chunk JSON 文件路径
            doc_embedding_list: 文档嵌入列表
            chunk_embedding_list: chunk 嵌入列表
            
        Returns:
            保存响应
        """
        try:
            request = SaveToDBRequest(
                doc_json_file_path=doc_json_file_path,
                chunk_json_file_path=chunk_json_file_path,
                doc_embedding_list=doc_embedding_list,
                chunk_embedding_list=chunk_embedding_list
            )
            response_data = self._make_request_sync(
                "POST", 
                "/save-to-db", 
                json_data=request.dict()
            )
            return SaveToDBResponse(**response_data)
        except PydanticValidationError as e:
            raise KnowledgeRAGValidationError(f"保存数据库请求验证失败: {e}")
    
    # ===== 检索方法 =====
    
    async def retrieve_documents_async(self, query: str) -> RetrievalResponse:
        """
        异步文档检索
        
        Args:
            query: 查询文本
            
        Returns:
            检索响应
        """
        try:
            request = DocRetrievalRequest(query=query)
            response_data = await self._make_request_async(
                "POST", 
                "/doc-retrieval", 
                json_data=request.model_dump()
            )
            return RetrievalResponse(**response_data)
        except PydanticValidationError as e:
            raise KnowledgeRAGValidationError(f"文档检索请求验证失败: {e}")
    
    def retrieve_documents(self, query: str) -> RetrievalResponse:
        """
        同步文档检索
        
        Args:
            query: 查询文本
            
        Returns:
            检索响应
        """
        try:
            request = DocRetrievalRequest(query=query)
            response_data = self._make_request_sync(
                "POST", 
                "/doc-retrieval", 
                json_data=request.model_dump()
            )
            return RetrievalResponse(**response_data)
        except PydanticValidationError as e:
            raise KnowledgeRAGValidationError(f"文档检索请求验证失败: {e}")
    
    async def retrieve_documents_by_dataset_async(
        self, 
        data_set_type: str, 
        query: str
    ) -> RetrievalResponse:
        """
        异步按数据集类型检索文档
        
        Args:
            data_set_type: 数据集类型
            query: 查询文本
            
        Returns:
            检索响应
        """
        try:
            request = DocRetrievalByDataSetRequest(
                data_set_type=data_set_type,
                query=query
            )
            response_data = await self._make_request_async(
                "POST", 
                "/doc-retrieval-by-dataset", 
                json_data=request.model_dump()
            )
            return RetrievalResponse(**response_data)
        except PydanticValidationError as e:
            raise KnowledgeRAGValidationError(f"数据集文档检索请求验证失败: {e}")
    
    def retrieve_documents_by_dataset(
        self, 
        data_set_type: str, 
        query: str
    ) -> RetrievalResponse:
        """
        同步按数据集类型检索文档
        
        Args:
            data_set_type: 数据集类型
            query: 查询文本
            
        Returns:
            检索响应
        """
        try:
            request = DocRetrievalByDataSetRequest(
                data_set_type=data_set_type,
                query=query
            )
            response_data = self._make_request_sync(
                "POST", 
                "/doc-retrieval-by-dataset", 
                json_data=request.dict()
            )
            return RetrievalResponse(**response_data)
        except PydanticValidationError as e:
            raise KnowledgeRAGValidationError(f"数据集文档检索请求验证失败: {e}")
    
    async def retrieve_chunks_async(
        self, 
        query: str, 
        doc_ids: List[str], 
        each_doc_chunk_number: int = 10
    ) -> RetrievalResponse:
        """
        异步 chunk 检索
        
        Args:
            query: 查询文本
            doc_ids: 文档 ID 列表
            each_doc_chunk_number: 每个文档返回的 chunk 数量
            
        Returns:
            检索响应
        """
        try:
            request = ChunkRetrievalRequest(
                query=query,
                doc_ids=doc_ids,
                each_doc_chunk_number=each_doc_chunk_number
            )
            response_data = await self._make_request_async(
                "POST", 
                "/chunk-retrieval", 
                json_data=request.dict()
            )
            return RetrievalResponse(**response_data)
        except PydanticValidationError as e:
            raise KnowledgeRAGValidationError(f"Chunk 检索请求验证失败: {e}")
    
    def retrieve_chunks(
        self, 
        query: str, 
        doc_ids: List[str], 
        each_doc_chunk_number: int = 10
    ) -> RetrievalResponse:
        """
        同步 chunk 检索
        
        Args:
            query: 查询文本
            doc_ids: 文档 ID 列表
            each_doc_chunk_number: 每个文档返回的 chunk 数量
            
        Returns:
            检索响应
        """
        try:
            request = ChunkRetrievalRequest(
                query=query,
                doc_ids=doc_ids,
                each_doc_chunk_number=each_doc_chunk_number
            )
            response_data = self._make_request_sync(
                "POST", 
                "/chunk-retrieval", 
                json_data=request.dict()
            )
            return RetrievalResponse(**response_data)
        except PydanticValidationError as e:
            raise KnowledgeRAGValidationError(f"Chunk 检索请求验证失败: {e}")
    
    async def make_chunk_decision_async(
        self, 
        query_text: str, 
        chunks: List[dict]
    ) -> RetrievalResponse:
        """
        异步 chunk 决策匹配
        
        Args:
            query_text: 查询文本
            chunks: chunk 列表
            
        Returns:
            检索响应
        """
        try:
            request = ChunkDecisionRequest(
                query_text=query_text,
                chunks=chunks
            )
            response_data = await self._make_request_async(
                "POST", 
                "/chunk-decision", 
                json_data=request.dict()
            )
            return RetrievalResponse(**response_data)
        except PydanticValidationError as e:
            raise KnowledgeRAGValidationError(f"Chunk 决策请求验证失败: {e}")
    
    def make_chunk_decision(
        self, 
        query_text: str, 
        chunks: List[dict]
    ) -> RetrievalResponse:
        """
        同步 chunk 决策匹配
        
        Args:
            query_text: 查询文本
            chunks: chunk 列表
            
        Returns:
            检索响应
        """
        try:
            request = ChunkDecisionRequest(
                query_text=query_text,
                chunks=chunks
            )
            response_data = self._make_request_sync(
                "POST", 
                "/chunk-decision", 
                json_data=request.dict()
            )
            return RetrievalResponse(**response_data)
        except PydanticValidationError as e:
            raise KnowledgeRAGValidationError(f"Chunk 决策请求验证失败: {e}")
    
    # ===== 便利方法 =====
    
    async def full_document_pipeline_async(
        self,
        markdown_documents: List[str],
        file_names: List[str],
        save_to_db: bool = True
    ) -> Dict[str, Union[UploadResponse, EmbeddingResponse, SaveToDBResponse]]:
        """
        异步完整文档处理管道
        
        执行完整的文档处理流程：上传 -> 嵌入 -> 保存
        
        Args:
            markdown_documents: Markdown 文档内容列表
            file_names: 文件名列表
            save_to_db: 是否保存到数据库
            
        Returns:
            包含所有步骤结果的字典
        """
        results: Dict[str, Union[UploadResponse, EmbeddingResponse, SaveToDBResponse]] = {}
        
        # 1. 上传文档
        logger.info("开始文档上传...")
        upload_response = await self.upload_documents_async(markdown_documents, file_names)
        results['upload'] = upload_response
        
        # 2. 生成嵌入
        logger.info("开始生成嵌入...")
        embedding_response = await self.generate_embeddings_async(
            upload_response.doc_analysis_list,
            upload_response.chunk_analysis_list
        )
        results['embedding'] = embedding_response
        
        # 3. 保存到数据库（可选）
        if save_to_db:
            logger.info("开始保存到数据库...")
            save_response = await self.save_to_database_async(
                doc_embedding_list=embedding_response.doc_embedding_list,
                chunk_embedding_list=embedding_response.chunk_embedding_list
            )
            results['save'] = save_response
        
        logger.info("完整文档处理管道执行完成")
        return results
    
    def full_document_pipeline(
        self,
        markdown_documents: List[str],
        file_names: List[str],
        save_to_db: bool = True
    ) -> Dict[str, Union[UploadResponse, EmbeddingResponse, SaveToDBResponse]]:
        """
        同步完整文档处理管道
        
        执行完整的文档处理流程：上传 -> 嵌入 -> 保存
        
        Args:
            markdown_documents: Markdown 文档内容列表
            file_names: 文件名列表
            save_to_db: 是否保存到数据库
            
        Returns:
            包含所有步骤结果的字典
        """
        results: Dict[str, Union[UploadResponse, EmbeddingResponse, SaveToDBResponse]] = {}
        
        # 1. 上传文档
        logger.info("开始文档上传...")
        upload_response = self.upload_documents(markdown_documents, file_names)
        results['upload'] = upload_response
        
        # 2. 生成嵌入
        logger.info("开始生成嵌入...")
        embedding_response = self.generate_embeddings(
            upload_response.doc_analysis_list,
            upload_response.chunk_analysis_list
        )
        results['embedding'] = embedding_response
        
        # 3. 保存到数据库（可选）
        if save_to_db:
            logger.info("开始保存到数据库...")
            save_response = self.save_to_database(
                doc_embedding_list=embedding_response.doc_embedding_list,
                chunk_embedding_list=embedding_response.chunk_embedding_list
            )
            results['save'] = save_response
        
        logger.info("完整文档处理管道执行完成")
        return results
    
    async def full_retrieval_pipeline_async(
        self,
        query: str,
        data_set_type: Optional[str] = None,
        each_doc_chunk_number: int = 10,
        use_chunk_decision: bool = True
    ) -> Dict[str, RetrievalResponse]:
        """
        异步完整检索管道
        
        执行完整的检索流程：文档检索 -> chunk 检索 -> 决策匹配
        
        Args:
            query: 查询文本
            data_set_type: 数据集类型（可选）
            each_doc_chunk_number: 每个文档返回的 chunk 数量
            use_chunk_decision: 是否使用 chunk 决策
            
        Returns:
            包含所有步骤结果的字典
        """
        results: Dict[str, RetrievalResponse] = {}
        
        # 1. 文档检索
        logger.info(f"开始文档检索: {query}")
        if data_set_type:
            doc_response = await self.retrieve_documents_by_dataset_async(data_set_type, query)
        else:
            doc_response = await self.retrieve_documents_async(query)
        results['documents'] = doc_response
        
        # 2. 提取文档 ID
        doc_ids = [doc.get('id') for doc in doc_response.results if doc.get('id')]
        
        if doc_ids:
            # 3. Chunk 检索
            logger.info(f"开始 chunk 检索: {len(doc_ids)} 个文档")
            chunk_response = await self.retrieve_chunks_async(
                query, doc_ids, each_doc_chunk_number
            )
            results['chunks'] = chunk_response
            
            # 4. Chunk 决策（可选）
            if use_chunk_decision and chunk_response.results:
                logger.info("开始 chunk 决策匹配")
                decision_response = await self.make_chunk_decision_async(
                    query, chunk_response.results
                )
                results['decision'] = decision_response
        
        logger.info("完整检索管道执行完成")
        return results
    
    def full_retrieval_pipeline(
        self,
        query: str,
        data_set_type: Optional[str] = None,
        each_doc_chunk_number: int = 10,
        use_chunk_decision: bool = True
    ) -> Dict[str, RetrievalResponse]:
        """
        同步完整检索管道
        
        执行完整的检索流程：文档检索 -> chunk 检索 -> 决策匹配
        
        Args:
            query: 查询文本
            data_set_type: 数据集类型（可选）
            each_doc_chunk_number: 每个文档返回的 chunk 数量
            use_chunk_decision: 是否使用 chunk 决策
            
        Returns:
            包含所有步骤结果的字典
        """
        results: Dict[str, RetrievalResponse] = {}
        
        # 1. 文档检索
        logger.info(f"开始文档检索: {query}")
        if data_set_type:
            doc_response = self.retrieve_documents_by_dataset(data_set_type, query)
        else:
            doc_response = self.retrieve_documents(query)
        results['documents'] = doc_response
        
        # 2. 提取文档 ID
        doc_ids = [doc.get('id') for doc in doc_response.results if doc.get('id')]
        
        if doc_ids:
            # 3. Chunk 检索
            logger.info(f"开始 chunk 检索: {len(doc_ids)} 个文档")
            chunk_response = self.retrieve_chunks(
                query, doc_ids, each_doc_chunk_number
            )
            results['chunks'] = chunk_response
            
            # 4. Chunk 决策（可选）
            if use_chunk_decision and chunk_response.results:
                logger.info("开始 chunk 决策匹配")
                decision_response = self.make_chunk_decision(
                    query, chunk_response.results
                )
                results['decision'] = decision_response
        
        logger.info("完整检索管道执行完成")
        return results


# ===== 便利函数 =====

def create_client(
    base_url: str = "http://localhost:8955",
    timeout: float = 30.0,
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> KnowledgeRAGClient:
    """
    创建 KnowledgeRAG 客户端实例
    
    Args:
        base_url: API 服务器基础 URL
        timeout: 请求超时时间（秒）
        max_retries: 最大重试次数
        retry_delay: 重试间隔（秒）
        
    Returns:
        客户端实例
    """
    return KnowledgeRAGClient(
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
        retry_delay=retry_delay
    )


# ===== 使用示例 =====

if __name__ == "__main__":
    import asyncio
    
    async def example_usage():
        """使用示例"""
        # 创建客户端
        async with KnowledgeRAGClient("http://localhost:8955") as client:
            try:
                # 健康检查
                health = await client.health_check_async()
                print(f"服务状态: {health.status}")
                
                # 文档上传示例
                documents = ["# 示例文档\n\n这是一个示例文档内容。"]
                file_names = ["example.md"]
                
                # 执行完整文档处理管道
                pipeline_result = await client.full_document_pipeline_async(
                    documents, file_names, save_to_db=True
                )
                print(f"文档处理完成: {pipeline_result['upload'].total_docs} 个文档")
                
                # 检索示例
                retrieval_result = await client.full_retrieval_pipeline_async(
                    "示例查询",
                    use_chunk_decision=True
                )
                print(f"检索完成: {retrieval_result['documents'].total_results} 个文档")
                
            except Exception as e:
                print(f"示例执行失败: {e}")
    
    # 运行示例
    # asyncio.run(example_usage())
