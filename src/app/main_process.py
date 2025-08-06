"""
KnowledgeRAG FastAPI 服务器
==========================

作者: Bin Liang
日期: 2025-08-01
描述: 使用 FastAPI 将 KnowledgeRAG 系统部署为 HTTP API 服务

主要功能:
1. 文档上传和处理
2. 文档嵌入和存储
3. 知识检索
4. 文档和 chunk 检索
5. 维护全局 retriever 单例

技术栈:
- FastAPI: 高性能异步 Web 框架
- Pydantic: 数据验证和序列化
- 全局单例模式: 维护 retriever 实例
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from loguru import logger

# 导入 KnowledgeRAG 模块
from knowledge_rag.main_process import (
    upload_md_file,
    embedding_doc_and_chunk,
    save_file_to_db,
    doc_retrieval,
    doc_retrieval_by_data_set_type,
    chunk_retrieval,
    make_decision_of_chunk_retrieval
)
from knowledge_rag.config import KnowledgeRAGSettings
from knowledge_rag.knowledge_retrieval.db_retriever import create_retriever, DBRetriever

# 全局变量
retriever: Optional[DBRetriever] = None
config: Optional[KnowledgeRAGSettings] = None

# ===== Pydantic 数据模型 =====

class UploadRequest(BaseModel):
    """文档上传请求模型"""
    markdown_documents: List[str] = Field(..., description="Markdown 文档内容列表")
    file_names: List[str] = Field(..., description="文件名列表")
    
    class Config:
        json_schema_extra = {
            "example": {
                "markdown_documents": ["# 标题\n\n这是文档内容..."],
                "file_names": ["document1.md"]
            }
        }

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
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "机器学习算法的应用"
            }
        }

class DocRetrievalByDataSetRequest(BaseModel):
    """按数据集类型检索文档请求模型"""
    data_set_type: str = Field(..., description="数据集类型")
    query: str = Field(..., description="查询文本", min_length=1)
    
    class Config:
        json_schema_extra = {
            "example": {
                "data_set_type": "paper_set_1",
                "query": "深度学习在计算机视觉中的应用"
            }
        }

class ChunkRetrievalRequest(BaseModel):
    """chunk 检索请求模型"""
    query: str = Field(..., description="查询文本", min_length=1)
    doc_ids: List[str] = Field(..., description="文档 ID 列表")
    each_doc_chunk_number: int = Field(default=10, description="每个文档返回的 chunk 数量", ge=1, le=50)
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "神经网络优化方法",
                "doc_ids": ["doc_001", "doc_002"],
                "each_doc_chunk_number": 10
            }
        }

class ChunkDecisionRequest(BaseModel):
    """chunk 决策请求模型"""
    query_text: str = Field(..., description="查询文本", min_length=1)
    chunks: List[dict] = Field(..., description="chunk 列表")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query_text": "机器学习模型评估",
                "chunks": [{"content": "chunk 内容", "metadata": {}}]
            }
        }

# ===== 响应模型 =====

class UploadResponse(BaseModel):
    """文档上传响应模型"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")
    doc_analysis_list: List[dict] = Field(..., description="文档分析结果")
    chunk_analysis_list: List[dict] = Field(..., description="chunk 分析结果")
    total_docs: int = Field(..., description="处理的文档总数")
    total_chunks: int = Field(..., description="处理的 chunk 总数")

class EmbeddingResponse(BaseModel):
    """嵌入响应模型"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")
    doc_embedding_list: List[dict] = Field(..., description="文档嵌入列表")
    chunk_embedding_list: List[dict] = Field(..., description="chunk 嵌入列表")

class SaveToDBResponse(BaseModel):
    """保存到数据库响应模型"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")

class RetrievalResponse(BaseModel):
    """检索响应模型"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")
    results: List[Any] = Field(..., description="检索结果")
    total_results: int = Field(..., description="结果总数")

# ===== 生命周期管理 =====

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global retriever, config
    
    # 启动时初始化
    logger.info("🚀 启动 KnowledgeRAG API 服务器...")
    
    try:
        # 加载配置
        config = KnowledgeRAGSettings.from_env()
        logger.info("✅ 配置加载完成")
        
        # 创建 retriever 单例
        mysql_config = {
            'host': config.database.host,
            'port': config.database.port,
            'user': config.database.user,
            'password': config.database.password,
            'database': config.database.database,
            'charset': config.database.charset
        }
        
        milvus_config = {
            'host': config.milvus.host,
            'port': config.milvus.port,
            'alias': config.milvus.alias
        }
        
        retriever = await create_retriever(mysql_config, milvus_config)
        logger.info("✅ Retriever 单例创建成功")
        
    except Exception as e:
        logger.error(f"❌ 服务器启动失败: {e}")
        raise
    
    yield
    
    # 关闭时清理资源
    logger.info("🔄 关闭 KnowledgeRAG API 服务器...")
    if retriever:
        retriever.close()
        logger.info("✅ Retriever 连接已关闭")

# ===== FastAPI 应用初始化 =====

app = FastAPI(
    title="KnowledgeRAG API",
    description="基于 RAG 的知识检索系统 API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== API 端点 =====

@app.get("/", summary="健康检查", tags=["系统"])
async def root():
    """API 健康检查端点"""
    return {
        "message": "KnowledgeRAG API 服务正常运行",
        "version": "1.0.0",
        "status": "healthy",
        "retriever_status": "connected" if retriever else "disconnected"
    }

@app.get("/health", summary="详细健康检查", tags=["系统"])
async def health_check():
    """详细的健康检查端点"""
    return {
        "status": "healthy",
        "retriever": {
            "connected": retriever is not None,
            "mysql_connected": retriever.mysql_conn is not None if retriever else False,
            "milvus_connected": retriever.milvus_connected if retriever else False
        },
        "config": {
            "environment": config.environment if config else "unknown",
            "debug": config.debug if config else False
        }
    }

@app.post("/upload", 
          response_model=UploadResponse,
          summary="上传并处理 Markdown 文档", 
          tags=["文档处理"])
async def upload_documents(request: UploadRequest):
    """
    上传并处理 Markdown 文档
    
    - **markdown_documents**: Markdown 文档内容列表
    - **file_names**: 对应的文件名列表
    
    返回文档和 chunk 的分析结果
    """
    try:
        if len(request.markdown_documents) != len(request.file_names):
            raise HTTPException(
                status_code=400, 
                detail="文档数量与文件名数量不匹配"
            )
        
        logger.info(f"开始处理 {len(request.markdown_documents)} 个文档")
        
        doc_analysis_list, chunk_analysis_list = await upload_md_file(
            request.markdown_documents, 
            request.file_names
        )
        
        logger.info(f"文档处理完成: {len(doc_analysis_list)} 个文档, {len(chunk_analysis_list)} 个 chunks")
        
        return UploadResponse(
            success=True,
            message="文档上传和处理成功",
            doc_analysis_list=doc_analysis_list,
            chunk_analysis_list=chunk_analysis_list,
            total_docs=len(doc_analysis_list),
            total_chunks=len(chunk_analysis_list)
        )
        
    except Exception as e:
        logger.error(f"文档上传处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")

@app.post("/embedding", 
          response_model=EmbeddingResponse,
          summary="生成文档和 chunk 嵌入", 
          tags=["文档处理"])
async def generate_embeddings(request: EmbeddingRequest):
    """
    为文档和 chunk 生成嵌入向量
    
    - **doc_analysis_list**: 文档分析结果列表
    - **chunk_analysis_list**: chunk 分析结果列表
    
    返回带有嵌入向量的文档和 chunk 数据
    """
    try:
        logger.info(f"开始生成嵌入: {len(request.doc_analysis_list)} 个文档, {len(request.chunk_analysis_list)} 个 chunks")
        
        doc_embedding_list, chunk_embedding_list = await embedding_doc_and_chunk(
            request.doc_analysis_list,
            request.chunk_analysis_list
        )
        
        logger.info("嵌入生成完成")
        
        return EmbeddingResponse(
            success=True,
            message="嵌入生成成功",
            doc_embedding_list=doc_embedding_list,
            chunk_embedding_list=chunk_embedding_list
        )
        
    except Exception as e:
        logger.error(f"嵌入生成失败: {e}")
        raise HTTPException(status_code=500, detail=f"嵌入生成失败: {str(e)}")

@app.post("/save-to-db", 
          response_model=SaveToDBResponse,
          summary="保存数据到数据库", 
          tags=["文档处理"])
async def save_to_database(request: SaveToDBRequest):
    """
    将处理后的数据保存到数据库
    
    可以通过以下两种方式之一提供数据:
    1. 提供 JSON 文件路径
    2. 直接提供嵌入数据列表
    """
    try:
        logger.info("开始保存数据到数据库")
        
        await save_file_to_db(
            doc_json_file_path=request.doc_json_file_path,
            chunk_json_file_path=request.chunk_json_file_path,
            doc_embedding_list=request.doc_embedding_list,
            chunk_embedding_list=request.chunk_embedding_list
        )
        
        logger.info("数据保存到数据库完成")
        
        return SaveToDBResponse(
            success=True,
            message="数据保存到数据库成功"
        )
        
    except Exception as e:
        logger.error(f"数据保存失败: {e}")
        raise HTTPException(status_code=500, detail=f"数据保存失败: {str(e)}")

@app.post("/doc-retrieval", 
          response_model=RetrievalResponse,
          summary="文档检索", 
          tags=["知识检索"])
async def document_retrieval(request: DocRetrievalRequest):
    """
    根据查询文本检索相关文档
    
    - **query**: 查询文本
    
    返回相关文档的匹配结果
    """
    try:
        if not retriever:
            raise HTTPException(status_code=503, detail="Retriever 未初始化")
        
        logger.info(f"开始文档检索: {request.query}")
        
        results = await doc_retrieval(retriever, request.query)
        
        logger.info(f"文档检索完成: 找到 {len(results)} 个结果")
        
        return RetrievalResponse(
            success=True,
            message="文档检索成功",
            results=results,
            total_results=len(results)
        )
        
    except Exception as e:
        logger.error(f"文档检索失败: {e}")
        raise HTTPException(status_code=500, detail=f"文档检索失败: {str(e)}")

@app.post("/doc-retrieval-by-dataset", 
          response_model=RetrievalResponse,
          summary="按数据集类型检索文档", 
          tags=["知识检索"])
async def document_retrieval_by_dataset(request: DocRetrievalByDataSetRequest):
    """
    在指定数据集类型中检索相关文档
    
    - **data_set_type**: 数据集类型
    - **query**: 查询文本
    
    返回指定数据集中的相关文档
    """
    try:
        if not retriever:
            raise HTTPException(status_code=503, detail="Retriever 未初始化")
        
        logger.info(f"开始按数据集检索: {request.data_set_type} - {request.query}")
        
        results = await doc_retrieval_by_data_set_type(
            retriever, 
            request.data_set_type, 
            request.query
        )
        
        logger.info(f"数据集文档检索完成: 找到 {len(results)} 个结果")
        
        return RetrievalResponse(
            success=True,
            message="数据集文档检索成功",
            results=results,
            total_results=len(results)
        )
        
    except Exception as e:
        logger.error(f"数据集文档检索失败: {e}")
        raise HTTPException(status_code=500, detail=f"数据集文档检索失败: {str(e)}")

@app.post("/chunk-retrieval", 
          response_model=RetrievalResponse,
          summary="chunk 检索", 
          tags=["知识检索"])
async def chunk_retrieval_endpoint(request: ChunkRetrievalRequest):
    """
    在指定文档中检索相关 chunks
    
    - **query**: 查询文本
    - **doc_ids**: 要搜索的文档 ID 列表
    - **each_doc_chunk_number**: 每个文档返回的 chunk 数量 (默认: 10)
    
    返回相关的 chunk 片段
    """
    try:
        if not retriever:
            raise HTTPException(status_code=503, detail="Retriever 未初始化")
        
        logger.info(f"开始 chunk 检索: {request.query} 在 {len(request.doc_ids)} 个文档中")
        
        results = await chunk_retrieval(
            retriever,
            request.query,
            request.doc_ids,
            request.each_doc_chunk_number
        )
        
        logger.info(f"Chunk 检索完成: 找到 {len(results)} 个结果")
        
        return RetrievalResponse(
            success=True,
            message="Chunk 检索成功",
            results=results,
            total_results=len(results)
        )
        
    except Exception as e:
        logger.error(f"Chunk 检索失败: {e}")
        raise HTTPException(status_code=500, detail=f"Chunk 检索失败: {str(e)}")

@app.post("/chunk-decision", 
          response_model=RetrievalResponse,
          summary="chunk 决策匹配", 
          tags=["知识检索"])
async def chunk_decision_endpoint(request: ChunkDecisionRequest):
    """
    对 chunks 进行决策匹配，筛选最相关的内容
    
    - **query_text**: 查询文本
    - **chunks**: chunk 列表
    
    返回经过决策匹配的 chunk 结果
    """
    try:
        logger.info(f"开始 chunk 决策匹配: {len(request.chunks)} 个 chunks")
        
        results = await make_decision_of_chunk_retrieval(
            request.query_text,
            request.chunks
        )
        
        logger.info(f"Chunk 决策匹配完成: {len(results)} 个结果")
        
        return RetrievalResponse(
            success=True,
            message="Chunk 决策匹配成功",
            results=results,
            total_results=len(results)
        )
        
    except Exception as e:
        logger.error(f"Chunk 决策匹配失败: {e}")
        raise HTTPException(status_code=500, detail=f"Chunk 决策匹配失败: {str(e)}")

# ===== 错误处理 =====

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP 异常处理器"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail,
            "status_code": exc.status_code
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """通用异常处理器"""
    logger.error(f"未处理的异常: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "内部服务器错误",
            "status_code": 500
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    # 运行服务器
    uvicorn.run(
        "main_process:app",
        host="0.0.0.0",
        port=8955,
        reload=True,
        log_level="info"
    )
