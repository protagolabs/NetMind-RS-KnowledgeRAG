#!/usr/bin/env python3
"""
Retrieval API Server

This module provides a FastAPI server for document retrieval operations.
It uses the DocumentRetriever class to perform hierarchical search and
returns structured results.

Author: yujing.wang
Date: 2025.07.25
"""

import os
import asyncio
import json
from typing import List, Dict, Optional, Any
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import logging

from src.agentic_rag.document_retriever import DocumentRetriever, RetrievalResult, SearchResult

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Document Retrieval API",
    description="API for hierarchical document retrieval using vector search",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global retriever instance
retriever: Optional[DocumentRetriever] = None

# Pydantic models
class SearchRequest(BaseModel):
    """Search request model."""
    query: str = Field(..., description="Search query")
    search_type: str = Field(default="hierarchical", description="Search type: hierarchical, summary, chunk")
    model: str = Field(default="gpt-4o", description="Model to use for synthesis")
    similarity_threshold: Optional[float] = Field(default=None, description="Similarity threshold (0.0-1.0)")

class SearchResultResponse(BaseModel):
    """Search result response model."""
    doc_id: str
    content: str
    score: float
    source_type: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RetrievalResultResponse(BaseModel):
    """Complete retrieval result response model."""
    success: bool
    query: str
    summary_results: List[SearchResultResponse] = Field(default_factory=list)
    chunk_results: List[SearchResultResponse] = Field(default_factory=list)
    final_answer: str = ""
    search_stats: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    documents_count: int
    structure_files: int
    vector_stores_configured: bool
    summary_strategy: str
    chunk_strategy: str

class StatusResponse(BaseModel):
    """Status response model."""
    vector_store_ids: Dict[str, Optional[str]]
    documents_count: int
    structure_directory: str
    structure_files: int
    summary_search_strategy: str
    chunk_search_strategy: str

@app.on_event("startup")
async def startup_event():
    """Initialize retriever on startup."""
    global retriever
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    try:
        # Load configuration
        config_file = "document_processor_config.json"
        if not Path(config_file).exists():
            raise ValueError(f"Configuration file not found: {config_file}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Extract configuration data
        documents_info = config.get("documents_info", {})
        vector_store_ids = config.get("vector_store_ids", {})
        structure_directory = config.get("structure_directory", "pdf_structure")
        
        if not documents_info:
            raise ValueError("No documents found in configuration")
        
        # Create document retriever
        retriever = DocumentRetriever(
            openai_api_key=openai_api_key,
            documents_info=documents_info,
            vector_store_ids=vector_store_ids,
            structure_directory=structure_directory
        )
        
        logger.info("✅ Document retriever initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize document retriever: {e}")
        raise

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if not retriever:
        raise HTTPException(status_code=503, detail="Document retriever not initialized")
    
    try:
        status = retriever.get_status()
        return HealthResponse(
            status="healthy",
            documents_count=status["documents_count"],
            structure_files=status["structure_files"],
            vector_stores_configured=bool(status["vector_store_ids"].get("summaries") or 
                                        status["vector_store_ids"].get("chunks")),
            summary_strategy=status["summary_search_strategy"],
            chunk_strategy=status["chunk_search_strategy"]
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {e}")

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get detailed system status."""
    if not retriever:
        raise HTTPException(status_code=503, detail="Document retriever not initialized")
    
    try:
        return StatusResponse(**retriever.get_status())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get status: {e}")

@app.post("/search", response_model=RetrievalResultResponse)
async def search_documents(request: SearchRequest):
    """Search documents using hierarchical search."""
    if not retriever:
        raise HTTPException(status_code=503, detail="Document retriever not initialized")
    
    try:
        logger.info(f"Search request: {request.query[:50]}... (type: {request.search_type})")
        
        # Check if documents are available
        status = retriever.get_status()
        if status["documents_count"] == 0:
            return RetrievalResultResponse(
                success=False,
                query=request.query,
                error="No documents available. Please process documents first."
            )
        
        # Execute search based on type
        if request.search_type == "hierarchical":
            result = await retriever.hierarchical_search(request.query, request.model)
        else:
            # For other search types, get the result string
            result_string = await retriever.search_documents(
                request.query, request.search_type, request.model
            )
            # Create a simple result structure
            result = RetrievalResult(
                query=request.query,
                final_answer=result_string,
                search_stats={"search_type": request.search_type}
            )
        
        # Convert to response format
        summary_results = [
            SearchResultResponse(
                doc_id=sr.doc_id,
                content=sr.content,
                score=sr.score,
                source_type=sr.source_type,
                metadata=sr.metadata
            ) for sr in result.summary_results
        ]
        
        chunk_results = [
            SearchResultResponse(
                doc_id=cr.doc_id,
                content=cr.content,
                score=cr.score,
                source_type=cr.source_type,
                metadata=cr.metadata
            ) for cr in result.chunk_results
        ]
        
        return RetrievalResultResponse(
            success=True,
            query=result.query,
            summary_results=summary_results,
            chunk_results=chunk_results,
            final_answer=result.final_answer,
            search_stats=result.search_stats
        )
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return RetrievalResultResponse(
            success=False,
            query=request.query,
            error=f"Search failed: {str(e)}"
        )

@app.post("/search/simple")
async def simple_search(request: SearchRequest):
    """Simple search endpoint that returns only the final answer."""
    if not retriever:
        raise HTTPException(status_code=503, detail="Document retriever not initialized")
    
    try:
        logger.info(f"Simple search request: {request.query[:50]}...")
        
        # Check if documents are available
        status = retriever.get_status()
        if status["documents_count"] == 0:
            return {
                "success": False,
                "error": "No documents available. Please process documents first."
            }
        
        # Execute search
        if request.search_type == "hierarchical":
            result = await retriever.hierarchical_search(request.query, request.model)
            return {
                "success": True,
                "data": result.final_answer
            }
        else:
            result_string = await retriever.search_documents(
                request.query, request.search_type, request.model
            )
            return {
                "success": True,
                "data": result_string
            }
        
    except Exception as e:
        logger.error(f"Simple search failed: {e}")
        return {
            "success": False,
            "error": f"Search failed: {str(e)}"
        }

@app.get("/document/{doc_id}/structure")
async def get_document_structure(doc_id: str):
    """Get document structure from local file."""
    if not retriever:
        raise HTTPException(status_code=503, detail="Document retriever not initialized")
    
    try:
        structure = retriever.get_document_structure(doc_id)
        if structure:
            return {
                "success": True,
                "doc_id": doc_id,
                "structure": structure
            }
        else:
            return {
                "success": False,
                "error": f"Document structure not found for {doc_id}"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document structure: {e}")

@app.post("/reload")
async def reload_config():
    """Reload configuration from file."""
    global retriever
    
    try:
        # Reinitialize retriever
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        config_file = "document_processor_config.json"
        if not Path(config_file).exists():
            return {"success": False, "message": f"Configuration file not found: {config_file}"}
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        documents_info = config.get("documents_info", {})
        vector_store_ids = config.get("vector_store_ids", {})
        structure_directory = config.get("structure_directory", "pdf_structure")
        
        if not documents_info:
            return {"success": False, "message": "No documents found in configuration"}
        
        retriever = DocumentRetriever(
            openai_api_key=openai_api_key,
            documents_info=documents_info,
            vector_store_ids=vector_store_ids,
            structure_directory=structure_directory
        )
        
        return {"success": True, "message": "Configuration reloaded successfully"}
        
    except Exception as e:
        return {"success": False, "message": f"Failed to reload configuration: {e}"}

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Document Retrieval API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "status": "/status",
            "search": "/search",
            "simple_search": "/search/simple",
            "document_structure": "/document/{doc_id}/structure",
            "reload": "/reload"
        },
        "documentation": "/docs"
    }

if __name__ == "__main__":
    # Run server
    uvicorn.run(
        "retrieval_api_server:app",
        host="0.0.0.0",
        port=8001,  # Different port from the original RAG API
        reload=True,
        log_level="info"
    ) 