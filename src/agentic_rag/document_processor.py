#!/usr/bin/env python3
"""
Document Processor Module

This module handles document processing, including file upload, content extraction,
vectorization, and storage operations. It follows the Single Responsibility Principle
by focusing solely on document processing tasks.

The module processes PDF documents by:
1. Extracting text and images from PDF pages
2. Generating document summaries using AI
3. Creating text chunks for vectorization
4. Storing summaries and chunks in vector databases
5. Saving document structures to local files

Author: yujing.wang
Date: 2025.07.25
"""

import os
import json
import asyncio
import time
import fitz  # PyMuPDF
from typing import List, Dict, Optional, Union, Tuple, Any
from pathlib import Path
import openai
from openai import OpenAI
import litellm
from litellm.vector_stores.vector_store_registry import VectorStoreRegistry, LiteLLM_ManagedVectorStore
from litellm.vector_stores.main import acreate
import logging
import hashlib
from PIL import Image
import io
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def retry_with_backoff(func, max_retries: int = 3, base_delay: float = 1.0):
    """Retry function with exponential backoff.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retries
        base_delay: Base delay in seconds
        
    Returns:
        Function result
        
    Raises:
        Exception: If all retries fail
    """
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except Exception as e:
            if attempt == max_retries:
                raise e
            
            # Calculate delay with jitter
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f} seconds...")
            await asyncio.sleep(delay)


def validate_file_size(content: str, max_size_mb: int = 25) -> bool:
    """Validate file size before upload.
    
    Args:
        content: File content
        max_size_mb: Maximum file size in MB
        
    Returns:
        True if file size is acceptable
    """
    size_mb = len(content.encode('utf-8')) / (1024 * 1024)
    return size_mb <= max_size_mb


def truncate_content(content: str, max_chars: int = 100000) -> str:
    """Truncate content if it's too long.
    
    Args:
        content: Original content
        max_chars: Maximum number of characters
        
    Returns:
        Truncated content
    """
    if len(content) <= max_chars:
        return content
    
    # Try to truncate at sentence boundary
    truncated = content[:max_chars]
    last_period = truncated.rfind('.')
    last_newline = truncated.rfind('\n')
    
    if last_period > max_chars * 0.8:  # If period is in last 20%
        return truncated[:last_period + 1]
    elif last_newline > max_chars * 0.8:  # If newline is in last 20%
        return truncated[:last_newline + 1]
    else:
        return truncated + "..."


@dataclass
class DocumentMetadata:
    """Document metadata information.
    
    Attributes:
        doc_id: Unique document identifier
        doc_name: Document name without extension
        file_extension: File extension
        file_size: File size in bytes
        upload_time: Upload timestamp
        page_count: Number of pages in document
        text_length: Total text length in characters
        image_count: Number of page images generated
    """
    doc_id: str
    doc_name: str
    file_extension: str
    file_size: int
    upload_time: str
    page_count: int = 0
    text_length: int = 0
    image_count: int = 0


@dataclass
class DocumentContent:
    """Extracted document content.
    
    Attributes:
        text: Full document text
        pages: List of page information
        page_images: List of page image information
        chunks: Text chunks for vectorization
        summary: Document summary
        structure: Document structural analysis (saved to local file)
    """
    text: str = ""
    pages: List[Dict[str, Any]] = field(default_factory=list)
    page_images: List[Dict[str, Any]] = field(default_factory=list)
    chunks: List[str] = field(default_factory=list)
    summary: str = ""
    structure: str = ""


class ContentExtractor(ABC):
    """Abstract base class for content extraction strategies."""
    
    @abstractmethod
    def extract_content(self, file_path: Path) -> DocumentContent:
        """Extract content from document file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Extracted document content
        """
        pass


class PDFContentExtractor(ContentExtractor):
    """PDF content extraction implementation."""
    
    def __init__(self, images_directory: Path):
        """Initialize PDF content extractor.
        
        Args:
            images_directory: Directory to store extracted images
        """
        self.images_directory = images_directory
        self.images_directory.mkdir(exist_ok=True)
    
    def extract_content(self, file_path: Path) -> DocumentContent:
        """Extract content from PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted document content
        """
        try:
            doc = fitz.open(str(file_path))
            content = DocumentContent()
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Extract text
                page_text = page.get_text()
                content.text += f"\n--- 第 {page_num + 1} 页 ---\n{page_text}"
                
                # Page information
                page_info = {
                    "page_number": page_num + 1,
                    "text": page_text,
                    "page_image_path": ""
                }
                
                # Convert page to image
                page_image_info = self._convert_page_to_image(page, page_num, file_path)
                if page_image_info:
                    page_info["page_image_path"] = page_image_info["path"]
                    content.page_images.append(page_image_info)
                
                content.pages.append(page_info)
            
            doc.close()
            logger.info(f"Successfully extracted content: {file_path.name} "
                       f"({len(content.pages)} pages, {len(content.page_images)} images)")
            return content
            
        except Exception as e:
            logger.error(f"Failed to extract content from {file_path.name}: {e}")
            return DocumentContent()
    
    def _convert_page_to_image(self, page: fitz.Page, page_num: int, 
                              file_path: Path) -> Optional[Dict[str, Any]]:
        """Convert PDF page to image.
        
        Args:
            page: PDF page object
            page_num: Page number (0-based)
            file_path: Original PDF file path
            
        Returns:
            Image information dictionary or None if conversion fails
        """
        try:
            # Set zoom factor for better image quality
            zoom = 2.0
            mat = fitz.Matrix(zoom, zoom)
            
            # Render page as image
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Save page image
            doc_id = self._generate_document_id(file_path)
            img_dir = self.images_directory / doc_id
            img_dir.mkdir(exist_ok=True)
            
            # Image naming: page_页码.png
            img_filename = f"page_{page_num + 1}.png"
            img_path = img_dir / img_filename
            
            with open(img_path, "wb") as img_file:
                img_file.write(img_data)
            
            pix = None  # Release memory
            
            return {
                "page": page_num + 1,
                "path": str(img_path),
                "filename": img_filename
            }
            
        except Exception as e:
            logger.warning(f"Failed to convert page {page_num + 1} to image: {e}")
            return None
    
    def _generate_document_id(self, file_path: Path) -> str:
        """Generate unique document ID.
        
        Args:
            file_path: Document file path
            
        Returns:
            Unique document ID
        """
        content = f"{file_path.absolute()}_{file_path.stat().st_mtime}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


class TextChunker:
    """Text chunking utility class."""
    
    def __init__(self, chunk_size: int = 4000, overlap: int = 500, max_chunks_per_doc: int = 50):
        """Initialize text chunker.
        
        Args:
            chunk_size: Size of each text chunk
            overlap: Overlap between chunks
            max_chunks_per_doc: Maximum number of chunks per document
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_chunks_per_doc = max_chunks_per_doc
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # If not the last chunk, try to split at sentence boundary
            if end < len(text):
                # Find the nearest period or newline
                for i in range(end, max(start + self.chunk_size - 200, start), -1):
                    if text[i] in '.。\n':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - self.overlap
            if start >= len(text):
                break
            
            # Check if we've exceeded max chunks
            if len(chunks) >= self.max_chunks_per_doc:
                logger.warning(f"Reached maximum chunks per document ({self.max_chunks_per_doc}). "
                             f"Truncating remaining text.")
                break
        
        logger.info(f"Text chunking completed: {len(chunks)} chunks")
        return chunks
    
    def adjust_chunk_size_for_large_text(self, text: str) -> List[str]:
        """Adjust chunk size for very large texts to avoid too many chunks.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        # Estimate number of chunks with current settings
        estimated_chunks = len(text) // (self.chunk_size - self.overlap)
        
        if estimated_chunks > self.max_chunks_per_doc:
            # Increase chunk size to reduce number of chunks
            new_chunk_size = int(len(text) / self.max_chunks_per_doc) + self.overlap
            logger.info(f"Adjusting chunk size from {self.chunk_size} to {new_chunk_size} "
                       f"to avoid too many chunks")
            
            # Create temporary chunker with adjusted size
            temp_chunker = TextChunker(
                chunk_size=new_chunk_size,
                overlap=self.overlap,
                max_chunks_per_doc=self.max_chunks_per_doc
            )
            return temp_chunker.chunk_text(text)
        
        return self.chunk_text(text)


class DocumentAnalyzer:
    """Document analysis and summarization class."""
    
    def __init__(self, openai_api_key: str):
        """Initialize document analyzer.
        
        Args:
            openai_api_key: OpenAI API key
        """
        self.openai_api_key = openai_api_key
        litellm.api_key = openai_api_key
    
    async def generate_document_summary(self, pages_data: List[Dict], 
                                      doc_name: str, model_name: str) -> str:
        """Generate document summary with batch processing.
        
        Args:
            pages_data: List of page data dictionaries
            doc_name: Document name
            
        Returns:
            Generated document summary
        """
        try:
            if not pages_data:
                return "Document is empty, cannot generate summary"
            
            logger.info(f"Starting document summary generation: {doc_name} "
                       f"(Total pages: {len(pages_data)})")
            
            # Process in batches of 10 pages
            batch_size = 10
            total_batches = (len(pages_data) + batch_size - 1) // batch_size
            previous_summary = ""
            
            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, len(pages_data))
                current_batch = pages_data[start_idx:end_idx]
                
                logger.info(f"Processing batch {batch_num + 1}/{total_batches}: "
                           f"Pages {start_idx + 1}-{end_idx}")
                
                # Build system message with all text content
                system_prompt = self._build_summary_system_prompt(
                    doc_name, start_idx, end_idx, len(pages_data), 
                    previous_summary, batch_num, total_batches
                )
                
                # # Build user message with images
                # user_content = self._build_summary_user_content(
                #     current_batch, start_idx, end_idx, doc_name
                # )
                
                # Call LLM
                response = await litellm.acompletion(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        # {"role": "user", "content": user_content}
                    ],
                )
                
                current_summary = response.choices[0].message.content
                
                # Update previous summary
                if batch_num == total_batches - 1:
                    logger.info(f"Summary generation completed: {doc_name}")
                    return current_summary
                else:
                    if previous_summary:
                        previous_summary = f"{previous_summary}\n\n{current_summary}"
                    else:
                        previous_summary = current_summary
                
                logger.info(f"✅ Batch {batch_num + 1} processing completed")
            
        except Exception as e:
            logger.error(f"Summary generation failed for {doc_name}: {e}")
            return f"Document summary generation failed: {e}"
    
    async def generate_document_structure(self, pages_data: List[Dict], 
                                        doc_name: str, model_name: str) -> str:
        """Generate document structural analysis.
        
        Args:
            pages_data: List of page data dictionaries
            doc_name: Document name
            
        Returns:
            Generated structural analysis
        """
        try:
            if not pages_data:
                return "Document is empty, cannot generate structural summary"
            
            logger.info(f"Starting document structural analysis: {doc_name} "
                       f"(Total pages: {len(pages_data)})")
            
            # Process in batches of 10 pages
            batch_size = 10
            total_batches = (len(pages_data) + batch_size - 1) // batch_size
            previous_structure = ""
            
            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, len(pages_data))
                current_batch = pages_data[start_idx:end_idx]
                
                logger.info(f"Processing batch {batch_num + 1}/{total_batches}: "
                           f"Pages {start_idx + 1}-{end_idx}")
                
                # Build system message
                system_prompt = self._build_structure_system_prompt(
                    doc_name, start_idx, end_idx, len(pages_data), 
                    previous_structure, batch_num, total_batches
                )
                
                # # Build user message with images
                # user_content = self._build_structure_user_content(
                #     current_batch, start_idx, end_idx, doc_name
                # )
                
                # Call LLM
                response = await litellm.acompletion(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        # {"role": "user", "content": user_content}
                    ],
                )
                
                current_structure = response.choices[0].message.content
                
                # Update previous structure
                if batch_num == total_batches - 1:
                    logger.info(f"Structural analysis completed: {doc_name}")
                    return current_structure
                else:
                    if previous_structure:
                        previous_structure = f"{previous_structure}\n\n{current_structure}"
                    else:
                        previous_structure = current_structure
                
                logger.info(f"✅ Batch {batch_num + 1} processing completed")
            
        except Exception as e:
            logger.error(f"Structural analysis failed for {doc_name}: {e}")
            return f"Structural analysis generation failed: {e}"
    
    def _build_summary_system_prompt(self, doc_name: str, start_idx: int, 
                                   end_idx: int, total_pages: int, 
                                   previous_summary: str, batch_num: int, 
                                   total_batches: int) -> str:
        """Build system prompt for summary generation.
        
        Args:
            doc_name: Document name
            start_idx: Start page index
            end_idx: End page index
            total_pages: Total number of pages
            previous_summary: Previous summary content
            batch_num: Current batch number
            total_batches: Total number of batches
            
        Returns:
            System prompt string
        """
        prompt = f"""
        You are an expert document analyst. Please analyze pages {start_idx + 1}-{end_idx} 
        of the following document and generate a comprehensive summary.
        
        Document Name: {doc_name}
        Current Processing: Pages {start_idx + 1}-{end_idx} (Total pages: {total_pages})
        """
        
        if previous_summary:
            prompt += f"""
            
            Previous Summary Content (Pages 1-{start_idx}):
            {previous_summary}
            
            Please continue analyzing pages {start_idx + 1}-{end_idx} and build upon 
            the previous summary to create a coherent overall summary.
            """
        else:
            prompt += f"""
            
            Please analyze pages {start_idx + 1}-{end_idx} and start building 
            a comprehensive document summary.
            """
        
        if batch_num == total_batches - 1:  # Final batch
            prompt += f"""
            
            This is the final part of the document. Please generate a complete, 
            coherent document summary that integrates all the content from pages 1-{total_pages}. 
            The summary should be:
            
            1. **Comprehensive**: Cover all major themes, objectives, methods, findings, and conclusions
            2. **Coherent**: Flow logically from one section to the next without page-by-page breakdown
            3. **Structured**: Organize information by main topics rather than page numbers
            4. **Insightful**: Highlight key contributions, innovations, and significance
            5. **Complete**: Provide a unified overview of the entire document
            
            Generate a single, unified document summary:
            """
        else:
            prompt += f"""
            
            Please analyze pages {start_idx + 1}-{end_idx} and provide insights that 
            will help build a comprehensive summary. Focus on:
            1. Key themes and concepts introduced
            2. Important methods or approaches described
            3. Significant findings or results presented
            4. How this content relates to the overall document structure
            
            Provide analysis for integration into the final summary:
            """
        
        return prompt
    
    def _build_structure_system_prompt(self, doc_name: str, start_idx: int, 
                                     end_idx: int, total_pages: int, 
                                     previous_structure: str, batch_num: int, 
                                     total_batches: int) -> str:
        """Build system prompt for structural analysis.
        
        Args:
            doc_name: Document name
            start_idx: Start page index
            end_idx: End page index
            total_pages: Total number of pages
            previous_structure: Previous structural analysis
            batch_num: Current batch number
            total_batches: Total number of batches
            
        Returns:
            System prompt string
        """
        prompt = f"""
        You are an expert document structure analyst. Please analyze the organizational 
        structure and content hierarchy of pages {start_idx + 1}-{end_idx} of the following document.
        
        Document Name: {doc_name}
        Current Processing: Pages {start_idx + 1}-{end_idx} (Total pages: {total_pages})
        """
        
        if previous_structure:
            prompt += f"""
            
            Previous Structural Analysis (Pages 1-{start_idx}):
            {previous_structure}
            
            Please continue analyzing the structure of pages {start_idx + 1}-{end_idx} 
            and build upon the previous analysis to create a coherent overall structural understanding.
            """
        else:
            prompt += f"""
            
            Please analyze the structure of pages {start_idx + 1}-{end_idx} and start 
            building a comprehensive structural analysis of the document.
            """
        
        if batch_num == total_batches - 1:  # Final batch
            prompt += f"""
            
            This is the final part of the document. Please generate a complete, 
            coherent structural analysis that integrates all the content from pages 1-{total_pages}. 
            The structural analysis should be:
            
            1. **Comprehensive**: Cover the entire document's organizational structure, 
               content hierarchy, and logical flow
            2. **Coherent**: Present a unified structural overview without page-by-page breakdown
            3. **Analytical**: Identify key concepts, terminology, argument structures, 
               and reasoning processes
            4. **Visual**: Describe how charts, data, and visual elements are organized
            5. **Architectural**: Provide a complete understanding of the document's overall architecture
            
            Generate a single, unified structural analysis covering:
            - Overall document structure and organization
            - Content hierarchy and logical relationships
            - Key concepts and terminology system
            - Argument structure and reasoning processes
            - Organization of visual elements and data
            - Complete document architecture
            
            Generate a single, unified structural analysis:
            """
        else:
            prompt += f"""
            
            Please analyze the structure of pages {start_idx + 1}-{end_idx} and provide 
            structural insights that will help build a comprehensive analysis. Focus on:
            1. Organizational patterns and section structures
            2. Content hierarchy and relationships
            3. Key concepts and terminology introduced
            4. Logical connections with previous content
            5. Visual and data organization patterns
            
            Provide structural analysis for integration into the final comprehensive analysis:
            """
        
        return prompt
    
    def _build_summary_user_content(self, current_batch: List[Dict], 
                                  start_idx: int, end_idx: int, 
                                  doc_name: str) -> List[Dict]:
        """Build user content for summary generation.
        
        Args:
            current_batch: Current batch of pages
            start_idx: Start page index
            end_idx: End page index
            doc_name: Document name
            
        Returns:
            User content list with text and images
        """
        user_content = [
            {
                "type": "text", 
                "text": f"These are the images for pages {start_idx + 1}-{end_idx} "
                       f"of the document '{doc_name}'. Please analyze both the text "
                       f"content (provided in system message) and these visual images "
                       f"to contribute to a comprehensive document summary."
            }
        ]
        
        # Add images to user content
        for page_info in current_batch:
            page_image_path = page_info.get('page_image_path', '')
            
            if page_image_path and Path(page_image_path).exists():
                try:
                    with open(page_image_path, "rb") as image_file:
                        import base64
                        image_data = base64.b64encode(image_file.read()).decode('utf-8')
                        
                        user_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}",
                                "detail": "high"
                            }
                        })
                except Exception as e:
                    logger.warning(f"Failed to load image for page {page_info['page_number']}: {e}")
        
        return user_content
    
    def _build_structure_user_content(self, current_batch: List[Dict], 
                                    start_idx: int, end_idx: int, 
                                    doc_name: str) -> List[Dict]:
        """Build user content for structural analysis.
        
        Args:
            current_batch: Current batch of pages
            start_idx: Start page index
            end_idx: End page index
            doc_name: Document name
            
        Returns:
            User content list with text and images
        """
        user_content = [
            {
                "type": "text", 
                "text": f"These are the images for pages {start_idx + 1}-{end_idx} "
                       f"of the document '{doc_name}'. Please analyze both the text "
                       f"content (provided in system message) and these visual images "
                       f"to contribute to a comprehensive structural analysis."
            }
        ]
        
        # Add images to user content
        for page_info in current_batch:
            page_image_path = page_info.get('page_image_path', '')
            
            if page_image_path and Path(page_image_path).exists():
                try:
                    with open(page_image_path, "rb") as image_file:
                        import base64
                        image_data = base64.b64encode(image_file.read()).decode('utf-8')
                        
                        user_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}",
                                "detail": "high"
                            }
                        })
                except Exception as e:
                    logger.warning(f"Failed to load image for page {page_info['page_number']}: {e}")
        
        return user_content


class VectorStoreManager:
    """Manages vector store operations for summaries and chunks only."""
    
    def __init__(self, openai_api_key: str):
        """Initialize vector store manager.
        
        Args:
            openai_api_key: OpenAI API key
        """
        self.openai_api_key = openai_api_key
        self.client = OpenAI(api_key=openai_api_key)
        self.vector_store_registry = VectorStoreRegistry()
        litellm.vector_store_registry = self.vector_store_registry
        
        # Vector store IDs (only summaries and chunks)
        self.summaries_vector_store_id = None
        self.chunks_vector_store_id = None
        
        # File IDs for tracking
        self.file_ids = []
    
    async def create_vector_stores(self, documents_info: Dict[str, Dict]) -> Dict[str, str]:
        """Create vector stores for summaries and chunks only.
        
        Args:
            documents_info: Dictionary containing document information
            
        Returns:
            Dictionary mapping store types to their IDs
        """
        logger.info("Starting vector store creation...")
        
        vector_store_ids = {}
        
        try:
            # Create summaries vector store
            summaries_file_ids = await self._create_summaries_files(documents_info)
            if summaries_file_ids:
                logger.info(f"Creating summaries vector store with {len(summaries_file_ids)} files...")
                
                async def create_summaries_store():
                    return await acreate(
                        name="Document Summaries Vector Store",
                        file_ids=summaries_file_ids,
                        custom_llm_provider="openai"
                    )
                
                summaries_store = await retry_with_backoff(create_summaries_store, max_retries=3)
                self.summaries_vector_store_id = summaries_store["id"]
                vector_store_ids["summaries"] = summaries_store["id"]
                logger.info(f"Summaries vector store created successfully, "
                           f"ID: {summaries_store['id']}")
            
            # Create chunks vector store
            chunks_file_ids = await self._create_chunks_files(documents_info)
            if chunks_file_ids:
                logger.info(f"Creating chunks vector store with {len(chunks_file_ids)} files...")
                
                # Check if we have too many files
                if len(chunks_file_ids) > 100:
                    logger.warning(f"Large number of chunks ({len(chunks_file_ids)}). "
                                 f"This may cause API issues. Consider reducing chunk size.")
                
                async def create_chunks_store():
                    return await acreate(
                        name="Document Chunks Vector Store",
                        file_ids=chunks_file_ids,
                        custom_llm_provider="openai"
                    )
                
                chunks_store = await retry_with_backoff(create_chunks_store, max_retries=3)
                self.chunks_vector_store_id = chunks_store["id"]
                vector_store_ids["chunks"] = chunks_store["id"]
                logger.info(f"Chunks vector store created successfully, "
                           f"ID: {chunks_store['id']} (files: {len(chunks_file_ids)})")
            
            # Register vector stores
            self._register_vector_stores()
            
            return vector_store_ids
            
        except Exception as e:
            logger.error(f"Failed to create vector stores: {e}")
            
            # Provide specific error messages
            if "server_error" in str(e) or "500" in str(e):
                logger.error("OpenAI server error. This may be temporary. Please try again later.")
                logger.error("Consider:")
                logger.error("1. Reducing the number of documents")
                logger.error("2. Reducing chunk size")
                logger.error("3. Waiting a few minutes before retrying")
            elif "rate_limit" in str(e) or "429" in str(e):
                logger.error("Rate limit exceeded. Please wait before retrying.")
            elif "quota" in str(e):
                logger.error("API quota exceeded. Please check your OpenAI account.")
            
            raise
    
    async def _create_summaries_files(self, documents_info: Dict[str, Dict]) -> List[str]:
        """Create and upload summary files.
        
        Args:
            documents_info: Dictionary containing document information
            
        Returns:
            List of uploaded file IDs
        """
        file_ids = []
        logger.info(f"Creating {len(documents_info)} summary files...")
        
        for doc_id, doc_info in documents_info.items():
            try:
                # Create summary file content
                summary_content = (f"DOCUMENT_ID: {doc_id}\n"
                                 f"文档名称: {doc_info['doc_name']}\n\n"
                                 f"摘要:\n{doc_info['summary']}")
                
                # Validate and truncate content if necessary
                if not validate_file_size(summary_content):
                    logger.warning(f"Summary for {doc_id} is too large, truncating...")
                    summary_content = truncate_content(summary_content)
                
                # Save as temporary file
                temp_file = Path(f"summary_{doc_id}_{doc_info['doc_name']}.txt")
                with open(temp_file, "w", encoding="utf-8") as f:
                    f.write(summary_content)
                
                # Upload file with retry
                async def upload_file():
                    with open(temp_file, "rb") as file:
                        response = self.client.files.create(
                            file=file,
                            purpose="assistants"
                        )
                        return response.id
                
                try:
                    file_id = await retry_with_backoff(upload_file, max_retries=2)
                    file_ids.append(file_id)
                    
                    # Add small delay to avoid rate limiting
                    await asyncio.sleep(0.2)
                    
                except Exception as e:
                    logger.error(f"Failed to upload summary for {doc_id}: {e}")
                    continue
                finally:
                    # Clean up temporary file
                    if temp_file.exists():
                        temp_file.unlink()
                
            except Exception as e:
                logger.error(f"Failed to create summary file for {doc_id}: {e}")
        
        logger.info(f"Successfully created {len(file_ids)} summary files")
        return file_ids
    
    async def _create_chunks_files(self, documents_info: Dict[str, Dict]) -> List[str]:
        """Create and upload chunk files.
        
        Args:
            documents_info: Dictionary containing document information
            
        Returns:
            List of uploaded file IDs
        """
        file_ids = []
        total_chunks = sum(len(doc_info['chunks']) for doc_info in documents_info.values())
        logger.info(f"Creating {total_chunks} chunk files...")
        
        for doc_id, doc_info in documents_info.items():
            try:
                # Create separate file for each chunk
                for i, chunk in enumerate(doc_info['chunks']):
                    # Create chunk file content
                    chunk_content = (f"DOCUMENT_ID: {doc_id}\n"
                                   f"文档名称: {doc_info['doc_name']}\n"
                                   f"Chunk索引: {i+1}\n\n"
                                   f"内容:\n{chunk}")
                    
                    # Validate and truncate content if necessary
                    if not validate_file_size(chunk_content):
                        logger.warning(f"Chunk {i+1} for {doc_id} is too large, truncating...")
                        chunk_content = truncate_content(chunk_content)
                    
                    # Save as temporary file
                    temp_file = Path(f"chunk_{doc_id}_{doc_info['doc_name']}_{i+1}.txt")
                    with open(temp_file, "w", encoding="utf-8") as f:
                        f.write(chunk_content)
                    
                    # Upload file with retry
                    async def upload_file():
                        with open(temp_file, "rb") as file:
                            response = self.client.files.create(
                                file=file,
                                purpose="assistants"
                            )
                            return response.id
                    
                    try:
                        file_id = await retry_with_backoff(upload_file, max_retries=2)
                        file_ids.append(file_id)
                        
                        # Add small delay to avoid rate limiting
                        if len(file_ids) % 10 == 0:
                            await asyncio.sleep(0.5)
                            
                    except Exception as e:
                        logger.error(f"Failed to upload chunk {i+1} for {doc_id}: {e}")
                        continue
                    finally:
                        # Clean up temporary file
                        if temp_file.exists():
                            temp_file.unlink()
                
            except Exception as e:
                logger.error(f"Failed to create chunks files for {doc_id}: {e}")
        
        logger.info(f"Successfully created {len(file_ids)} chunk files")
        return file_ids
    
    def _register_vector_stores(self):
        """Register vector stores with LiteLLM."""
        if self.summaries_vector_store_id:
            summaries_store = LiteLLM_ManagedVectorStore(
                vector_store_id=self.summaries_vector_store_id,
                custom_llm_provider="openai",
                vector_store_name="Document Summaries KB",
                vector_store_description="Knowledge base containing document summaries"
            )
            self.vector_store_registry.add_vector_store_to_registry(summaries_store)
        
        if self.chunks_vector_store_id:
            chunks_store = LiteLLM_ManagedVectorStore(
                vector_store_id=self.chunks_vector_store_id,
                custom_llm_provider="openai",
                vector_store_name="Document Chunks KB",
                vector_store_description="Knowledge base containing document chunks"
            )
            self.vector_store_registry.add_vector_store_to_registry(chunks_store)
        
        logger.info("Vector stores registered with LiteLLM")
    
    def get_vector_store_ids(self) -> Dict[str, Optional[str]]:
        """Get vector store IDs.
        
        Returns:
            Dictionary mapping store types to their IDs
        """
        return {
            "summaries": self.summaries_vector_store_id,
            "chunks": self.chunks_vector_store_id
        }


class SummaryFileManager:
    """Manages document summary files saved locally."""
    
    def __init__(self, summary_directory: Path):
        """Initialize summary file manager.
        
        Args:
            summary_directory: Directory to store summary files
        """
        self.summary_directory = summary_directory
        self.summary_directory.mkdir(exist_ok=True)
    
    def save_document_summary(self, doc_id: str, doc_name: str, 
                            summary: str) -> str:
        """Save document summary to local file.
        
        Args:
            doc_id: Document ID
            doc_name: Document name
            summary: Document summary content
            
        Returns:
            Path to saved summary file
        """
        try:
            # Create summary file
            summary_filename = f"summary_{doc_id}_{doc_name}.json"
            summary_path = self.summary_directory / summary_filename
            
            # Save summary as JSON
            summary_data = {
                "doc_id": doc_id,
                "doc_name": doc_name,
                "summary": summary,
                "file_path": str(summary_path), # Store file path for loading
                "created_at": str(Path().stat().st_mtime)
            }
            
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Document summary saved: {summary_path}")
            return str(summary_path)
            
        except Exception as e:
            logger.error(f"Failed to save document summary for {doc_id}: {e}")
            return ""
    
    def load_document_summary(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Load document summary from local file.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Summary data dictionary or None if not found
        """
        try:
            # Find summary file by doc_id
            for summary_file in self.summary_directory.glob(f"summary_{doc_id}_*.json"):
                with open(summary_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to load document summary for {doc_id}: {e}")
            return None
    
    def get_all_summary_files(self) -> List[Path]:
        """Get all summary files.
        
        Returns:
            List of summary file paths
        """
        return list(self.summary_directory.glob("summary_*.json"))


class StructureFileManager:
    """Manages document structure files saved locally."""
    
    def __init__(self, structure_directory: Path):
        """Initialize structure file manager.
        
        Args:
            structure_directory: Directory to store structure files
        """
        self.structure_directory = structure_directory
        self.structure_directory.mkdir(exist_ok=True)
    
    def save_document_structure(self, doc_id: str, doc_name: str, 
                              structure: str) -> str:
        """Save document structure to local file.
        
        Args:
            doc_id: Document ID
            doc_name: Document name
            structure: Document structure content
            
        Returns:
            Path to saved structure file
        """
        try:
            # Create structure file
            structure_filename = f"structure_{doc_id}_{doc_name}.json"
            structure_path = self.structure_directory / structure_filename
            
            # Save structure as JSON
            structure_data = {
                "doc_id": doc_id,
                "doc_name": doc_name,
                "structure": structure,
                "file_path": str(structure_path), # Store file path for loading
                "created_at": str(Path().stat().st_mtime)
            }
            
            with open(structure_path, "w", encoding="utf-8") as f:
                json.dump(structure_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Document structure saved: {structure_path}")
            return str(structure_path)
            
        except Exception as e:
            logger.error(f"Failed to save document structure for {doc_id}: {e}")
            return ""
    
    def load_document_structure(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Load document structure from local file.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Structure data dictionary or None if not found
        """
        try:
            # Find structure file by doc_id
            for structure_file in self.structure_directory.glob(f"structure_{doc_id}_*.json"):
                with open(structure_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to load document structure for {doc_id}: {e}")
            return None
    
    def get_all_structure_files(self) -> List[Path]:
        """Get all structure files.
        
        Returns:
            List of structure file paths
        """
        return list(self.structure_directory.glob("structure_*.json"))


class DocumentProcessor:
    """Main document processing class that orchestrates all processing steps."""
    
    def __init__(self, openai_api_key: str, documents_directory: str = "documents"):
        """Initialize document processor.
        
        Args:
            openai_api_key: OpenAI API key
            documents_directory: Directory containing document files
        """
        self.openai_api_key = openai_api_key
        self.documents_directory = Path(documents_directory)
        self.documents_directory.mkdir(exist_ok=True)
        self.model_name = "gpt-4.1"
        
        # Initialize components
        self.images_directory = Path("document_images")
        self.structure_directory = Path("pdf_structure")
        self.summary_directory = Path("pdf_summary")
        self.content_extractor = PDFContentExtractor(self.images_directory)
        self.text_chunker = TextChunker(chunk_size=4000, overlap=500, max_chunks_per_doc=50)
        self.document_analyzer = DocumentAnalyzer(openai_api_key)
        self.vector_store_manager = VectorStoreManager(openai_api_key)
        self.structure_file_manager = StructureFileManager(self.structure_directory)
        self.summary_file_manager = SummaryFileManager(self.summary_directory)
        
        # Document information storage
        self.documents_info = {}
        self.documents_metadata = {}
    
    def generate_document_id(self, file_path: Path) -> str:
        """Generate unique document ID.
        
        Args:
            file_path: Document file path
            
        Returns:
            Unique document ID
        """
        content = f"{file_path.absolute()}_{file_path.stat().st_mtime}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def prepare_document_metadata(self, doc_path: Path, doc_id: str, 
                                content: DocumentContent) -> DocumentMetadata:
        """Prepare document metadata.
        
        Args:
            doc_path: Document file path
            doc_id: Document ID
            content: Extracted document content
            
        Returns:
            Document metadata object
        """
        return DocumentMetadata(
            doc_id=doc_id,
            doc_name=doc_path.stem,
            file_extension=doc_path.suffix,
            file_size=doc_path.stat().st_size,
            upload_time=str(Path().stat().st_mtime),
            page_count=len(content.pages),
            text_length=len(content.text),
            image_count=len(content.page_images)
        )
    
    async def process_documents(self) -> Dict[str, Dict]:
        """Process all documents in the directory.
        
        Returns:
            Dictionary containing processed document information
        """
        logger.info("Starting document processing...")
        
        # Get all PDF files
        pdf_files = list(self.documents_directory.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.documents_directory}")
            return {}
        
        documents_info = {}
        processed_count = 0
        skipped_count = 0
        
        for i, pdf_path in enumerate(pdf_files, 1):
            try:
                logger.info(f"Processing document {i}/{len(pdf_files)}: {pdf_path.name}")
                
                # Generate document ID
                doc_id = self.generate_document_id(pdf_path)
                
                # Check if document has already been processed (only check local files)
                if self.check_document_processed(doc_id):
                    logger.info(f"Document {pdf_path.name} (ID: {doc_id}) already processed. Loading existing data.")
                    skipped_count += 1
                    
                    # Create document info from existing files
                    doc_info = self._create_document_info_from_files(doc_id, pdf_path)
                    if doc_info:
                        documents_info[doc_id] = doc_info
                    continue
                
                # Extract content
                content = self.content_extractor.extract_content(pdf_path)
                
                # Prepare page data for analysis
                pages_data = []
                for page_info in content.pages:
                    pages_data.append({
                        "page_number": page_info["page_number"],
                        "text": page_info["text"],
                        "page_image_path": page_info["page_image_path"]
                    })
                
                # Check if summary already exists
                existing_summary = self.summary_file_manager.load_document_summary(doc_id)
                if existing_summary:
                    logger.info(f"Loading existing summary for {pdf_path.name} (ID: {doc_id})")
                    content.summary = existing_summary["summary"]
                    summary_file_path = existing_summary.get("file_path", "")
                else:
                    # Generate summary and structural analysis
                    logger.info("Generating document summary...")
                    content.summary = await self.document_analyzer.generate_document_summary(
                        pages_data, pdf_path.stem, model_name=self.model_name
                    )
                    
                    # Save summary to local file
                    summary_file_path = self.summary_file_manager.save_document_summary(
                        doc_id, pdf_path.stem, content.summary
                    )
                
                # Check if structure already exists
                existing_structure = self.structure_file_manager.load_document_structure(doc_id)
                if existing_structure:
                    logger.info(f"Loading existing structure for {pdf_path.name} (ID: {doc_id})")
                    content.structure = existing_structure["structure"]
                    structure_file_path = existing_structure.get("file_path", "")
                else:
                    logger.info("Generating document structure...")
                    content.structure = await self.document_analyzer.generate_document_structure(
                        pages_data, pdf_path.stem, model_name=self.model_name
                    )
                    
                    # Save structure to local file
                    structure_file_path = self.structure_file_manager.save_document_structure(
                        doc_id, pdf_path.stem, content.structure
                    )
                
                # Chunk text
                if len(content.text) > 100000:  # Large document
                    content.chunks = self.text_chunker.adjust_chunk_size_for_large_text(content.text)
                else:
                    content.chunks = self.text_chunker.chunk_text(content.text)
                
                # Prepare metadata
                metadata = self.prepare_document_metadata(pdf_path, doc_id, content)
                
                # Store document information (without structure in memory)
                documents_info[doc_id] = {
                    "doc_path": str(pdf_path),
                    "doc_name": pdf_path.stem,
                    "summary": content.summary,
                    "summary_file_path": summary_file_path,
                    "structure_file_path": structure_file_path,
                    "chunks": content.chunks,
                    "pages": content.pages,
                    "page_images": content.page_images,
                    "metadata": metadata.__dict__
                }
                
                self.documents_metadata[doc_id] = metadata
                processed_count += 1
                
                logger.info(f"Document processing completed: {pdf_path.name} (ID: {doc_id})")
                
            except Exception as e:
                logger.error(f"Failed to process document {pdf_path.name}: {e}")
        
        self.documents_info = documents_info
        logger.info(f"Document processing completed: {processed_count} new documents, {skipped_count} skipped")
        return documents_info
    
    def _load_existing_documents(self):
        """Load existing documents from saved configuration."""
        config_file = "document_processor_config.json"
        if not Path(config_file).exists():
            return
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
                # Load documents info
                self.documents_info = config.get("documents_info", {})
                
                # Load metadata
                metadata_dict = config.get("documents_metadata", {})
                self.documents_metadata = {}
                for doc_id, metadata_data in metadata_dict.items():
                    self.documents_metadata[doc_id] = DocumentMetadata(**metadata_data)
                
                logger.info(f"Loaded {len(self.documents_info)} existing documents from config")
                
        except Exception as e:
            logger.warning(f"Failed to load existing documents: {e}")
    
    async def create_vector_stores(self) -> Dict[str, str]:
        """Create vector stores for processed documents.
        
        Returns:
            Dictionary mapping store types to their IDs
        """
        if not self.documents_info:
            logger.warning("No documents processed, cannot create vector stores")
            return {}
        
        return await self.vector_store_manager.create_vector_stores(self.documents_info)
    
    def save_config(self, config_file: str = "document_processor_config.json"):
        """Save configuration to file.
        
        Args:
            config_file: Configuration file path
        """
        config = {
            "vector_store_ids": self.vector_store_manager.get_vector_store_ids(),
            "file_ids": self.vector_store_manager.file_ids,
            "documents_directory": str(self.documents_directory),
            "structure_directory": str(self.structure_directory),
            "summary_directory": str(self.summary_directory),
            "documents_info": self.documents_info,
            "documents_metadata": {
                doc_id: metadata.__dict__ 
                for doc_id, metadata in self.documents_metadata.items()
            },
            "openai_api_key": self.openai_api_key[:10] + "..."  # Partially hidden
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Configuration saved to: {config_file}")
    
    def load_config(self, config_file: str = "document_processor_config.json") -> bool:
        """Load configuration from file.
        
        Args:
            config_file: Configuration file path
            
        Returns:
            True if configuration loaded successfully, False otherwise
        """
        if not Path(config_file).exists():
            logger.warning(f"Configuration file not found: {config_file}")
            return False
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Load vector store IDs
            vector_store_ids = config.get("vector_store_ids", {})
            self.vector_store_manager.summaries_vector_store_id = vector_store_ids.get("summaries")
            self.vector_store_manager.chunks_vector_store_id = vector_store_ids.get("chunks")
            
            # Load other data
            self.vector_store_manager.file_ids = config.get("file_ids", [])
            self.documents_info = config.get("documents_info", {})
            
            # Load metadata
            metadata_dict = config.get("documents_metadata", {})
            self.documents_metadata = {}
            for doc_id, metadata_data in metadata_dict.items():
                self.documents_metadata[doc_id] = DocumentMetadata(**metadata_data)
            
            # Register vector stores if they exist
            if any([self.vector_store_manager.summaries_vector_store_id,
                   self.vector_store_manager.chunks_vector_store_id]):
                self.vector_store_manager._register_vector_stores()
                logger.info(f"Configuration loaded, vector store IDs: "
                           f"{self.vector_store_manager.get_vector_store_ids()}")
                return True
            
            # Return True if we loaded any documents, regardless of vector stores
            if self.documents_info:
                logger.info(f"Configuration loaded successfully with {len(self.documents_info)} documents")
                return True
            else:
                logger.warning("Configuration loaded but no documents found")
                return False
        
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
        
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get processor status.
        
        Returns:
            Status information dictionary
        """
        return {
            "vector_store_ids": self.vector_store_manager.get_vector_store_ids(),
            "file_count": len(self.vector_store_manager.file_ids),
            "documents_count": len(self.documents_info),
            "documents_directory": str(self.documents_directory),
            "structure_directory": str(self.structure_directory),
            "summary_directory": str(self.summary_directory),
            "documents_available": len(list(self.documents_directory.glob("*.pdf"))),
            "structure_files": len(self.structure_file_manager.get_all_structure_files()),
            "summary_files": len(self.summary_file_manager.get_all_summary_files()),
            "is_configured": any(self.vector_store_manager.get_vector_store_ids().values())
        }
    
    def get_processing_status(self) -> Dict[str, Any]:
        """Get detailed processing status.
        
        Returns:
            Status information dictionary
        """
        # Get all PDF files
        pdf_files = list(self.documents_directory.glob("*.pdf"))
        total_files = len(pdf_files)
        
        # Get processed documents
        processed_docs = self.get_processed_documents()
        processed_count = len(processed_docs)
        
        # Get documents in memory
        in_memory_count = len(self.documents_info)
        
        # Calculate new documents to process
        new_docs = []
        for pdf_path in pdf_files:
            doc_id = self.generate_document_id(pdf_path)
            if doc_id not in processed_docs:
                new_docs.append(pdf_path.name)
        
        return {
            "total_pdf_files": total_files,
            "processed_documents": processed_count,
            "documents_in_memory": in_memory_count,
            "new_documents_to_process": len(new_docs),
            "new_document_names": new_docs,
            "processed_document_ids": processed_docs
        }
    
    async def close(self):
        """Close connections and clean up resources."""
        try:
            if hasattr(self.vector_store_manager, 'client') and self.vector_store_manager.client:
                await self.vector_store_manager.client.close()
                logger.info("OpenAI client closed")
        except Exception as e:
            logger.warning(f"Warning during client closure: {e}")

    def check_document_processed(self, doc_id: str) -> bool:
        """Check if a document has already been processed.
        
        Args:
            doc_id: Document ID to check
            
        Returns:
            True if document has been processed, False otherwise
        """
        # Check if structure file exists
        structure_data = self.structure_file_manager.load_document_structure(doc_id)
        if structure_data:
            return True
        
        # Check if summary file exists
        summary_data = self.summary_file_manager.load_document_summary(doc_id)
        if summary_data:
            return True
        
        # Check if document info exists in memory
        if doc_id in self.documents_info:
            return True
        
        return False
    
    def get_processed_documents(self) -> List[str]:
        """Get list of already processed document IDs.
        
        Returns:
            List of processed document IDs
        """
        processed_docs = []
        
        # Check structure files
        structure_files = self.structure_file_manager.get_all_structure_files()
        for structure_file in structure_files:
            try:
                with open(structure_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    doc_id = data.get('doc_id')
                    if doc_id:
                        processed_docs.append(doc_id)
            except Exception as e:
                logger.warning(f"Failed to read structure file {structure_file}: {e}")
        
        # Check summary files
        summary_files = self.summary_file_manager.get_all_summary_files()
        for summary_file in summary_files:
            try:
                with open(summary_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    doc_id = data.get('doc_id')
                    if doc_id:
                        processed_docs.append(doc_id)
            except Exception as e:
                logger.warning(f"Failed to read summary file {summary_file}: {e}")
        
        return list(set(processed_docs))  # Remove duplicates
    
    def load_existing_document_info(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Load existing document information from saved config.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document information dictionary or None if not found
        """
        config_file = "document_processor_config.json"
        if not Path(config_file).exists():
            return None
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                documents_info = config.get("documents_info", {})
                return documents_info.get(doc_id)
        except Exception as e:
            logger.warning(f"Failed to load existing document info for {doc_id}: {e}")
            return None

    def _create_document_info_from_files(self, doc_id: str, pdf_path: Path) -> Optional[Dict[str, Any]]:
        """Create document information from existing summary and structure files.
        
        Args:
            doc_id: Document ID
            pdf_path: Path to the PDF file
            
        Returns:
            Document information dictionary or None if files not found
        """
        summary_file = self.summary_file_manager.load_document_summary(doc_id)
        if not summary_file:
            logger.warning(f"Summary file not found for {doc_id}: {pdf_path.name}")
            return None
        
        structure_file = self.structure_file_manager.load_document_structure(doc_id)
        if not structure_file:
            logger.warning(f"Structure file not found for {doc_id}: {pdf_path.name}")
            return None
        
        # Load summary and structure content
        summary_content = summary_file["summary"]
        structure_content = structure_file["structure"]
        
        # Re-extract content from PDF for chunks and pages
        logger.info(f"Re-extracting content from PDF for {pdf_path.name} (ID: {doc_id})")
        content = self.content_extractor.extract_content(pdf_path)
        
        # Use existing summary and structure
        content.summary = summary_content
        content.structure = structure_content
        
        # Generate chunks from extracted text
        if len(content.text) > 100000:  # Large document
            content.chunks = self.text_chunker.adjust_chunk_size_for_large_text(content.text)
        else:
            content.chunks = self.text_chunker.chunk_text(content.text)
        
        # Prepare metadata
        metadata = self.prepare_document_metadata(pdf_path, doc_id, content)
        
        # Store metadata in memory
        self.documents_metadata[doc_id] = metadata
        
        return {
            "doc_path": str(pdf_path),
            "doc_name": pdf_path.stem,
            "summary": summary_content,
            "summary_file_path": summary_file["file_path"],
            "structure_file_path": structure_file["file_path"],
            "chunks": content.chunks,
            "pages": content.pages,
            "page_images": content.page_images,
            "metadata": metadata.__dict__
        }


async def main():
    """Main function for testing document processor."""
    print("Document Processor Module")
    print("=" * 50)
    
    # Get OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        openai_api_key = input("Enter your OpenAI API key: ").strip()
        if not openai_api_key:
            print("Error: OpenAI API key required")
            return
    
    # Configuration options
    documents_directory = "multi_llm_pdf"
    chunk_size = 4000
    max_chunks_per_doc = 50
    
    # Create document processor
    processor = DocumentProcessor(openai_api_key, documents_directory=documents_directory)
    
    # Update chunker settings
    processor.text_chunker = TextChunker(
        chunk_size=chunk_size,
        overlap=500,
        max_chunks_per_doc=max_chunks_per_doc
    )
    
    # Show processing status
    status = processor.get_processing_status()
    print(f"\n📊 Processing Status:")
    print(f"   Total PDF files: {status['total_pdf_files']}")
    print(f"   Already processed: {status['processed_documents']}")
    print(f"   New documents to process: {status['new_documents_to_process']}")
    
    if status['new_document_names']:
        print(f"   New documents: {', '.join(status['new_document_names'])}")
    
    # Check document files
    if status['total_pdf_files'] == 0:
        print(f"Please place PDF files in {processor.documents_directory} directory")
        return
    
    # Check if we have new documents to process
    if status['new_documents_to_process'] == 0:
        print("✅ All documents already processed!")
        print("You can proceed to create vector stores or use the documents.")
        
        # Ask user what to do
        print("\nOptions:")
        print("1. Create/update vector stores")
        print("2. Show status")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1/2/3): ").strip()
        
        if choice == "1":
            print("\nCreating vector stores...")
            try:
                await processor.create_vector_stores()
                processor.save_config()
                print("✅ Vector stores created successfully!")
            except Exception as e:
                print(f"❌ Vector store creation failed: {e}")
        elif choice == "2":
            final_status = processor.get_status()
            print(f"\nFinal Status:")
            print(f"- Documents: {final_status['documents_count']}")
            print(f"- Available: {final_status['documents_available']}")
            print(f"- Structure files: {final_status['structure_files']}")
            print(f"- Summary files: {final_status['summary_files']}")
            print(f"- Configured: {final_status['is_configured']}")
        else:
            print("Exiting...")
            return
    else:
        print(f"\n🔄 Processing {status['new_documents_to_process']} new documents...")
        
        start_time = time.time()
        
        print("Processing documents...")
        await processor.process_documents()
        
        # Create vector stores
        print("Creating vector stores...")
        try:
            await processor.create_vector_stores()
        except Exception as e:
            print(f"Vector store creation failed: {e}")
            print("\nTroubleshooting tips:")
            print("1. Check your OpenAI API quota")
            print("2. Try reducing chunk size or max chunks per document")
            print("3. Wait a few minutes and try again")
            print("4. Check if OpenAI services are experiencing issues")
            return
        
        # Save configuration
        processor.save_config()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Document processing completed in {elapsed_time:.2f} seconds")
    
    # Display final status
    final_status = processor.get_status()
    print(f"\nFinal Status:")
    print(f"- Documents: {final_status['documents_count']}")
    print(f"- Available: {final_status['documents_available']}")
    print(f"- Structure files: {final_status['structure_files']}")
    print(f"- Summary files: {final_status['summary_files']}")
    print(f"- Configured: {final_status['is_configured']}")


if __name__ == "__main__":
    asyncio.run(main())
    
