"""
Chunk summarizer module for generating concise summaries of text chunks.

This module provides functionality to summarize individual chunks or batches
of chunks using LLM-based summarization.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI library not installed. Install with: pip install openai")

logger = logging.getLogger(__name__)


class ChunkSummary(BaseModel):
    """Summary data for a chunk."""
    
    chunk_id: str = Field(description="Unique chunk identifier")
    summary: str = Field(description="Generated summary")
    section_title: Optional[str] = Field(default=None, description="Section title for context")
    references: List[str] = Field(default_factory=list, description="Table/image references")
    usage: Dict[str, int] = Field(default_factory=dict, description="API usage statistics")


class ChunkSummarizer:
    """
    Handles chunk summarization using LLM.
    
    Generates concise, informative summaries that capture the main topics,
    key points, and important relationships in each chunk.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.1,
        max_summary_tokens: int = 150
    ):
        """
        Initialize the chunk summarizer.
        
        Args:
            api_key: OpenAI API key
            model: Model to use for summarization
            temperature: Temperature for generation
            max_summary_tokens: Maximum tokens for each summary
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library required. Install with: pip install openai")
        
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key) if api_key else None
        self.model = model
        self.temperature = temperature
        self.max_summary_tokens = max_summary_tokens
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def summarize_chunk(
        self,
        content: str,
        chunk_id: str,
        section_title: Optional[str] = None,
        references: List[str] = None
    ) -> ChunkSummary:
        """
        Create a concise summary of a chunk's content.
        
        Args:
            content: Text content to summarize
            chunk_id: Unique chunk identifier
            section_title: Optional section title for context
            references: Optional list of table/image references
            
        Returns:
            ChunkSummary with generated summary and metadata
        """
        if not self.client:
            raise ValueError("OpenAI client not initialized. Provide API key.")
        
        # Build context-aware prompt
        context = f"This chunk is from section: '{section_title}'\n\n" if section_title else ""
        
        # Mention references if present
        if references:
            ref_types = []
            table_refs = [r for r in references if 'table' in r.lower()]
            image_refs = [r for r in references if 'image' in r.lower()]
            if table_refs:
                ref_types.append(f"{len(table_refs)} table(s)")
            if image_refs:
                ref_types.append(f"{len(image_refs)} image(s)")
            if ref_types:
                context += f"This chunk references: {', '.join(ref_types)}\n\n"
        
        prompt = f"""You are an expert at summarizing academic and technical content.

{context}TEXT TO SUMMARIZE:
{content[:4000]}

INSTRUCTIONS:
Create a concise, informative summary (2-3 sentences) that captures:
1. The main topic or concept being discussed
2. Key points, methods, or findings mentioned
3. Any important relationships or comparisons made
4. If tables or images are referenced, mention what they show

The summary should be self-contained and help someone quickly understand what this chunk discusses without reading the full text.

Focus on technical content and avoid generic descriptions."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise technical content summarizer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_summary_tokens
            )
            
            # Track usage for cost calculation
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            summary = response.choices[0].message.content.strip()
            
            return ChunkSummary(
                chunk_id=chunk_id,
                summary=summary,
                section_title=section_title,
                references=references or [],
                usage=usage
            )
            
        except Exception as e:
            self.logger.error(f"Chunk summarization failed for {chunk_id}: {e}")
            return ChunkSummary(
                chunk_id=chunk_id,
                summary="Summary generation failed",
                section_title=section_title,
                references=references or [],
                usage={}
            )
    
    def batch_summarize_chunks(
        self,
        chunks: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> List[ChunkSummary]:
        """
        Summarize multiple chunks in batch.
        
        Args:
            chunks: List of chunk dictionaries with content, id, etc.
            show_progress: Whether to log progress
            
        Returns:
            List of ChunkSummary objects
        """
        summaries = []
        total = len(chunks)
        
        for i, chunk in enumerate(chunks):
            if show_progress:
                self.logger.info(f"Summarizing chunk {i+1}/{total}")
            
            summary = self.summarize_chunk(
                content=chunk.get("content", ""),
                chunk_id=chunk.get("chunk_id", f"chunk_{i}"),
                section_title=chunk.get("section_title"),
                references=chunk.get("references", [])
            )
            summaries.append(summary)
        
        return summaries
    
    def create_fallback_summary(
        self,
        content: str,
        chunk_id: str
    ) -> ChunkSummary:
        """
        Create a basic summary without LLM (fallback method).
        
        Args:
            content: Text content
            chunk_id: Chunk identifier
            
        Returns:
            ChunkSummary with basic extraction
        """
        # Extract first and last sentences as basic summary
        sentences = content.split('. ')
        if len(sentences) > 2:
            summary = f"{sentences[0]}. ... {sentences[-1]}"
        else:
            summary = content[:200] + "..." if len(content) > 200 else content
        
        return ChunkSummary(
            chunk_id=chunk_id,
            summary=summary,
            usage={}
        )
    
    def get_total_usage(self, summaries: List[ChunkSummary]) -> Dict[str, int]:
        """
        Calculate total API usage from multiple summaries.
        
        Args:
            summaries: List of ChunkSummary objects
            
        Returns:
            Dictionary with total token usage
        """
        total = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        
        for summary in summaries:
            if summary.usage:
                total["prompt_tokens"] += summary.usage.get("prompt_tokens", 0)
                total["completion_tokens"] += summary.usage.get("completion_tokens", 0)
                total["total_tokens"] += summary.usage.get("total_tokens", 0)
        
        return total