#!/usr/bin/env python3
"""
Document Retriever Module

This module handles document retrieval operations using hierarchical search strategy.
It follows the Single Responsibility Principle by focusing solely on retrieval tasks.

The module implements:
1. Hierarchical search: Search summaries first, then filter chunks by doc_id
2. Summary similarity calculation using vector search
3. Chunk retrieval with document ID filtering
4. Result synthesis and formatting

Author: yujing.wang
Date: 2025.07.25
"""

import os
import json
import asyncio
from typing import List, Dict, Optional, Union, Tuple, Any
from pathlib import Path
import litellm
from litellm.vector_stores.main import asearch
import logging
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Search result data structure.
    
    Attributes:
        doc_id: Document ID
        content: Retrieved content
        score: Similarity score
        source_type: Type of content (summary, chunk)
        metadata: Additional metadata
    """
    doc_id: str
    content: str
    score: float
    source_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Complete retrieval result.
    
    Attributes:
        query: Original search query
        summary_results: Summary search results
        chunk_results: Chunk search results
        final_answer: Synthesized final answer
        search_stats: Search statistics
    """
    query: str
    summary_results: List[SearchResult] = field(default_factory=list)
    chunk_results: List[SearchResult] = field(default_factory=list)
    final_answer: str = ""
    search_stats: Dict[str, Any] = field(default_factory=dict)


class SearchStrategy(ABC):
    """Abstract base class for search strategies."""
    
    @abstractmethod
    async def search(self, query: str, **kwargs) -> List[SearchResult]:
        """Execute search strategy.
        
        Args:
            query: Search query
            **kwargs: Additional search parameters
            
        Returns:
            List of search results
        """
        pass


class VectorSearchStrategy(SearchStrategy):
    """Vector-based search strategy implementation."""
    
    def __init__(self, vector_store_id: str, 
                 custom_llm_provider: str = "openai",
                 similarity_threshold: float = 0.4):
        """Initialize vector search strategy.
        
        Args:
            vector_store_id: Vector store ID to search
            custom_llm_provider: LLM provider for search
        """
        self.vector_store_id = vector_store_id
        self.custom_llm_provider = custom_llm_provider
        self.similarity_threshold = similarity_threshold
    
    async def search(self, query: str, **kwargs) -> List[SearchResult]:
        """Execute vector search.
        
        Args:
            query: Search query
            **kwargs: Additional search parameters
            
        Returns:
            List of search results
        """
        try:
            logger.info(f"Executing vector search with query: {query[:50]}...")
            
            response = await asearch(
                vector_store_id=self.vector_store_id,
                query=query,
                custom_llm_provider=self.custom_llm_provider
            )
            

            results = []
            for result in response["data"]:
                # filter results by similarity score
                if result["score"] < self.similarity_threshold:
                    continue
                
                for content in result["content"]:
                    if content["type"] == "text":
                        # Extract document ID from text content
                        doc_id = self._extract_doc_id_from_text(content["text"])
                        print(f"score: {result['score']}, doc_id: {doc_id}")
                        
                        if doc_id:
                            search_result = SearchResult(
                                doc_id=doc_id,
                                content=content["text"],
                                score=result["score"],
                                source_type="vector_search",
                                metadata={
                                    "filename": result.get("filename", ""),
                                    "file_id": result.get("file_id", "")
                                }
                            )
                            results.append(search_result)            
            
            logger.info(f"Vector search completed: {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def _extract_doc_id_from_text(self, text: str) -> str:
        """Extract document ID from text content.
        
        Args:
            text: Text content containing document ID
            
        Returns:
            Document ID or empty string if not found
        """
        import re
        # Look for "DOCUMENT_ID: " followed by content
        match = re.search(r'DOCUMENT_ID:\s*([a-f0-9]+)', text)
        if match:
            return match.group(1)
        return ""


class TraditionalSearchStrategy(SearchStrategy):
    """Traditional search strategy using LLM-based content matching."""
    
    def __init__(self, documents_info: Dict[str, Dict], content_type: str):
        """Initialize traditional search strategy.
        
        Args:
            documents_info: Dictionary containing document information
            content_type: Type of content to search (summary, chunk)
        """
        self.documents_info = documents_info
        self.content_type = content_type
    
    async def search(self, query: str, **kwargs) -> List[SearchResult]:
        """Execute traditional search.
        
        Args:
            query: Search query
            **kwargs: Additional search parameters
            
        Returns:
            List of search results
        """
        try:
            logger.info(f"Executing traditional search for {self.content_type}")
            
            # Build context with all content
            context = ""
            for doc_id, doc_info in self.documents_info.items():
                if self.content_type == "summary":
                    context += f"\n\n文档ID: {doc_id}\n文档名称: {doc_info['doc_name']}\n摘要:\n{doc_info['summary']}"
                elif self.content_type == "chunk":
                    context += f"\n\n文档ID: {doc_id}\n文档名称: {doc_info['doc_name']}\n文本内容:\n"
                    for i, chunk in enumerate(doc_info['chunks']):
                        context += f"\n--- 块 {i+1} ---\n{chunk}"
            
            search_query = f"""
            基于以下查询，在文档{self.content_type}中搜索相关信息：
            
            查询：{query}
            
            文档{self.content_type}：
            {context}
            
            请提供相关的{self.content_type}信息。
            """
            
            response = await litellm.acompletion(
                model="gpt-4o",
                messages=[{"role": "user", "content": search_query}],
            )
            
            # Parse traditional search results
            results = []
            for line in response.choices[0].message.content.split('\n'):
                if "文档ID:" in line and ("摘要:" in line or "文本内容:" in line):
                    parts = line.split("文档ID:")
                    if len(parts) > 1:
                        doc_id = parts[1].split("摘要:" if self.content_type == "summary" else "文本内容:")[0].strip()
                        content_start = line.find("摘要:" if self.content_type == "summary" else "文本内容:") + len("摘要:" if self.content_type == "summary" else "文本内容:")
                        content = line[content_start:].strip()
                        results.append(SearchResult(
                            doc_id=doc_id,
                            content=content,
                            score=1.0,  # Traditional search doesn't have scores
                            source_type="traditional_search",
                            metadata={}
                        ))
            
            logger.info(f"Traditional search completed: {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Traditional search failed: {e}")
            return []


class StructureFileManager:
    """Manages document structure files saved locally."""
    
    def __init__(self, structure_directory: Path):
        """Initialize structure file manager.
        
        Args:
            structure_directory: Directory containing structure files
        """
        self.structure_directory = structure_directory
    
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


class ResultSynthesizer:
    """Synthesizes search results into final answers."""
    
    def __init__(self, openai_api_key: str):
        """Initialize result synthesizer.
        
        Args:
            openai_api_key: OpenAI API key
        """
        self.openai_api_key = openai_api_key
        litellm.api_key = openai_api_key
    
    async def synthesize_results(self, query: str, summary_results: List[SearchResult], 
                               chunk_results: List[SearchResult]):
        """Synthesize search results into final answer.
        
        Args:
            query: Original search query
            summary_results: Summary search results
            chunk_results: Chunk search results
            
        Returns:
            Synthesized final answer
        """
        try:
            logger.info("Synthesizing search results...")
            
            # Format summary results
            summary_text = self._format_search_results(summary_results, "summary")
            
            # Format chunk results
            chunk_text = self._format_search_results(chunk_results, "chunk")
            
            # response = await litellm.acompletion(
            #     model="gpt-4o",
            #     messages=[{"role": "user", "content": synthesis_prompt}],
            # )
            
            # final_answer = response.choices[0].message.content
            # logger.info("Result synthesis completed")
            result_dict = {
                "query": query,
                "summary_results": summary_text,
                "chunk_results": chunk_text
            }
            return result_dict
            
        except Exception as e:
            logger.error(f"Result synthesis failed: {e}")
            return f"抱歉，生成回答时出现错误: {e}"
    
    def _format_search_results(self, results: List[SearchResult], result_type: str) -> str:
        """Format search results for synthesis.
        
        Args:
            results: List of search results
            result_type: Type of results (summary, chunk)
            
        Returns:
            Formatted string
        """
        if not results:
            return "无相关结果"
        
        formatted = []
        for result in results:
            doc_id = result.doc_id
            score = result.score
            content = result.content
            
            # Truncate content to avoid excessive length
            # if len(content) > 2000:
            #     content = content[:2000] + "..."
            
            formatted.append(f"文档ID: {doc_id} (相关度: {score:.3f})\n{content}")
        
        return "\n\n".join(formatted)


class DocumentRetriever:
    """Main document retriever class implementing hierarchical search."""
    
    def __init__(self, openai_api_key: str, documents_info: Dict[str, Dict], 
                 vector_store_ids: Dict[str, str], structure_directory: str = "pdf_structure"):
        """Initialize document retriever.
        
        Args:
            openai_api_key: OpenAI API key
            documents_info: Dictionary containing document information
            vector_store_ids: Dictionary mapping store types to their IDs
            structure_directory: Directory containing structure files
        """
        self.openai_api_key = openai_api_key
        self.documents_info = documents_info
        self.vector_store_ids = vector_store_ids
        self.structure_directory = Path(structure_directory)
        
        # Initialize components
        self.structure_file_manager = StructureFileManager(self.structure_directory)
        self.result_synthesizer = ResultSynthesizer(openai_api_key)
        
        # Initialize search strategies
        self.summary_search_strategy = None
        self.chunk_search_strategy = None
        
        # Set up search strategies based on available vector stores
        if vector_store_ids.get("summaries"):
            self.summary_search_strategy = VectorSearchStrategy(
                vector_store_ids["summaries"]
            )
        else:
            self.summary_search_strategy = TraditionalSearchStrategy(
                documents_info, "summary"
            )
        
        if vector_store_ids.get("chunks"):
            self.chunk_search_strategy = VectorSearchStrategy(
                vector_store_ids["chunks"]
            )
        else:
            self.chunk_search_strategy = TraditionalSearchStrategy(
                documents_info, "chunk"
            )
    
    async def hierarchical_search(self, query: str, model: str = "gpt-4o") -> RetrievalResult:
        """Execute hierarchical search strategy.
        
        This method implements the hierarchical search approach:
        1. Search document summaries to find relevant documents
        2. Extract document IDs from summary results
        3. Search chunks filtered by document IDs
        4. Synthesize results into final answer
        
        Args:
            query: Search query
            model: Model to use for synthesis
            
        Returns:
            Complete retrieval result
        """
        logger.info(f"Starting hierarchical search: {query[:50]}...")
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Step 1: Search summaries to find relevant documents
            logger.info("Step 1: Searching document summaries...")
            summary_results = await self.summary_search_strategy.search(query)
            
            if not summary_results:
                logger.warning("No summary results found")
                return RetrievalResult(
                    query=query,
                    final_answer="抱歉，没有找到相关的文档信息。"
                )
            
            # Step 2: Extract document IDs from summary results
            relevant_doc_ids = list(set(result.doc_id for result in summary_results))
            logger.info(f"Found {len(relevant_doc_ids)} relevant documents: {relevant_doc_ids}")
            
            # Step 3: Search chunks filtered by document IDs
            logger.info("Step 2: Searching document chunks...")
            chunk_results = await self._search_chunks_filtered(query, relevant_doc_ids)
            
            # Step 4: Synthesize results
            logger.info("Step 3: Synthesizing results...")
            # change list to string
            final_answer = await self.result_synthesizer.synthesize_results(
                query, summary_results, chunk_results
            )
            
            # Calculate search statistics
            end_time = asyncio.get_event_loop().time()
            search_stats = {
                "total_time": end_time - start_time,
                "summary_results_count": len(summary_results),
                "chunk_results_count": len(chunk_results),
                "relevant_doc_ids": relevant_doc_ids,
                "model_used": model
            }
            
            # Create retrieval result
            result = RetrievalResult(
                query=query,
                summary_results=summary_results,
                chunk_results=chunk_results,
                final_answer=final_answer,
                search_stats=search_stats
            )
            
            logger.info(f"Hierarchical search completed in {search_stats['total_time']:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Hierarchical search failed: {e}")
            return RetrievalResult(
                query=query,
                final_answer=f"搜索过程中出现错误: {e}"
            )
    
    async def _search_chunks_filtered(self, query: str, doc_ids: List[str]) -> List[SearchResult]:
        """Search chunks filtered by document IDs.
        
        Args:
            query: Search query
            doc_ids: List of document IDs to filter by
            
        Returns:
            Filtered chunk search results
        """
        try:
            # Get all chunk results
            all_chunk_results = await self.chunk_search_strategy.search(query)
            
            # Filter by document IDs
            filtered_results = []
            for result in all_chunk_results:
                if result.doc_id in doc_ids:
                    filtered_results.append(result)
            
            logger.info(f"Chunk search filtered: {len(filtered_results)} results from {len(all_chunk_results)} total")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Filtered chunk search failed: {e}")
            return []
    
    async def search_documents(self, query: str, search_type: str = "hierarchical", 
                             model: str = "gpt-4o") -> str:
        """Search documents using specified strategy.
        
        Args:
            query: Search query
            search_type: Type of search strategy
            model: Model to use for synthesis
            
        Returns:
            Search result as string
        """
        logger.info(f"Searching documents: {query[:50]}... (type: {search_type})")
        
        try:
            if search_type == "hierarchical":
                result = await self.hierarchical_search(query, model)
                return result.final_answer
            elif search_type == "summary":
                results = await self.summary_search_strategy.search(query)
                return self._format_search_results(results, "summary")
            elif search_type == "chunk":
                results = await self.chunk_search_strategy.search(query)
                return self._format_search_results(results, "chunk")
            else:
                raise ValueError(f"Unsupported search type: {search_type}")
                
        except Exception as e:
            logger.error(f"Document search failed: {e}")
            raise
    
    def _format_search_results(self, results: List[SearchResult], result_type: str) -> str:
        """Format search results for display.
        
        Args:
            results: List of search results
            result_type: Type of results
            
        Returns:
            Formatted string
        """
        if not results:
            return "无相关结果"
        
        formatted = []
        for result in results:
            doc_id = result.doc_id
            score = result.score
            content = result.content
            
            # Truncate content
            if len(content) > 300:
                content = content[:300] + "..."
            
            formatted.append(f"文档ID: {doc_id} (相关度: {score:.3f})\n{content}")
        
        return "\n\n".join(formatted)
    
    def get_document_structure(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document structure from local file.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document structure or None if not found
        """
        return self.structure_file_manager.load_document_structure(doc_id)
    
    def get_status(self) -> Dict[str, Any]:
        """Get retriever status.
        
        Returns:
            Status information dictionary
        """
        return {
            "vector_store_ids": self.vector_store_ids,
            "documents_count": len(self.documents_info),
            "structure_directory": str(self.structure_directory),
            "structure_files": len(self.structure_file_manager.get_all_structure_files()),
            "summary_search_strategy": type(self.summary_search_strategy).__name__,
            "chunk_search_strategy": type(self.chunk_search_strategy).__name__
        }
    
    async def interactive_chat(self, model: str = "gpt-4o"):
        """Interactive chat mode for document retrieval.
        
        Args:
            model: Model to use for synthesis
        """
        if not self.documents_info:
            print("请先加载文档信息")
            return
        
        print("\n" + "="*50)
        print("文档检索系统 - 交互式聊天模式")
        print("输入 'quit' 或 'exit' 退出")
        print("="*50)
        
        while True:
            try:
                user_input = input("\n你: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("再见！")
                    break
                
                if not user_input:
                    continue
                
                # Execute hierarchical search
                result = await self.hierarchical_search(user_input, model)
                
                print(f"\nAI: {result.final_answer}")
                
                # Display search statistics
                stats = result.search_stats
                print(f"\n📊 搜索统计:")
                print(f"   ⏱️  用时: {stats['total_time']:.2f} 秒")
                print(f"   📄 摘要结果: {stats['summary_results_count']} 个")
                print(f"   📝 内容结果: {stats['chunk_results_count']} 个")
                print(f"   📚 相关文档: {len(stats['relevant_doc_ids'])} 个")
                
            except KeyboardInterrupt:
                print("\n\n再见！")
                break
            except Exception as e:
                print(f"\n错误: {e}")


async def main():
    """Main function for testing document retriever."""
    print("Document Retriever Module")
    print("=" * 50)
    
    # Get OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        openai_api_key = input("Enter your OpenAI API key: ").strip()
        if not openai_api_key:
            print("Error: OpenAI API key required")
            return
    
    # Load configuration
    config_file = "document_processor_config.json"
    if not Path(config_file).exists():
        print(f"Configuration file not found: {config_file}")
        print("Please run document processor first")
        return
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Extract configuration data
        documents_info = config.get("documents_info", {})
        vector_store_ids = config.get("vector_store_ids", {})
        structure_directory = config.get("structure_directory", "pdf_structure")
        
        if not documents_info:
            print("No documents found in configuration")
            return
        
        # Create document retriever
        retriever = DocumentRetriever(
            openai_api_key=openai_api_key,
            documents_info=documents_info,
            vector_store_ids=vector_store_ids,
            structure_directory=structure_directory
        )
        
        # Display status
        status = retriever.get_status()
        print(f"\nRetriever Status:")
        print(f"- Documents: {status['documents_count']}")
        print(f"- Structure files: {status['structure_files']}")
        print(f"- Summary strategy: {status['summary_search_strategy']}")
        print(f"- Chunk strategy: {status['chunk_search_strategy']}")
        
        # # Start interactive chat
        # print("\nStarting interactive chat...")
        # await retriever.interactive_chat()
        
        ## Test hierarchical search
        query = "What does memory mean in LLM agents?"
        print(f"\nTesting hierarchical search with query: {query}")
        result = await retriever.hierarchical_search(query)
        print(f"\nFinal Answer: {result.final_answer}")
        
        summary_results = result.summary_results
        print(f"\nSummary Results ({summary_results}):")
        
        chunk_results = result.chunk_results
        print(f"\nChunk Results ({chunk_results}):")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 