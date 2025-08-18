"""
Test hybrid retrieval and QA system.

This module tests the hybrid retrieval system by:
1. Loading test questions from sample_test_questions_by_type.json
2. Performing hybrid retrieval (semantic + BM25 search + graph traversal)
3. Generating structured answers with LLM and references
"""

import os
import sys
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field
from collections import Counter
import re

# load env
from dotenv import load_dotenv
load_dotenv()

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.storage.storage_manager import PaperStorageManager
from src.storage.models import SearchResult

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI library not installed. Install with: pip install openai")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RetrievedContent(BaseModel):
    """Model for retrieved content with source tracking."""
    
    content: str = Field(description="The retrieved content")
    source_file: str = Field(description="Source document")
    chunk_id: Optional[str] = Field(None, description="Chunk identifier")
    retrieval_method: str = Field(description="How this content was retrieved")
    score: float = Field(default=0.0, description="Relevance score")
    entity_name: Optional[str] = Field(None, description="Associated entity if applicable")
    entity_type: Optional[str] = Field(None, description="Entity type if applicable")
    relationship_type: Optional[str] = Field(None, description="Relationship type if applicable")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class QuestionAnswer(BaseModel):
    """Model for question-answer with references."""
    
    question: str = Field(description="The question")
    answer: str = Field(description="Structured answer with references")
    retrieved_contents: List[RetrievedContent] = Field(description="All retrieved contents")
    references: List[Dict[str, Any]] = Field(description="Reference details")
    llm_selected_indices: List[int] = Field(default_factory=list, description="Indices of contents selected by LLM")


class HybridRetrievalQA:
    """Test hybrid retrieval and QA system."""
    
    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_username: str = "neo4j",
        neo4j_password: str = "gfll9999",
        chroma_persist_dir: str = "./chroma_db_papers",
        openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY"),
        llm_model: str = "gpt-4"
    ):
        """
        Initialize the hybrid retrieval QA system.
        
        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_username: Neo4j username
            neo4j_password: Neo4j password
            chroma_persist_dir: ChromaDB persistence directory
            openai_api_key: OpenAI API key for embeddings and LLM
            llm_model: Model to use for answer generation
        """
        self.storage_manager = PaperStorageManager(
            neo4j_uri=neo4j_uri,
            neo4j_username=neo4j_username,
            neo4j_password=neo4j_password,
            chroma_persist_dir=chroma_persist_dir,
            openai_api_key=openai_api_key
        )
        
        # Initialize OpenAI client for LLM
        if OPENAI_AVAILABLE and openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
            self.llm_model = llm_model
        else:
            self.openai_client = None
            self.llm_model = None
            logger.warning("OpenAI client not initialized. LLM-based answer generation disabled.")
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def initialize(self):
        """Initialize the storage manager."""
        await self.storage_manager.initialize()
        self.logger.info("HybridRetrievalQA initialized")
    
    async def close(self):
        """Close connections."""
        await self.storage_manager.close()
    
    def _bm25_score(self, query: str, text: str) -> float:
        """
        Simple BM25-like scoring for keyword matching.
        
        Args:
            query: Search query
            text: Text to score
            
        Returns:
            BM25-like relevance score
        """
        # Tokenize
        query_terms = query.lower().split()
        text_terms = text.lower().split()
        text_term_freq = Counter(text_terms)
        
        # Parameters
        k1 = 1.2
        b = 0.75
        avg_doc_len = 200  # Approximate average
        doc_len = len(text_terms)
        
        score = 0.0
        for term in query_terms:
            if term in text_term_freq:
                tf = text_term_freq[term]
                # Simplified BM25 formula
                idf = 1.0  # Simplified - would need document frequency
                norm = (1 - b + b * (doc_len / avg_doc_len))
                score += idf * (tf * (k1 + 1)) / (tf + k1 * norm)
        
        return score
    
    async def _find_initial_entities(
        self,
        question: str,
        limit: int = 5
    ) -> List[Tuple[str, str, float, str]]:
        """
        Find initial entities using both semantic and BM25 search.
        
        Args:
            question: The question
            limit: Max entities per search type
            
        Returns:
            List of (entity_name, entity_type, score, method) tuples
        """
        entities = []
        
        # 1. Semantic entity search
        semantic_results = await self.storage_manager.vector_store.search_entities_semantic(
            query=question,
            limit=limit
        )
        
        for result in semantic_results:
            entities.append((
                result.entity_name,
                result.entity_type,
                result.score,
                "semantic_search"
            ))
        
        # 2. BM25/keyword search on entities (simplified using graph store text search)
        # Extract keywords from question - be more inclusive
        # Include capitalized words, acronyms, and important nouns
        keywords = re.findall(r'\b[A-Z][a-z]+\b|\b[A-Z]{2,}\b', question)
        
        # Also extract key terms even if not capitalized (for better coverage)
        important_terms = ['model', 'training', 'code', 'math', 'mathematical', 'reasoning', 
                          'BERT', 'DistilBERT', 'transformer', 'attention', 'language']
        for term in important_terms:
            if term.lower() in question.lower() and term not in keywords:
                keywords.append(term)
        
        if keywords:
            # Search for entities matching keywords
            keyword_query = " ".join(keywords)
            graph_results = await self.storage_manager.graph_store.search_entities(
                keyword_query,
                limit=limit
            )
            
            for result in graph_results:
                # Calculate BM25 score
                bm25_score = self._bm25_score(question, result.content)
                entities.append((
                    result.entity_name,
                    result.entity_type,
                    bm25_score,
                    "bm25_search"
                ))
        
        # Deduplicate and sort by score
        seen = set()
        unique_entities = []
        for entity in sorted(entities, key=lambda x: x[2], reverse=True):
            if entity[0] not in seen:
                seen.add(entity[0])
                unique_entities.append(entity)
        
        return unique_entities[:limit * 2]  # Return top entities
    
    def _score_neighbor_relevance(
        self,
        neighbor_entity: str,
        neighbor_type: str,
        relationship_type: str,
        query_context: str
    ) -> float:
        """
        Score neighbor relevance based on relationship type and entity type.
        
        Args:
            neighbor_entity: Neighbor entity name
            neighbor_type: Neighbor entity type
            relationship_type: Type of relationship
            query_context: Query for context
            
        Returns:
            Relevance score (0-1)
        """
        # Relationship type priorities for academic papers
        relationship_scores = {
            'IMPROVES_UPON': 1.0,
            'EXTENDS': 0.95,
            'OUTPERFORMS': 0.9,
            'BUILDS_ON': 0.85,
            'IMPLEMENTS': 0.8,
            'USES_DATASET': 0.75,
            'APPLIES': 0.7,
            'EVALUATES_WITH': 0.65,
            'COMPARES_TO': 0.6,
            'VALIDATES': 0.55,
            'INSPIRED_BY': 0.5,
            'RELATED_TO': 0.4,
            'MENTIONS': 0.3,
            'REFERENCES': 0.25
        }
        
        # Entity type priorities
        entity_type_scores = {
            'METHODOLOGY': 0.9,
            'MODEL': 0.9,
            'ALGORITHM': 0.85,
            'CONTRIBUTION': 0.85,
            'RESULT': 0.8,
            'EXPERIMENT': 0.75,
            'DATASET': 0.7,
            'METRIC': 0.65,
            'TECHNOLOGY': 0.6,
            'FRAMEWORK': 0.6,
            'CONCEPT': 0.5,
            'THEORY': 0.5
        }
        
        # Get base scores
        rel_score = relationship_scores.get(relationship_type, 0.2)
        type_score = entity_type_scores.get(neighbor_type, 0.3)
        
        # Check if entity name appears in query (bonus)
        name_bonus = 0.2 if neighbor_entity.lower() in query_context.lower() else 0.0
        
        # Combine scores (weighted average)
        final_score = (rel_score * 0.5 + type_score * 0.3 + name_bonus * 0.2)
        
        return min(final_score, 1.0)
    
    async def _intelligent_graph_traversal(
        self,
        initial_entities: List[Tuple[str, str, float, str]],
        question: str,
        max_neighbors_per_entity: int = 3,
        max_total_relationships: int = 10
    ) -> List[RetrievedContent]:
        """
        Intelligently traverse graph from initial entities.
        
        Args:
            initial_entities: List of (name, type, score, method) tuples
            question: Original question for context
            max_neighbors_per_entity: Max neighbors to select per entity
            max_total_relationships: Total relationship limit
            
        Returns:
            List of retrieved contents from graph
        """
        retrieved_contents = []
        total_relationships = 0
        
        for entity_name, entity_type, entity_score, discovery_method in initial_entities:
            if total_relationships >= max_total_relationships:
                break
            
            # Get all relationships for this entity
            relationships = await self.storage_manager.graph_store.find_relationships(entity_name)
            
            if not relationships:
                continue
            
            # Score and rank neighbors
            scored_neighbors = []
            for rel in relationships:
                # Determine neighbor entity (the one that's not our current entity)
                if hasattr(rel, 'source_entity') and rel.source_entity != entity_name:
                    neighbor = rel.source_entity
                    neighbor_type = rel.metadata.get('source_type', 'UNKNOWN')
                elif hasattr(rel, 'target_entity') and rel.target_entity != entity_name:
                    neighbor = rel.target_entity
                    neighbor_type = rel.metadata.get('target_type', 'UNKNOWN')
                else:
                    continue
                
                rel_type = rel.metadata.get('relationship_type', 'RELATED_TO')
                
                # Score this neighbor
                relevance_score = self._score_neighbor_relevance(
                    neighbor,
                    neighbor_type,
                    rel_type,
                    question
                )
                
                # Boost score based on initial entity score
                combined_score = relevance_score * (0.7 + 0.3 * entity_score)
                
                scored_neighbors.append((rel, combined_score, rel_type))
            
            # Sort by score and take top N
            scored_neighbors.sort(key=lambda x: x[1], reverse=True)
            
            for rel, score, rel_type in scored_neighbors[:max_neighbors_per_entity]:
                if total_relationships >= max_total_relationships:
                    break
                
                content = RetrievedContent(
                    content=rel.content if hasattr(rel, 'content') else str(rel),
                    source_file=rel.source_file if hasattr(rel, 'source_file') else "",
                    retrieval_method=f"graph_traversal_from_{discovery_method}",
                    score=score,
                    entity_name=entity_name,
                    entity_type=entity_type,
                    relationship_type=rel_type,
                    metadata={
                        'initial_entity_score': entity_score,
                        'traversal_depth': 1,
                        'discovery_method': discovery_method
                    }
                )
                retrieved_contents.append(content)
                total_relationships += 1
        
        return retrieved_contents
    
    async def retrieve_for_question(
        self,
        question: str,
        limit: int = 10
    ) -> List[RetrievedContent]:
        """
        Perform enhanced hybrid retrieval for a question.
        
        Args:
            question: The question to retrieve content for
            limit: Maximum results per retrieval type
            
        Returns:
            List of retrieved contents with source tracking
        """
        retrieved_contents = []
        
        try:
            # 1. Find initial entities using semantic + BM25
            initial_entities = await self._find_initial_entities(question, limit=5)
            self.logger.info(f"Found {len(initial_entities)} initial entities")
            
            # 2. Perform semantic chunk search
            chunk_results = await self.storage_manager.vector_store.search_chunks_semantic(
                query=question,
                limit=limit
            )
            
            for result in chunk_results:
                content = RetrievedContent(
                    content=result.content,
                    source_file=result.source_file,
                    chunk_id=result.metadata.get('chunk_id'),
                    retrieval_method="semantic_chunk_search",
                    score=result.score,
                    metadata={
                        'sequence': result.metadata.get('sequence'),
                        'paper_title': result.metadata.get('paper_title')
                    }
                )
                retrieved_contents.append(content)
            
            # 3. Add initial entities as content
            for entity_name, entity_type, score, method in initial_entities[:5]:
                # Get entity details from graph
                entity_results = await self.storage_manager.graph_store.search_entities(
                    entity_name,
                    limit=1
                )
                
                if entity_results:
                    entity_result = entity_results[0]
                    content = RetrievedContent(
                        content=entity_result.content,
                        source_file=entity_result.source_file,
                        retrieval_method=f"initial_entity_{method}",
                        score=score,
                        entity_name=entity_name,
                        entity_type=entity_type,
                        metadata=entity_result.metadata
                    )
                    retrieved_contents.append(content)
            
            # 4. Intelligent graph traversal
            graph_contents = await self._intelligent_graph_traversal(
                initial_entities,
                question,
                max_neighbors_per_entity=3,
                max_total_relationships=10
            )
            retrieved_contents.extend(graph_contents)
            
            # Log if no graph traversal happened
            if not graph_contents:
                self.logger.warning(f"No graph traversal results for question: {question[:50]}...")
            
            # Sort by score (highest first)
            retrieved_contents.sort(key=lambda x: x.score, reverse=True)
            
            self.logger.info(f"Total retrieved contents: {len(retrieved_contents)}")
            
            return retrieved_contents
            
        except Exception as e:
            self.logger.error(f"Retrieval failed for question: {e}")
            return []
    
    def generate_llm_answer(
        self,
        question: str,
        retrieved_contents: List[RetrievedContent]
    ) -> QuestionAnswer:
        """
        Generate answer using LLM with retrieved contents.
        
        Args:
            question: The question
            retrieved_contents: Retrieved contents
            
        Returns:
            QuestionAnswer with LLM-generated response
        """
        if not retrieved_contents:
            return QuestionAnswer(
                question=question,
                answer="No relevant content found for this question.",
                retrieved_contents=[],
                references=[]
            )
        
        if not self.openai_client:
            # Fallback to template-based answer
            return self._generate_template_answer(question, retrieved_contents)
        
        # Prepare context for LLM - include ALL retrieved contents without truncation
        context_items = []
        for i, content in enumerate(retrieved_contents, 1):  # ALL results, no limit
            context_item = f"[{i}] "
            if content.entity_name:
                context_item += f"Entity: {content.entity_name} ({content.entity_type})\n"
            if content.chunk_id:
                context_item += f"Chunk ID: {content.chunk_id}\n"
            context_item += f"Method: {content.retrieval_method}\n"
            context_item += f"Score: {content.score:.3f}\n"
            context_item += f"Content: {content.content}"  # Full content, no truncation
            context_items.append(context_item)
        
        context = "\n\n".join(context_items)
        
        # Create prompt for LLM
        prompt = f"""You are an expert research assistant. You MUST answer the user's question using the retrieved content below.

CRITICAL INSTRUCTIONS:
1. You MUST provide an answer based on the retrieved content, even if it's partial or indirect
2. Use bullet points for your answer
3. Each bullet point MUST include at least one reference using [number] format
4. Look for ANY relevant information in the retrieved content that relates to the question
5. If the content only partially answers the question, use what's available and note what's missing
6. Do NOT say "the retrieved content does not provide information" - find and use what IS there
7. Be specific and cite exact content where possible

User Question: {question}

Retrieved Content (YOU MUST USE THIS):
{context}

REQUIREMENT: Generate an answer using the above content. Find relevant pieces even if they don't directly answer the question. Always provide references [number] to support your points."""

        try:
            # Call OpenAI API
            response = self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a precise research assistant that answers questions based solely on provided content. Always cite your sources using [number] references."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            llm_answer = response.choices[0].message.content
            
            # Extract which content indices were referenced
            referenced_indices = []
            for i in range(1, len(retrieved_contents) + 1):
                if f"[{i}]" in llm_answer:
                    referenced_indices.append(i)
            
            # Build reference list
            references = []
            for idx in referenced_indices:
                if idx <= len(retrieved_contents):
                    content = retrieved_contents[idx - 1]
                    ref = {
                        'index': idx,
                        'source_file': content.source_file,
                        'chunk_id': content.chunk_id,
                        'retrieval_method': content.retrieval_method,
                        'entity': content.entity_name,
                        'score': content.score
                    }
                    references.append(ref)
            
            # Add reference details at the end
            reference_section = "\n\n**References:**\n"
            for ref in references:
                method_label = self._format_method_name(ref['retrieval_method'])
                reference_section += f"[{ref['index']}] {ref['source_file'] or 'Unknown source'} "
                reference_section += f"(chunk: {ref['chunk_id'] or 'N/A'}, method: {method_label}, "
                reference_section += f"score: {ref['score']:.3f})\n"
                if ref['entity']:
                    reference_section += f"    Entity: {ref['entity']}\n"
            
            final_answer = llm_answer + reference_section
            
            return QuestionAnswer(
                question=question,
                answer=final_answer,
                retrieved_contents=retrieved_contents,
                references=references,
                llm_selected_indices=referenced_indices
            )
            
        except Exception as e:
            self.logger.error(f"LLM answer generation failed: {e}")
            # Fallback to template-based answer
            return self._generate_template_answer(question, retrieved_contents)
    
    def _generate_template_answer(
        self,
        question: str,
        retrieved_contents: List[RetrievedContent]
    ) -> QuestionAnswer:
        """Fallback template-based answer generation."""
        answer_parts = ["Based on the retrieved content:"]
        references = []
        
        # Group by retrieval method
        by_method = {}
        for i, content in enumerate(retrieved_contents[:5], 1):
            method = content.retrieval_method
            if method not in by_method:
                by_method[method] = []
            by_method[method].append((i, content))
            
            ref = {
                'index': i,
                'source_file': content.source_file,
                'chunk_id': content.chunk_id,
                'retrieval_method': content.retrieval_method,
                'entity': content.entity_name,
                'score': content.score
            }
            references.append(ref)
        
        # Build answer
        for method, contents in by_method.items():
            method_name = self._format_method_name(method)
            answer_parts.append(f"\n• **{method_name}:**")
            for ref_idx, content in contents[:2]:
                summary = content.content[:150] + "..."
                if content.entity_name:
                    answer_parts.append(f"  - {content.entity_name}: {summary} [{ref_idx}]")
                else:
                    answer_parts.append(f"  - {summary} [{ref_idx}]")
        
        # Add references
        answer_parts.append("\n**References:**")
        for ref in references:
            answer_parts.append(
                f"[{ref['index']}] {ref['source_file'] or 'Unknown'} "
                f"(method: {self._format_method_name(ref['retrieval_method'])}, "
                f"score: {ref['score']:.3f})"
            )
        
        return QuestionAnswer(
            question=question,
            answer="\n".join(answer_parts),
            retrieved_contents=retrieved_contents,
            references=references
        )
    
    def _format_method_name(self, method: str) -> str:
        """Format retrieval method name for display."""
        method_names = {
            'semantic_chunk_search': 'Semantic Chunk Search',
            'initial_entity_semantic_search': 'Initial Entity (Semantic)',
            'initial_entity_bm25_search': 'Initial Entity (BM25)',
            'graph_traversal_from_semantic_search': 'Graph Traversal (from Semantic)',
            'graph_traversal_from_bm25_search': 'Graph Traversal (from BM25)',
            'semantic_entity_search': 'Semantic Entity Search',
            'graph_entity_center_node': 'Graph Center Node',
            'graph_extended_relationship': 'Graph Extended Node'
        }
        return method_names.get(method, method.replace('_', ' ').title())
    
    async def test_questions(
        self,
        questions_file: str,
        output_file: str,
        max_questions: Optional[int] = None
    ):
        """
        Test hybrid retrieval with questions from file.
        
        Args:
            questions_file: Path to questions JSON file
            output_file: Path to save results
            max_questions: Maximum number of questions to test
        """
        # Load questions
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        
        if max_questions:
            questions = questions[:max_questions]
        
        results = []
        
        for i, q_data in enumerate(questions, 1):
            question = q_data['question']
            doc_chunk_type = q_data['doc_chunk_type']
            question_type = q_data['question_type']
            expected_answer = q_data['answer']
            expected_chunks = q_data['chunk']
            
            self.logger.info(f"Processing question {i}/{len(questions)}: {question[:100]}...")
            
            # Retrieve content
            retrieved_contents = await self.retrieve_for_question(question)
            
            # Generate answer with LLM
            qa_result = self.generate_llm_answer(question, retrieved_contents)
            
            # Store result
            result = {
                'question_id': i,
                'doc_chunk_type': doc_chunk_type,
                'question_type': question_type,
                'question': question,
                'expected_answer': expected_answer,
                'expected_chunks': expected_chunks,
                'generated_answer': qa_result.answer,
                'retrieved_count': len(retrieved_contents),
                'llm_selected_count': len(qa_result.llm_selected_indices),
                'retrieval_methods_used': list(set(c.retrieval_method for c in retrieved_contents)),
                'top_scores': [c.score for c in retrieved_contents[:5]],
                'references': qa_result.references,
                'retrieved_context': [
                    {
                        'index': idx + 1,
                        'entity_name': content.entity_name,
                        'entity_type': content.entity_type,
                        'chunk_id': content.chunk_id,
                        'method': content.retrieval_method,
                        'score': content.score,
                        'content_preview': content.content[:200] + '...' if len(content.content) > 200 else content.content
                    }
                    for idx, content in enumerate(retrieved_contents[:10])  # Show top 10 for inspection
                ]
            }
            results.append(result)
            
            # Log progress
            self.logger.info(f"  Retrieved {len(retrieved_contents)} contents")
            self.logger.info(f"  LLM selected {len(qa_result.llm_selected_indices)} for answer")
            self.logger.info(f"  Methods: {list(set(c.retrieval_method for c in retrieved_contents))[:3]}")
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"Results saved to {output_file}")
        
        # Print summary
        print("\n" + "="*80)
        print("ENHANCED HYBRID RETRIEVAL QA TEST SUMMARY")
        print("="*80)
        print(f"Total questions tested: {len(results)}")
        print(f"Average retrieved contents: {sum(r['retrieved_count'] for r in results) / len(results):.1f}")
        print(f"Average LLM-selected contents: {sum(r['llm_selected_count'] for r in results) / len(results):.1f}")
        
        # Retrieval method statistics
        all_methods = []
        for r in results:
            all_methods.extend(r['retrieval_methods_used'])
        
        method_counts = Counter(all_methods)
        print("\nRetrieval methods used:")
        for method, count in method_counts.most_common():
            print(f"  - {self._format_method_name(method)}: {count} questions")
        
        print(f"\nResults saved to: {output_file}")


async def main():
    """Main test function."""
    # Initialize QA system
    qa_system = HybridRetrievalQA(
        chroma_persist_dir="/home/administrator/projects/NetMind-RS-KnowledgeRAG/chroma_db_papers"
    )
    
    await qa_system.initialize()
    
    try:
        # Test with sample questions
        questions_file = "/home/administrator/projects/NetMind-RS-KnowledgeRAG/test_question/sample_test_questions_by_type.json"
        output_file = "/home/administrator/projects/NetMind-RS-KnowledgeRAG/src/tests/enhanced_hybrid_retrieval_qa_results.json"
        
        # Test first 10 questions for demonstration
        await qa_system.test_questions(
            questions_file=questions_file,
            output_file=output_file,
            max_questions=10
        )
        
    finally:
        await qa_system.close()


if __name__ == "__main__":
    asyncio.run(main())