"""
LightRAG-Style Query Answering System
=====================================
This system answers queries by:
1. Retrieving relevant entities from Neo4j
2. Getting associated chunks for those entities
3. Building context from entities and chunks
4. Generating answers using LLM with the context
"""

import os
import sys
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import numpy as np
from dataclasses import dataclass, field
import openai

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.storage.storage_manager import PaperStorageManager
from src.tests.neo4j_entity_retriever_enhanced import EnhancedEntityRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class QueryContext:
    """Container for query context including entities, relationships, and chunks."""
    entities: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    total_tokens: int = 0
    

class LightRAGStyleQueryAnswerer:
    """Query answering system inspired by LightRAG's approach."""
    
    def __init__(
        self, 
        database_name: str = "testsampledoc2",
        max_entity_tokens: int = 2000,
        max_chunk_tokens: int = 3000,
        max_total_tokens: int = 6000
    ):
        """
        Initialize the query answering system.
        
        Args:
            database_name: Name of the Neo4j database
            max_entity_tokens: Maximum tokens for entity context
            max_chunk_tokens: Maximum tokens for chunk context
            max_total_tokens: Maximum total tokens for context
        """
        self.database_name = database_name
        self.max_entity_tokens = max_entity_tokens
        self.max_chunk_tokens = max_chunk_tokens
        self.max_total_tokens = max_total_tokens
        
        self.entity_retriever = EnhancedEntityRetriever(database_name)
        self.storage_manager = None
        self.openai_client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
    async def initialize(self):
        """Initialize the system components."""
        logger.info("Initializing LightRAG-style query answering system...")
        
        # Initialize entity retriever
        await self.entity_retriever.initialize()
        
        # Use the storage manager from entity retriever
        self.storage_manager = self.entity_retriever.storage_manager
        
        logger.info("System initialized successfully")
    
    async def get_chunks_for_entities(
        self, 
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks associated with the given entities.
        
        Args:
            entities: List of entity information dictionaries
            
        Returns:
            List of chunk dictionaries with content and metadata
        """
        chunks = []
        seen_chunk_ids = set()
        seen_contexts = set()  # Track unique contexts to avoid duplicates
        
        for entity in entities:
            entity_name = entity['name']
            
            # First, add the entity's own context as a chunk if available
            if entity.get('context'):
                context_hash = hash(entity['context'])
                if context_hash not in seen_contexts:
                    seen_contexts.add(context_hash)
                    chunks.append({
                        'chunk_id': f"entity_context_{entity_name}",
                        'content': entity['context'],
                        'source_entity': entity_name,
                        'relationship_type': 'ENTITY_CONTEXT',
                        'context': f"Context for entity: {entity_name}"
                    })
            
            # Get chunks from relationships that have context or chunk_id
            for rel_type in ['outgoing', 'incoming']:
                for rel in entity.get('relationships', {}).get(rel_type, []):
                    # First try to use the context from the relationship
                    rel_context = rel.get('properties', {}).get('context', '')
                    if rel_context and len(rel_context) > 50:  # Only use substantial contexts
                        context_hash = hash(rel_context)
                        if context_hash not in seen_contexts:
                            seen_contexts.add(context_hash)
                            chunks.append({
                                'chunk_id': f"rel_context_{len(chunks)}",
                                'content': rel_context,
                                'source_entity': entity_name,
                                'relationship_type': rel.get('relationship'),
                                'context': f"From {entity_name} {rel.get('relationship')} {rel.get('target') or rel.get('source')}"
                            })
                    
                    # Also try to get chunk by ID if available
                    chunk_id = rel.get('properties', {}).get('chunk_id')
                    if chunk_id and chunk_id not in seen_chunk_ids:
                        seen_chunk_ids.add(chunk_id)
                        
                        # Retrieve chunk content from Neo4j or ChromaDB
                        chunk_content = await self.get_chunk_by_id(chunk_id)
                        if chunk_content:
                            chunks.append({
                                'chunk_id': chunk_id,
                                'content': chunk_content,
                                'source_entity': entity_name,
                                'relationship_type': rel.get('relationship'),
                                'context': rel.get('properties', {}).get('context', '')
                            })
        
        # Also try to get chunks directly associated with entities
        for entity in entities[:3]:  # Limit to top 3 entities to avoid too many chunks
            entity_chunks = await self.get_chunks_for_entity_name(entity['name'])
            for chunk in entity_chunks:
                if 'chunk_id' in chunk and chunk['chunk_id'] not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk['chunk_id'])
                    chunks.append(chunk)
        
        return chunks
    
    async def get_chunk_by_id(self, chunk_id: str) -> Optional[str]:
        """
        Retrieve chunk content by its ID from Neo4j or ChromaDB.
        
        Args:
            chunk_id: The chunk identifier
            
        Returns:
            Chunk content or None if not found
        """
        # First try Neo4j
        query = """
        MATCH (c:Chunk {chunk_id: $chunk_id})
        RETURN c.content as content, c.chunk_id as id
        """
        
        try:
            async with self.storage_manager.graph_store.driver.session(
                database=self.database_name
            ) as session:
                result = await session.run(query, chunk_id=chunk_id)
                record = await result.single()
                
                if record:
                    return record['content']
        except Exception as e:
            logger.debug(f"Chunk {chunk_id} not found in Neo4j: {e}")
        
        # If not found in Neo4j, try ChromaDB
        try:
            # Query ChromaDB for the chunk
            result = await self.storage_manager.vector_store.chunks_collection.get(
                ids=[chunk_id]
            )
            
            if result and result['documents']:
                return result['documents'][0]
        except Exception as e:
            logger.debug(f"Chunk {chunk_id} not found in ChromaDB: {e}")
        
        return None
    
    async def get_chunks_for_entity_name(
        self, 
        entity_name: str
    ) -> List[Dict[str, Any]]:
        """
        Get chunks directly related to an entity by name.
        
        Args:
            entity_name: Name of the entity
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        
        # First try: Query for chunks connected to the entity
        query = """
        MATCH (e:Entity {name: $entity_name})-[r]-(c:Chunk)
        RETURN c.chunk_id as chunk_id, 
               c.content as content,
               type(r) as relationship,
               r.context as context
        LIMIT 5
        """
        
        try:
            async with self.storage_manager.graph_store.driver.session(
                database=self.database_name
            ) as session:
                result = await session.run(query, entity_name=entity_name)
                records = await result.data()
                
                for record in records:
                    chunks.append({
                        'chunk_id': record['chunk_id'],
                        'content': record['content'],
                        'source_entity': entity_name,
                        'relationship_type': record['relationship'],
                        'context': record.get('context', '')
                    })
        except Exception as e:
            logger.debug(f"No direct chunks found for entity {entity_name}: {e}")
        
        # If no chunks found, try to get them from ChromaDB using semantic search
        if not chunks:
            try:
                # Use the entity description for semantic search
                search_results = await self.storage_manager.search_hybrid(
                    query=entity_name,
                    search_types=['semantic_chunks'],
                    limit=3
                )
                
                if 'semantic_chunks' in search_results:
                    for result in search_results['semantic_chunks']:
                        chunks.append({
                            'chunk_id': result.id,
                            'content': result.content,
                            'source_entity': entity_name,
                            'relationship_type': 'SEMANTIC_MATCH',
                            'context': f"Semantically related to {entity_name}"
                        })
            except Exception as e:
                logger.debug(f"Failed to get semantic chunks for entity {entity_name}: {e}")
        
        return chunks
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text.
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        # Simple estimation: ~4 characters per token
        return len(text) // 4
    
    def truncate_by_tokens(
        self, 
        items: List[Dict[str, Any]], 
        max_tokens: int,
        get_text_func=None
    ) -> List[Dict[str, Any]]:
        """
        Truncate a list of items to fit within token limit.
        
        Args:
            items: List of items to truncate
            max_tokens: Maximum token limit
            get_text_func: Function to extract text from item
            
        Returns:
            Truncated list of items
        """
        if not get_text_func:
            get_text_func = lambda x: json.dumps(x, ensure_ascii=False)
        
        truncated = []
        total_tokens = 0
        
        for item in items:
            text = get_text_func(item)
            tokens = self.estimate_tokens(text)
            
            if total_tokens + tokens > max_tokens:
                break
                
            truncated.append(item)
            total_tokens += tokens
        
        return truncated
    
    async def build_context(
        self, 
        query: str,
        top_k_entities: int = 5
    ) -> QueryContext:
        """
        Build context for the query by retrieving entities and chunks.
        
        Args:
            query: User query
            top_k_entities: Number of top entities to retrieve
            
        Returns:
            QueryContext object with entities, relationships, and chunks
        """
        context = QueryContext()
        
        # Step 1: Retrieve relevant entities
        logger.info(f"Retrieving entities for query: {query}")
        entities = await self.entity_retriever.find_closest_entities_combined(
            query, 
            top_k=top_k_entities
        )
        
        if not entities:
            logger.warning("No entities found for query")
            return context
        
        # Step 2: Process entities and extract relationships
        for entity in entities:
            # Create entity context (similar to LightRAG format)
            entity_context = {
                "entity": entity['name'],
                "type": entity.get('type', 'Unknown'),
                "description": entity.get('description', ''),
                "confidence": entity.get('confidence', 0.0)
            }
            context.entities.append(entity_context)
            
            # Extract relationships
            for rel in entity.get('relationships', {}).get('outgoing', []):
                rel_context = {
                    "source": entity['name'],
                    "target": rel.get('target'),
                    "relationship": rel.get('relationship'),
                    "description": rel.get('properties', {}).get('description', '')
                }
                context.relationships.append(rel_context)
        
        # Step 3: Truncate entities and relationships to fit token limits
        context.entities = self.truncate_by_tokens(
            context.entities,
            self.max_entity_tokens // 2  # Split between entities and relationships
        )
        
        context.relationships = self.truncate_by_tokens(
            context.relationships,
            self.max_entity_tokens // 2
        )
        
        # Step 4: Retrieve chunks for the entities
        logger.info("Retrieving chunks for entities...")
        chunks = await self.get_chunks_for_entities(entities)
        
        # Step 5: Rank and truncate chunks
        # Prioritize chunks based on entity match scores
        entity_scores = {e['name']: e.get('match_info', {}).get('combined_score', 0) 
                        for e in entities}
        
        for chunk in chunks:
            chunk['score'] = entity_scores.get(chunk.get('source_entity'), 0)
        
        # Sort chunks by score
        chunks.sort(key=lambda x: x['score'], reverse=True)
        
        # Truncate chunks to fit token limit
        context.chunks = self.truncate_by_tokens(
            chunks,
            self.max_chunk_tokens,
            get_text_func=lambda x: x.get('content', '')
        )
        
        # Calculate total tokens
        context.total_tokens = (
            sum(self.estimate_tokens(json.dumps(e)) for e in context.entities) +
            sum(self.estimate_tokens(json.dumps(r)) for r in context.relationships) +
            sum(self.estimate_tokens(c.get('content', '')) for c in context.chunks)
        )
        
        logger.info(f"Context built: {len(context.entities)} entities, "
                   f"{len(context.relationships)} relationships, "
                   f"{len(context.chunks)} chunks, "
                   f"~{context.total_tokens} tokens")
        
        return context
    
    def format_context_for_prompt(self, context: QueryContext) -> str:
        """
        Format the context into a string for the LLM prompt.
        
        Args:
            context: QueryContext object
            
        Returns:
            Formatted context string
        """
        formatted_parts = []
        
        # Format entities
        if context.entities:
            formatted_parts.append("# Entities\n")
            for i, entity in enumerate(context.entities, 1):
                formatted_parts.append(
                    f"{i}. **{entity['entity']}** ({entity['type']}): "
                    f"{entity['description']}\n"
                )
            formatted_parts.append("\n")
        
        # Format relationships
        if context.relationships:
            formatted_parts.append("# Relationships\n")
            for i, rel in enumerate(context.relationships, 1):
                formatted_parts.append(
                    f"{i}. {rel['source']} --[{rel['relationship']}]--> "
                    f"{rel['target']}: {rel.get('description', '')}\n"
                )
            formatted_parts.append("\n")
        
        # Format chunks
        if context.chunks:
            formatted_parts.append("# Source Documents\n")
            for i, chunk in enumerate(context.chunks, 1):
                formatted_parts.append(
                    f"## Chunk {i} (from {chunk.get('source_entity', 'Unknown')})\n"
                    f"{chunk['content'][:500]}...\n\n"  # Limit chunk display
                )
        
        return "".join(formatted_parts)
    
    async def answer_query(
        self, 
        query: str,
        system_prompt: Optional[str] = None,
        response_type: str = "multiple paragraphs"
    ) -> str:
        """
        Answer a query using the LightRAG-style approach.
        
        Args:
            query: User query
            system_prompt: Optional custom system prompt
            response_type: Type of response expected
            
        Returns:
            Generated answer
        """
        # Build context
        context = await self.build_context(query)
        
        if not context.entities and not context.chunks:
            return "I couldn't find relevant information to answer your query."
        
        # Format context
        formatted_context = self.format_context_for_prompt(context)
        
        # Build system prompt (inspired by LightRAG)
        if not system_prompt:
            system_prompt = f"""You are an intelligent assistant that answers questions based on provided context.

## Context Information
{formatted_context}

## Instructions
- Answer the question based ONLY on the provided context
- Be comprehensive and detailed in your response
- Format your response as {response_type}
- If the context doesn't contain enough information, say so
- Cite specific entities or relationships when relevant
"""
        
        # Generate response using OpenAI
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            return answer
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
    
    async def close(self):
        """Close connections."""
        await self.entity_retriever.close()


async def main():
    """Main execution function."""
    
    # Example queries
    test_queries = [
        "What is multi-head attention and how does it work?",
        "Explain the Transformer architecture",
        "How does scaled dot-product attention differ from additive attention?",
        "What are the key components of the attention mechanism?"
    ]
    
    # Get query from command line or use default
    import sys
    if len(sys.argv) > 1:
        query = ' '.join(sys.argv[1:])
    else:
        query = test_queries[0]
    
    print("\n" + "="*80)
    print("LIGHTRAG-STYLE QUERY ANSWERING SYSTEM")
    print("="*80)
    print(f"Database: testsampledoc2")
    print(f"Query: {query}")
    print("-"*80)
    
    # Initialize the system
    answerer = LightRAGStyleQueryAnswerer(database_name="testsampledoc2")
    
    try:
        await answerer.initialize()
        
        # Get answer
        print("\nGenerating answer...")
        answer = await answerer.answer_query(query)
        
        print("\n" + "="*80)
        print("ANSWER")
        print("="*80)
        print(answer)
        
        # Save results
        output_file = Path("query_answer_results.json")
        results = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "answer": answer,
            "database": "testsampledoc2"
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n\nResults saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Error during query answering: {e}", exc_info=True)
    
    finally:
        await answerer.close()


if __name__ == "__main__":
    asyncio.run(main())