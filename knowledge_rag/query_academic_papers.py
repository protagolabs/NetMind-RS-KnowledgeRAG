#!/usr/bin/env python3
"""
Academic Papers Query and Answer System

Retrieves relevant information from the processed academic papers knowledge base
and generates comprehensive answers with proper citations using GPT-4o.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

@dataclass
class RetrievedChunk:
    """A chunk of retrieved information"""
    content: str
    document_title: str
    document_file: str
    episode_id: str
    entity_names: List[str]
    relationships: List[str]
    relevance_score: float
    chunk_type: str  # 'episode', 'entity', 'relationship'
    retrieval_method: str  # 'keyword_bm25', 'embedding_semantic', 'graph_bfs'
    metadata: Dict[str, Any]

class AcademicQuerySystem:
    """System for querying academic papers knowledge base"""
    
    def __init__(self):
        self.driver = None
        self.neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.neo4j_username = os.getenv('NEO4J_USERNAME', 'neo4j')
        self.neo4j_password = os.getenv('NEO4J_PASSWORD', 'password')
        
        # Initialize OpenAI client
        from openai import OpenAI
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model_name = "gpt-4o-2024-08-06"
        
        # Initialize ChromaDB for embedding-based retrieval
        self.chroma_collection = None
        self._init_embedding_system()
        
        print(f"🤖 Initialized query system with model: {self.model_name}")
    
    def _init_embedding_system(self):
        """Initialize ChromaDB for embedding-based retrieval"""
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Initialize ChromaDB client
            chroma_client = chromadb.PersistentClient(
                path="./academic_papers_chroma_db",
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            try:
                self.chroma_collection = chroma_client.get_collection("academic_episodes")
                print(f"✅ Connected to existing ChromaDB collection")
            except:
                print(f"⚠️  ChromaDB collection not found - embedding retrieval will be disabled")
                self.chroma_collection = None
                
        except ImportError:
            print(f"⚠️  ChromaDB not installed - embedding retrieval will be disabled")
            self.chroma_collection = None
        except Exception as e:
            print(f"⚠️  ChromaDB initialization failed: {e} - embedding retrieval will be disabled")
            self.chroma_collection = None
    
    def connect(self):
        """Connect to Neo4j database"""
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(
                self.neo4j_uri, 
                auth=(self.neo4j_username, self.neo4j_password)
            )
            
            # Test connection
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                result.single()
            
            print(f"✅ Connected to Neo4j database")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j: {e}")
            return False
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
    
    def get_database_overview(self) -> Dict[str, Any]:
        """Get overview of the knowledge base"""
        with self.driver.session() as session:
            # Get document count
            result = session.run("MATCH (d:Document) RETURN count(d) as doc_count")
            doc_count = result.single()['doc_count']
            
            # Get episode count
            result = session.run("MATCH (ep:Episode) RETURN count(ep) as episode_count")
            episode_count = result.single()['episode_count']
            
            # Get entity count
            result = session.run("MATCH (e:Entity) RETURN count(e) as entity_count")
            entity_count = result.single()['entity_count']
            
            # Get relationship edge count (not relationship nodes)
            result = session.run("MATCH ()-[r:RELATES_TO|MENTIONS]->() RETURN count(r) as rel_count")
            rel_count = result.single()['rel_count']
            
            # Get document titles
            result = session.run("""
                MATCH (d:Document)
                RETURN d.title as title, d.file_name as file_name
                ORDER BY d.title
            """)
            documents = [{"title": record['title'], "file_name": record['file_name']} for record in result]
            
            return {
                "documents": doc_count,
                "episodes": episode_count,
                "entities": entity_count,
                "relationships": rel_count,
                "document_list": documents
            }
    
    def retrieve_by_multiple_methods(self, query: str, max_chunks: int = 20) -> List[RetrievedChunk]:
        """Retrieve relevant chunks using multiple retrieval methods"""
        all_chunks = []
        
        # Extract keywords for BM25-style retrieval
        keywords = self._extract_keywords(query)
        print(f"🔍 Extracted keywords: {keywords}")
        
        with self.driver.session() as session:
            # Method 1: Keyword/BM25-style retrieval (40% of chunks)
            keyword_limit = max(1, int(max_chunks * 0.4))
            print(f"🔤 Keyword/BM25 retrieval (limit: {keyword_limit})")
            keyword_chunks = self._retrieve_by_keywords(session, keywords, keyword_limit)
            all_chunks.extend(keyword_chunks)
            print(f"   📄 Found {len(keyword_chunks)} keyword-based chunks")
            
            # Method 2: Embedding/Semantic retrieval (40% of chunks)
            embedding_limit = max(1, int(max_chunks * 0.4))
            print(f"🧠 Embedding/Semantic retrieval (limit: {embedding_limit})")
            embedding_chunks = self._retrieve_by_embeddings(query, embedding_limit)
            all_chunks.extend(embedding_chunks)
            print(f"   📄 Found {len(embedding_chunks)} embedding-based chunks")
            
            # Method 3: Graph/BFS retrieval (20% of chunks)
            graph_limit = max(1, int(max_chunks * 0.2))
            print(f"🕸️  Graph/BFS retrieval (limit: {graph_limit})")
            graph_chunks = self._retrieve_by_graph_bfs(session, keywords, graph_limit)
            all_chunks.extend(graph_chunks)
            print(f"   📄 Found {len(graph_chunks)} graph-based chunks")
        
        # Remove duplicates while preserving the best retrieval method for each chunk
        unique_chunks = self._deduplicate_chunks(all_chunks)
        
        # Sort by relevance score and limit
        unique_chunks.sort(key=lambda x: x.relevance_score, reverse=True)
        final_chunks = unique_chunks[:max_chunks]
        
        # Print retrieval method summary
        method_counts = {}
        for chunk in final_chunks:
            method_counts[chunk.retrieval_method] = method_counts.get(chunk.retrieval_method, 0) + 1
        
        print(f"📊 Final retrieval summary:")
        for method, count in method_counts.items():
            print(f"   • {method}: {count} chunks")
        
        return final_chunks
    
    def _retrieve_by_keywords(self, session, keywords: List[str], limit: int) -> List[RetrievedChunk]:
        """BM25-style keyword retrieval from episodes and entities"""
        chunks = []
        
        # Search in episodes
        episode_chunks = self._search_episodes(session, keywords, limit // 2)
        chunks.extend(episode_chunks)
        
        # Search in entities
        entity_chunks = self._search_entities(session, keywords, limit // 2)
        chunks.extend(entity_chunks)
        
        return chunks
    
    def _retrieve_by_embeddings(self, query: str, limit: int) -> List[RetrievedChunk]:
        """Semantic retrieval using embeddings (ChromaDB)"""
        chunks = []
        
        if not self.chroma_collection:
            return chunks
        
        try:
            # Query ChromaDB for semantically similar content
            results = self.chroma_collection.query(
                query_texts=[query],
                n_results=limit,
                include=['documents', 'metadatas', 'distances']
            )
            
            if not results['documents'] or not results['documents'][0]:
                return chunks
            
            # Convert ChromaDB results to RetrievedChunk objects
            documents = results['documents'][0]
            metadatas = results['metadatas'][0] if results['metadatas'] else []
            distances = results['distances'][0] if results['distances'] else []
            
            for i, (content, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
                # Convert distance to relevance score (lower distance = higher relevance)
                relevance_score = max(0.0, 1.0 - (distance / 2.0))  # Normalize distance to 0-1 range
                
                chunk = RetrievedChunk(
                    content=content,
                    document_title=metadata.get('document_title', 'Unknown'),
                    document_file=metadata.get('document_file', 'Unknown'),
                    episode_id=metadata.get('episode_id', ''),
                    entity_names=metadata.get('entity_names', []),
                    relationships=[],
                    relevance_score=relevance_score,
                    chunk_type='episode',
                    retrieval_method='embedding_semantic',
                    metadata={
                        'embedding_distance': distance,
                        'semantic_similarity': relevance_score
                    }
                )
                chunks.append(chunk)
                
        except Exception as e:
            print(f"   ⚠️  Embedding retrieval failed: {e}")
        
        return chunks
    
    def _retrieve_by_graph_bfs(self, session, keywords: List[str], limit: int) -> List[RetrievedChunk]:
        """Graph-based retrieval using BFS traversal from relevant entities"""
        chunks = []
        
        try:
            # Step 1: Find seed entities relevant to the query
            seed_entities = self._find_seed_entities(session, keywords)
            if not seed_entities:
                return chunks
            
            print(f"   🌱 Found {len(seed_entities)} seed entities: {[e['name'] for e in seed_entities[:3]]}...")
            
            # Step 2: Perform BFS traversal to find connected information
            visited_entities = set()
            chunks_found = []
            
            for seed_entity in seed_entities[:3]:  # Limit seed entities to avoid explosion
                entity_chunks = self._bfs_traverse_from_entity(session, seed_entity, visited_entities, limit // 3)
                chunks_found.extend(entity_chunks)
            
            # Step 3: Score chunks based on graph distance and relevance
            for chunk in chunks_found[:limit]:
                chunks.append(chunk)
                
        except Exception as e:
            print(f"   ⚠️  Graph BFS retrieval failed: {e}")
        
        return chunks
    
    def _find_seed_entities(self, session, keywords: List[str]) -> List[Dict]:
        """Find entities that match the query keywords to use as seeds for BFS"""
        if not keywords:
            return []
        
        # Build search conditions for entities
        conditions = []
        for keyword in keywords[:3]:  # Limit to first 3 keywords
            conditions.append(f"(toLower(e.name) CONTAINS toLower('{keyword}') OR toLower(e.summary) CONTAINS toLower('{keyword}'))")
        
        query = f"""
        MATCH (e:Entity)
        WHERE {' OR '.join(conditions)}
        WITH e, size([keyword IN {keywords} WHERE toLower(e.name) CONTAINS toLower(keyword) OR toLower(e.summary) CONTAINS toLower(keyword)]) as match_count
        RETURN e.uuid as entity_id, e.name as name, e.entity_type as type, 
               e.summary as summary, match_count
        ORDER BY match_count DESC
        LIMIT 5
        """
        
        result = session.run(query)
        return [dict(record) for record in result]
    
    def _bfs_traverse_from_entity(self, session, seed_entity: Dict, visited_entities: set, limit: int) -> List[RetrievedChunk]:
        """Perform BFS traversal from a seed entity to find connected information"""
        chunks = []
        entity_id = seed_entity['entity_id']
        
        if entity_id in visited_entities:
            return chunks
        
        visited_entities.add(entity_id)
        
        # Find episodes where this entity is mentioned
        query = """
        MATCH (e:Entity {uuid: $entity_id})-[:MENTIONED_IN]->(ep:Episode)<-[:CONTAINS]-(d:Document)
        RETURN DISTINCT d.title as doc_title, d.file_name as doc_file,
               ep.uuid as episode_id, ep.content as content,
               collect(e.name) as entity_names
        LIMIT $limit
        """
        
        result = session.run(query, entity_id=entity_id, limit=limit)
        
        for record in result:
            # Calculate relevance based on entity importance and graph distance
            relevance_score = min(1.0, seed_entity['match_count'] * 0.3 + 0.4)  # Base score from seed relevance
            
            chunk = RetrievedChunk(
                content=record['content'],
                document_title=record['doc_title'],
                document_file=record['doc_file'],
                episode_id=record['episode_id'],
                entity_names=record['entity_names'],
                relationships=[],
                relevance_score=relevance_score,
                chunk_type='episode',
                retrieval_method='graph_bfs',
                metadata={
                    'seed_entity': seed_entity['name'],
                    'seed_entity_type': seed_entity['type'],
                    'bfs_distance': 1,
                    'graph_relevance': relevance_score
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _deduplicate_chunks(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """Remove duplicate chunks while preserving the best retrieval method for each"""
        seen_episodes = {}
        unique_chunks = []
        
        for chunk in chunks:
            episode_key = f"{chunk.document_title}:{chunk.episode_id}"
            
            if episode_key not in seen_episodes:
                seen_episodes[episode_key] = chunk
                unique_chunks.append(chunk)
            else:
                # Keep the chunk with higher relevance score
                existing_chunk = seen_episodes[episode_key]
                if chunk.relevance_score > existing_chunk.relevance_score:
                    # Replace with better chunk
                    unique_chunks.remove(existing_chunk)
                    seen_episodes[episode_key] = chunk
                    unique_chunks.append(chunk)
        
        return unique_chunks
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query using simple methods"""
        # For Chinese queries, we'll use character-based matching
        # For better results, you could integrate jieba or other Chinese NLP tools
        
        # Common academic keywords (English and Chinese)
        academic_keywords = [
            # Model training related
            "训练", "预训练", "微调", "监督", "无监督", "半监督",
            "training", "pretraining", "fine-tuning", "supervised", "unsupervised",
            
            # Model architectures
            "transformer", "bert", "gpt", "attention", "encoder", "decoder",
            "模型", "架构", "注意力", "编码器", "解码器",
            
            # Technical terms
            "算法", "方法", "技术", "机制", "框架",
            "algorithm", "method", "technique", "mechanism", "framework",
            
            # Performance metrics
            "准确率", "性能", "效果", "结果", "评估",
            "accuracy", "performance", "results", "evaluation"
        ]
        
        # Extract keywords that appear in the query
        found_keywords = []
        query_lower = query.lower()
        
        for keyword in academic_keywords:
            if keyword.lower() in query_lower:
                found_keywords.append(keyword)
        
        # Also add query terms as keywords
        # Simple tokenization for Chinese/English mixed text
        import re
        tokens = re.findall(r'\w+', query)
        found_keywords.extend([token for token in tokens if len(token) > 1])
        
        return list(set(found_keywords))
    
    def _search_episodes(self, session, keywords: List[str], limit: int) -> List[RetrievedChunk]:
        """Search in episode content"""
        chunks = []
        
        # Build search conditions
        conditions = []
        for keyword in keywords[:5]:  # Limit to first 5 keywords
            conditions.append(f"toLower(ep.content) CONTAINS toLower('{keyword}')")
        
        if not conditions:
            return chunks
        
        query = f"""
        MATCH (d:Document)-[:CONTAINS]->(ep:Episode)
        WHERE {' OR '.join(conditions)}
        WITH d, ep, 
             size([keyword IN {keywords} WHERE toLower(ep.content) CONTAINS toLower(keyword)]) as match_count
        
        OPTIONAL MATCH (ep)<-[:MENTIONED_IN]-(e:Entity)
        WITH d, ep, match_count, collect(e.name) as entity_names
        
        RETURN d.title as doc_title, d.file_name as doc_file, 
               ep.uuid as episode_id, ep.content as content,
               entity_names, match_count
        ORDER BY match_count DESC, size(ep.content) ASC
        LIMIT {limit}
        """
        
        result = session.run(query)
        for record in result:
            chunk = RetrievedChunk(
                content=record['content'],
                document_title=record['doc_title'],
                document_file=record['doc_file'],
                episode_id=record['episode_id'],
                entity_names=record['entity_names'] or [],
                relationships=[],
                relevance_score=record['match_count'] / len(keywords),
                chunk_type='episode',
                retrieval_method='keyword_bm25',
                metadata={'match_count': record['match_count']}
            )
            chunks.append(chunk)
        
        return chunks
    
    def _search_entities(self, session, keywords: List[str], limit: int) -> List[RetrievedChunk]:
        """Search in entity information"""
        chunks = []
        
        conditions = []
        for keyword in keywords[:5]:
            conditions.append(f"(toLower(e.name) CONTAINS toLower('{keyword}') OR toLower(e.summary) CONTAINS toLower('{keyword}'))")
        
        if not conditions:
            return chunks
        
        query = f"""
        MATCH (e:Entity)-[:MENTIONED_IN]->(ep:Episode)<-[:CONTAINS]-(d:Document)
        WHERE {' OR '.join(conditions)}
        WITH e, d, ep,
             size([keyword IN {keywords} WHERE toLower(e.name) CONTAINS toLower(keyword) OR toLower(e.summary) CONTAINS toLower(keyword)]) as match_count
        
        RETURN DISTINCT e.name as entity_name, e.entity_type as entity_type, 
               e.summary as entity_summary, 
               collect(DISTINCT d.title)[0] as doc_title,
               collect(DISTINCT d.file_name)[0] as doc_file,
               collect(DISTINCT ep.uuid)[0] as episode_id,
               match_count
        ORDER BY match_count DESC
        LIMIT {limit}
        """
        
        result = session.run(query)
        for record in result:
            content = f"实体: {record['entity_name']} ({record['entity_type']})\n摘要: {record['entity_summary']}"
            
            chunk = RetrievedChunk(
                content=content,
                document_title=record['doc_title'],
                document_file=record['doc_file'],
                episode_id=record['episode_id'],
                entity_names=[record['entity_name']],
                relationships=[],
                relevance_score=record['match_count'] / len(keywords),
                chunk_type='entity',
                retrieval_method='keyword_bm25',
                metadata={
                    'entity_type': record['entity_type'],
                    'match_count': record['match_count']
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def generate_answer_with_references(self, query: str, retrieved_chunks: List[RetrievedChunk]) -> Dict[str, Any]:
        """Generate comprehensive answer using retrieved chunks with proper citations"""
        
        # Prepare context for the LLM
        context_parts = []
        references = {}
        
        # Group chunks by retrieval method for better organization
        method_groups = {
            'keyword_bm25': [],
            'embedding_semantic': [],
            'graph_bfs': []
        }
        
        for chunk in retrieved_chunks:
            method_groups[chunk.retrieval_method].append(chunk)
        
        ref_counter = 1
        
        # Process each retrieval method group
        for method, chunks in method_groups.items():
            if not chunks:
                continue
                
            method_names = {
                'keyword_bm25': '关键词/BM25检索',
                'embedding_semantic': '语义嵌入检索', 
                'graph_bfs': '图结构BFS检索'
            }
            
            for chunk in chunks:
                ref_id = f"[{ref_counter}]"
                references[ref_id] = {
                    "document": chunk.document_title,
                    "file": chunk.document_file,
                    "episode_id": chunk.episode_id,
                    "type": chunk.chunk_type,
                    "retrieval_method": chunk.retrieval_method,
                    "retrieval_method_name": method_names[chunk.retrieval_method],
                    "entities": chunk.entity_names,
                    "relevance": chunk.relevance_score,
                    "metadata": chunk.metadata,
                    "full_content": chunk.content  # Store full content for excerpt extraction
                }
                
                # Add retrieval method indicator to context
                method_indicator = f"[{method_names[chunk.retrieval_method]}]"
                context_parts.append(f"{ref_id} {method_indicator} {chunk.content}\n来源: {chunk.document_title}")
                ref_counter += 1
        
        context_text = "\n\n".join(context_parts)
        
        # Create enhanced prompt that explains retrieval methods
        prompt = f"""你是一个学术研究助手。基于通过不同检索方法获得的学术论文片段，请为用户的问题提供全面、准确的答案。

检索方法说明:
- [关键词/BM25检索]: 基于关键词匹配的传统检索
- [语义嵌入检索]: 基于语义相似度的向量检索
- [图结构BFS检索]: 基于知识图谱关系的图遍历检索

要求:
1. 答案要结构清晰，分点论述
2. 每个要点都必须包含明确的引用，使用[数字]格式引用相关片段
3. 引用要准确指向支持该观点的具体内容
4. 答案要专业、客观，基于提供的材料
5. 如果材料不足以完全回答问题，请说明局限性
6. 可以适当提及不同检索方法获得的信息的互补性

用户问题: {query}

相关材料:
{context_text}

请提供详细的答案:"""

        # Generate answer using OpenAI
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一个专业的学术研究助手，擅长分析和总结学术论文内容，理解不同检索方法的特点。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=4000
            )
            
            answer = response.choices[0].message.content
            
            # Extract relevant excerpts for episode references
            print("3️⃣ Extracting precise excerpts from episode references...")
            enhanced_references = self._extract_reference_excerpts(answer, references)
            
            return {
                "query": query,
                "answer": answer,
                "references": enhanced_references,
                "retrieved_chunks_count": len(retrieved_chunks),
                "generation_model": self.model_name,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            return {
                "query": query,
                "answer": f"抱歉，生成答案时发生错误: {str(e)}",
                "references": references,
                "retrieved_chunks_count": len(retrieved_chunks),
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _extract_reference_excerpts(self, answer: str, references: Dict[str, Any]) -> Dict[str, Any]:
        """Extract the most relevant excerpts from episode references based on how they were used in the answer"""
        enhanced_references = references.copy()
        
        # Find all reference IDs used in the answer
        import re
        used_ref_ids = re.findall(r'\[(\d+)\]', answer)
        
        for ref_id_num in used_ref_ids:
            ref_id = f"[{ref_id_num}]"
            if ref_id not in references:
                continue
                
            ref_info = references[ref_id]
            
            # Only extract excerpts for episode-type references (they tend to be longer)
            if ref_info['type'] != 'episode':
                continue
                
            full_content = ref_info.get('full_content', '')
            if not full_content or len(full_content) < 200:  # Skip short content
                continue
                
            try:
                # Find the context around this reference in the answer
                ref_context = self._find_reference_context_in_answer(answer, ref_id)
                
                # Extract the most relevant excerpt
                excerpt = self._extract_most_relevant_excerpt(full_content, ref_context, ref_info['document'])
                
                if excerpt and excerpt != full_content:
                    enhanced_references[ref_id]['relevant_excerpt'] = excerpt
                    enhanced_references[ref_id]['excerpt_extracted'] = True
                    enhanced_references[ref_id]['excerpt_length'] = len(excerpt)
                    enhanced_references[ref_id]['full_content_length'] = len(full_content)
                else:
                    enhanced_references[ref_id]['relevant_excerpt'] = "整个段落都相关"
                    enhanced_references[ref_id]['excerpt_extracted'] = False
                    
            except Exception as e:
                print(f"   ⚠️ Failed to extract excerpt for {ref_id}: {e}")
                enhanced_references[ref_id]['relevant_excerpt'] = "提取失败，请参考完整内容"
                enhanced_references[ref_id]['excerpt_extracted'] = False
        
        return enhanced_references
    
    def _find_reference_context_in_answer(self, answer: str, ref_id: str) -> str:
        """Find the sentence or context where a reference is used in the answer"""
        sentences = answer.split('。')  # Split by Chinese period
        
        for sentence in sentences:
            if ref_id in sentence:
                # Return the sentence with some context
                return sentence.strip() + '。'
        
        # Fallback: find paragraph containing the reference
        paragraphs = answer.split('\n')
        for paragraph in paragraphs:
            if ref_id in paragraph:
                return paragraph.strip()
        
        return ""
    
    def _extract_most_relevant_excerpt(self, full_content: str, ref_context: str, document_title: str) -> str:
        """Use LLM to extract the most relevant excerpt from episode content"""
        
        if len(full_content) < 300:  # Don't extract from short content
            return full_content
            
        try:
            prompt = f"""你是一个文献引用专家。给定一个学术论文的段落和在答案中引用它的上下文，请提取出最相关的部分作为精确引用。

原始段落（来自《{document_title}》）:
{full_content}

引用上下文（答案中如何使用这个引用）:
{ref_context}

要求:
1. 提取原始段落中最直接支持引用上下文的部分
2. 保持提取内容的完整性和连贯性
3. 如果整个段落都相关，返回"整个段落都相关"
4. 如果只有部分相关，提取1-3句最关键的内容
5. 保留足够的上下文信息使提取内容有意义

请只返回提取的相关部分，不要添加其他解释:"""

            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一个专业的文献引用和内容提取专家。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=800
            )
            
            excerpt = response.choices[0].message.content.strip()
            
            # Clean up the response
            if excerpt.startswith('"') and excerpt.endswith('"'):
                excerpt = excerpt[1:-1]
                
            return excerpt
            
        except Exception as e:
            print(f"   ⚠️ LLM excerpt extraction failed: {e}")
            return full_content
    
    def process_query(self, query: str, max_chunks: int = 20) -> Dict[str, Any]:
        """Process a complete query from retrieval to answer generation"""
        print(f"\n🔍 Processing query: {query}")
        print("=" * 60)
        
        # Step 1: Retrieve relevant chunks using multiple methods
        print("1️⃣ Retrieving relevant information using multiple methods...")
        retrieved_chunks = self.retrieve_by_multiple_methods(query, max_chunks)
        print(f"   📄 Total retrieved: {len(retrieved_chunks)} relevant chunks")
        
        # Step 2: Generate answer with references
        print("2️⃣ Generating comprehensive answer...")
        result = self.generate_answer_with_references(query, retrieved_chunks)
        
        # Step 3 is now inside generate_answer_with_references (excerpt extraction)
        
        # Add detailed retrieval analysis to result
        retrieval_analysis = self._analyze_retrieval_methods(retrieved_chunks)
        result["retrieval_details"] = retrieval_analysis
        
        print("   ✅ Answer generated with precise excerpted references")
        return result
    
    def _analyze_retrieval_methods(self, chunks: List[RetrievedChunk]) -> Dict[str, Any]:
        """Analyze the distribution and effectiveness of different retrieval methods"""
        method_stats = {
            'keyword_bm25': {'count': 0, 'avg_relevance': 0.0, 'chunks': []},
            'embedding_semantic': {'count': 0, 'avg_relevance': 0.0, 'chunks': []},
            'graph_bfs': {'count': 0, 'avg_relevance': 0.0, 'chunks': []}
        }
        
        for chunk in chunks:
            method = chunk.retrieval_method
            if method in method_stats:
                method_stats[method]['count'] += 1
                method_stats[method]['chunks'].append(chunk.relevance_score)
        
        # Calculate average relevance for each method
        for method, stats in method_stats.items():
            if stats['count'] > 0:
                stats['avg_relevance'] = sum(stats['chunks']) / len(stats['chunks'])
                del stats['chunks']  # Remove raw scores for cleaner output
        
        return {
            "total_chunks": len(chunks),
            "method_breakdown": method_stats,
            "overall_avg_relevance": sum(c.relevance_score for c in chunks) / len(chunks) if chunks else 0.0,
            "chunk_types": {
                "episode": len([c for c in chunks if c.chunk_type == "episode"]),
                "entity": len([c for c in chunks if c.chunk_type == "entity"])
            }
        }

def main():
    """Main function to demonstrate the query system"""
    print("🎓 Academic Papers Query System")
    print("=" * 40)
    
    # Initialize system
    query_system = AcademicQuerySystem()
    
    if not query_system.connect():
        print("❌ Cannot connect to database")
        return
    
    try:
        # Show database overview
        overview = query_system.get_database_overview()
        print(f"\n📊 Knowledge Base Overview:")
        print(f"   • Documents: {overview['documents']}")
        print(f"   • Episodes: {overview['episodes']}")
        print(f"   • Entities: {overview['entities']}")
        print(f"   • Relationships: {overview['relationships']}")
        
        print(f"\n📚 Available Documents:")
        for doc in overview['document_list'][:10]:  # Show first 10
            print(f"   • {doc['title']}")
        if len(overview['document_list']) > 10:
            print(f"   ... and {len(overview['document_list']) - 10} more")
        
        # Test queries
        test_queries = [
            "请为我梳理一下模型训练的方法的变化",
            "请为我整理一下这些 paper 里的所有专业术语"
        ]
        
        results = []
        
        for query in test_queries:
            print(f"\n{'='*80}")
            result = query_system.process_query(query, max_chunks=15)
            results.append(result)
            
            # Display result summary
            print(f"\n📋 Query Result Summary:")
            print(f"   • Question: {result['query']}")
            print(f"   • Total chunks retrieved: {result['retrieval_details']['total_chunks']}")
            print(f"   • Overall avg relevance: {result['retrieval_details']['overall_avg_relevance']:.3f}")
            print(f"   • Answer length: {len(result['answer'])} characters")
            
            # Show retrieval method breakdown
            print(f"\n🔍 Retrieval Method Breakdown:")
            method_breakdown = result['retrieval_details']['method_breakdown']
            method_names = {
                'keyword_bm25': 'Keyword/BM25',
                'embedding_semantic': 'Embedding/Semantic', 
                'graph_bfs': 'Graph/BFS'
            }
            
            for method, stats in method_breakdown.items():
                if stats['count'] > 0:
                    print(f"   • {method_names[method]}: {stats['count']} chunks (avg relevance: {stats['avg_relevance']:.3f})")
            
            # Show excerpt extraction summary
            episode_refs = [ref for ref in result['references'].values() if ref['type'] == 'episode']
            extracted_refs = [ref for ref in episode_refs if ref.get('excerpt_extracted', False)]
            
            if episode_refs:
                print(f"\n📝 Reference Excerpt Extraction:")
                print(f"   • Episode references: {len(episode_refs)}")
                print(f"   • Excerpts extracted: {len(extracted_refs)}")
                if extracted_refs:
                    avg_compression = sum(ref['excerpt_length'] / ref['full_content_length'] for ref in extracted_refs) / len(extracted_refs)
                    print(f"   • Average compression: {avg_compression:.2%}")
            
            print(f"\n💡 Answer Preview:")
            print(result['answer'][:300] + "..." if len(result['answer']) > 300 else result['answer'])
        
        # Save results to JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"query_results_{timestamp}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        # Also save human-readable version
        readable_file = f"query_answers_{timestamp}.txt"
        with open(readable_file, 'w', encoding='utf-8') as f:
            f.write("Academic Papers Query Results with Multi-Method Retrieval\n")
            f.write("=" * 60 + "\n\n")
            
            for i, result in enumerate(results, 1):
                f.write(f"Query {i}: {result['query']}\n")
                f.write("-" * 50 + "\n")
                
                # Write retrieval method summary
                f.write("Retrieval Method Summary:\n")
                method_breakdown = result['retrieval_details']['method_breakdown']
                method_names = {
                    'keyword_bm25': 'Keyword/BM25检索',
                    'embedding_semantic': '语义嵌入检索', 
                    'graph_bfs': '图结构BFS检索'
                }
                
                for method, stats in method_breakdown.items():
                    if stats['count'] > 0:
                        f.write(f"• {method_names[method]}: {stats['count']} chunks (平均相关度: {stats['avg_relevance']:.3f})\n")
                f.write(f"• 总计: {result['retrieval_details']['total_chunks']} chunks\n\n")
                
                f.write(f"Answer:\n{result['answer']}\n\n")
                
                f.write("References (with Retrieval Methods & Precise Excerpts):\n")
                f.write("-" * 50 + "\n")
                
                for ref_id, ref_info in result['references'].items():
                    episode_display = ref_info['episode_id'][-8:] if ref_info['episode_id'] else 'N/A'
                    method_display = ref_info.get('retrieval_method_name', ref_info.get('retrieval_method', 'Unknown'))
                    relevance = ref_info.get('relevance', 0.0)
                    
                    # Basic reference info
                    f.write(f"{ref_id} [{method_display}] {ref_info['document']}\n")
                    f.write(f"    Episode: {episode_display} | 相关度: {relevance:.3f}\n")
                    
                    # Add extracted excerpt for episode references
                    if ref_info['type'] == 'episode' and 'relevant_excerpt' in ref_info:
                        excerpt = ref_info['relevant_excerpt']
                        excerpt_extracted = ref_info.get('excerpt_extracted', False)
                        
                        if excerpt_extracted:
                            excerpt_len = ref_info.get('excerpt_length', 0)
                            full_len = ref_info.get('full_content_length', 0)
                            f.write(f"    📝 精确引用片段 ({excerpt_len}/{full_len} 字符):\n")
                            f.write(f"    \"{excerpt}\"\n")
                        else:
                            f.write(f"    📝 引用说明: {excerpt}\n")
                    elif ref_info['type'] == 'entity':
                        f.write(f"    📝 实体引用: {', '.join(ref_info.get('entities', []))}\n")
                    
                    f.write("\n")
                
                f.write("="*80 + "\n\n")
        
        print(f"📝 Human-readable results saved to: {readable_file}")
        
    finally:
        query_system.close()

if __name__ == "__main__":
    main() 