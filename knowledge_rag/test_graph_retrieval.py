#!/usr/bin/env python3
"""
Graph Database Retrieval Testing

Tests various retrieval patterns and queries on the Neo4j graph database
to validate that extracted knowledge is properly stored and accessible.
"""

import sys
import os
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import json

# Add the current directory to Python path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️  python-dotenv not installed, trying system environment variables")


class GraphRetrieval:
    """Handles various retrieval operations from Neo4j graph database"""
    
    def __init__(self):
        self.driver = None
        self.neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.neo4j_username = os.getenv('NEO4J_USERNAME', 'neo4j')
        self.neo4j_password = os.getenv('NEO4J_PASSWORD', 'password')
    
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
                session.run("RETURN 1 as test")
            
            print(f"✅ Connected to Neo4j at {self.neo4j_uri}")
            return True
            
        except ImportError:
            print("❌ Neo4j driver not installed. Run: pip install neo4j")
            return False
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j: {e}")
            print("💡 Make sure Neo4j is running: docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest")
            return False
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            print("✅ Neo4j connection closed")
    
    def get_database_overview(self) -> Dict[str, Any]:
        """Get an overview of what's stored in the database"""
        print("\n📊 Database Overview")
        print("-" * 30)
        
        overview = {}
        
        with self.driver.session() as session:
            # Count nodes by label (using simple approach instead of APOC)
            labels = ["Document", "Episode", "Entity"]
            node_counts = {}
            
            for label in labels:
                try:
                    result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                    record = result.single()
                    count = record["count"] if record else 0
                    node_counts[label] = count
                except Exception as e:
                    print(f"⚠️  Could not count {label} nodes: {e}")
                    node_counts[label] = 0
            
            overview["nodes"] = node_counts
            
            # Count relationships by actual types we know exist
            rel_types = ["CONTAINS", "MENTIONED_IN"]
            rel_counts = {}
            
            for rel_type in rel_types:
                try:
                    result = session.run(f"MATCH ()-[r:{rel_type}]-() RETURN count(r) as count")
                    record = result.single()
                    count = record["count"] if record else 0
                    rel_counts[rel_type] = count
                except Exception as e:
                    print(f"⚠️  Could not count {rel_type} relationships: {e}")
                    rel_counts[rel_type] = 0
            
            overview["relationships"] = rel_counts
        
        # Print overview
        print("📈 Node Counts:")
        for label, count in overview["nodes"].items():
            print(f"   • {label}: {count:,}")
        
        print("\n🔗 Relationship Counts:")
        for rel_type, count in overview["relationships"].items():
            print(f"   • {rel_type}: {count:,}")
        
        return overview
    
    def test_document_retrieval(self) -> List[Dict]:
        """Test document retrieval"""
        print("\n📚 Document Retrieval Test")
        print("-" * 35)
        
        with self.driver.session() as session:
            # Get all documents with flexible episode counting (using actual CONTAINS relationship)
            result = session.run("""
                MATCH (d:Document)
                OPTIONAL MATCH (d)-[:CONTAINS]->(ep:Episode)
                WITH d, count(ep) as episode_count
                RETURN d.uuid as uuid, d.title as title, d.file_name as file_name,
                       d.document_type as document_type, 
                       CASE WHEN d.created_at IS NOT NULL THEN d.created_at ELSE 'Unknown' END as created_at,
                       episode_count
                ORDER BY 
                    CASE WHEN d.created_at IS NOT NULL THEN d.created_at ELSE '1970-01-01' END DESC
            """)
            
            documents = []
            for record in result:
                doc = dict(record)
                documents.append(doc)
                
                print(f"📄 Document: {doc['title']}")
                print(f"   • UUID: {doc['uuid']}")
                print(f"   • File: {doc['file_name']}")
                print(f"   • Type: {doc['document_type']}")
                print(f"   • Episodes: {doc['episode_count']}")
                print(f"   • Created: {doc['created_at']}")
                print()
        
        print(f"✅ Found {len(documents)} documents")
        return documents
    
    def test_entity_retrieval(self, limit: int = 20) -> List[Dict]:
        """Test entity retrieval with different patterns"""
        print(f"\n🏷️  Entity Retrieval Test (top {limit})")
        print("-" * 40)
        
        entities = []
        
        with self.driver.session() as session:
            # Get entities by type with counts
            result = session.run("""
                MATCH (e:Entity)
                RETURN e.entity_type as type, count(e) as count
                ORDER BY count DESC
            """)
            
            print("📊 Entities by Type:")
            for record in result:
                print(f"   • {record['type']}: {record['count']} entities")
            
            print()
            
            # Get top entities with most relationships
            result = session.run(f"""
                MATCH (e:Entity)
                OPTIONAL MATCH (e)-[r]-()
                WITH e, count(r) as rel_count
                RETURN e.uuid as uuid, e.name as name, e.entity_type as type,
                       e.summary as summary, e.confidence as confidence,
                       rel_count
                ORDER BY rel_count DESC, e.confidence DESC
                LIMIT {limit}
            """)
            
            print("🔥 Top Connected Entities:")
            for i, record in enumerate(result, 1):
                entity = dict(record)
                entities.append(entity)
                
                print(f"   {i}. {entity['name']} ({entity['type']})")
                print(f"      • Confidence: {entity['confidence']:.3f}")
                print(f"      • Relationships: {entity['rel_count']}")
                if entity['summary']:
                    summary_preview = entity['summary'][:80] + "..." if len(entity['summary']) > 80 else entity['summary']
                    print(f"      • Summary: {summary_preview}")
                print()
        
        print(f"✅ Retrieved {len(entities)} top entities")
        return entities
    
    def test_relationship_retrieval(self, limit: int = 15) -> List[Dict]:
        """Test relationship retrieval"""
        print(f"\n🔗 Relationship Retrieval Test (top {limit})")
        print("-" * 45)
        
        relationships = []
        
        with self.driver.session() as session:
            # First check if we have Entity-to-Entity relationships
            result = session.run("""
                MATCH (source:Entity)-[r]->(target:Entity)
                RETURN count(r) as total_entity_relationships
            """)
            
            total_entity_rels = result.single()["total_entity_relationships"]
            
            if total_entity_rels > 0:
                # Get relationships between entities
                result = session.run(f"""
                    MATCH (source:Entity)-[r]->(target:Entity)
                    RETURN source.name as source_name, source.entity_type as source_type,
                           target.name as target_name, target.entity_type as target_type,
                           type(r) as rel_type, 
                           CASE WHEN r.fact IS NOT NULL THEN r.fact ELSE toString(r) END as fact,
                           CASE WHEN r.confidence IS NOT NULL THEN r.confidence ELSE 0.5 END as confidence,
                           CASE WHEN r.created_at IS NOT NULL THEN r.created_at ELSE 'Unknown' END as created_at
                    ORDER BY 
                        CASE WHEN r.confidence IS NOT NULL THEN r.confidence ELSE 0.5 END DESC,
                        CASE WHEN r.created_at IS NOT NULL THEN r.created_at ELSE 'Unknown' END DESC
                    LIMIT {limit}
                """)
                
                print("🔥 Top Entity Relationships:")
                for i, record in enumerate(result, 1):
                    rel = dict(record)
                    relationships.append(rel)
                    
                    print(f"   {i}. {rel['source_name']} ({rel['source_type']})")
                    print(f"      ↓ {rel['rel_type']}")
                    print(f"      {rel['target_name']} ({rel['target_type']})")
                    print(f"      • Confidence: {rel['confidence']:.3f}")
                    if rel['fact'] and rel['fact'] != 'None':
                        fact_preview = rel['fact'][:100] + "..." if len(rel['fact']) > 100 else rel['fact']
                        print(f"      • Fact: {fact_preview}")
                    print()
            else:
                # Show the actual relationships we found
                print("🔗 Actual Relationship Patterns Found:")
                print("   • Document -[:CONTAINS]-> Episode")
                print("   • Entity -[:MENTIONED_IN]-> Episode")
                
                # Show some sample MENTIONED_IN relationships
                result = session.run(f"""
                    MATCH (entity:Entity)-[:MENTIONED_IN]->(episode:Episode)
                    RETURN entity.name as entity_name, entity.entity_type as entity_type,
                           episode.sequence_number as episode_number,
                           substring(episode.content, 0, 100) as episode_preview
                    ORDER BY episode.sequence_number
                    LIMIT {limit}
                """)
                
                print("\n📍 Sample Entity Mentions:")
                for i, record in enumerate(result, 1):
                    print(f"   {i}. {record['entity_name']} ({record['entity_type']})")
                    print(f"      → mentioned in Episode #{record['episode_number']}")
                    preview = record['episode_preview'].replace('\n', ' ')
                    print(f"      → \"{preview}...\"")
                    print()
        
        print(f"✅ Retrieved {len(relationships)} relationships")
        return relationships
    
    def test_episode_retrieval(self, limit: int = 10) -> List[Dict]:
        """Test episode retrieval"""
        print(f"\n📑 Episode Retrieval Test (sample {limit})")
        print("-" * 40)
        
        episodes = []
        
        with self.driver.session() as session:
            # Use actual relationships: Document -[:CONTAINS]-> Episode <-[:MENTIONED_IN]- Entity
            result = session.run(f"""
                MATCH (d:Document)-[:CONTAINS]->(e:Episode)
                OPTIONAL MATCH (e)<-[:MENTIONED_IN]-(entity:Entity)
                WITH d, e, count(entity) as entity_mentions
                RETURN e.uuid as uuid, 
                       CASE WHEN e.episode_type IS NOT NULL THEN e.episode_type ELSE 'unknown' END as type,
                       CASE WHEN e.sequence_number IS NOT NULL THEN e.sequence_number ELSE 0 END as sequence, 
                       e.content as content,
                       CASE WHEN e.timestamp IS NOT NULL THEN e.timestamp ELSE 'Unknown' END as timestamp, 
                       d.title as document_title,
                       entity_mentions
                ORDER BY 
                    CASE WHEN d.created_at IS NOT NULL THEN d.created_at ELSE '1970-01-01' END DESC, 
                    CASE WHEN e.sequence_number IS NOT NULL THEN e.sequence_number ELSE 0 END ASC
                LIMIT {limit}
            """)
            
            print("📖 Sample Episodes:")
            for i, record in enumerate(result, 1):
                episode = dict(record)
                episodes.append(episode)
                
                content = episode.get('content', 'No content available')
                if content:
                    content_preview = content[:150] + "..." if len(content) > 150 else content
                    content_preview = content_preview.replace('\n', ' ')
                else:
                    content_preview = "No content available"
                
                print(f"   {i}. Episode #{episode['sequence']} ({episode['type']})")
                print(f"      • Document: {episode['document_title']}")
                print(f"      • Entity mentions: {episode['entity_mentions']}")
                print(f"      • Content: {content_preview}")
                print()
        
        print(f"✅ Retrieved {len(episodes)} episodes")
        return episodes
    
    def test_graph_traversal_queries(self):
        """Test complex graph traversal queries"""
        print("\n🕸️  Graph Traversal Queries")
        print("-" * 35)
        
        with self.driver.session() as session:
            # Query 1: Find entities mentioned in the same episodes (co-occurrence)
            print("1️⃣  Entity Co-occurrence in Episodes:")
            result = session.run("""
                MATCH (e1:Entity)-[:MENTIONED_IN]->(ep:Episode)<-[:MENTIONED_IN]-(e2:Entity)
                WHERE e1.uuid <> e2.uuid
                WITH e1, e2, count(ep) as shared_episodes
                WHERE shared_episodes > 1
                RETURN e1.name as entity1, e1.entity_type as type1,
                       e2.name as entity2, e2.entity_type as type2,
                       shared_episodes
                ORDER BY shared_episodes DESC
                LIMIT 10
            """)
            
            for record in result:
                print(f"   • {record['entity1']} ({record['type1']}) & {record['entity2']} ({record['type2']}) - {record['shared_episodes']} shared episodes")
            
            print()
            
            # Query 2: Find documents with most diverse entity types
            print("2️⃣  Documents with Most Diverse Entities:")
            result = session.run("""
                MATCH (d:Document)-[:CONTAINS]->(ep:Episode)<-[:MENTIONED_IN]-(e:Entity)
                WITH d, collect(DISTINCT e.entity_type) as entity_types
                RETURN d.title as document, size(entity_types) as type_diversity, entity_types
                ORDER BY type_diversity DESC
                LIMIT 5
            """)
            
            for record in result:
                types_str = ", ".join(record['entity_types'])
                print(f"   • {record['document']}: {record['type_diversity']} types ({types_str})")
            
            print()
            
            # Query 3: Find most mentioned entities
            print("3️⃣  Most Frequently Mentioned Entities:")
            result = session.run("""
                MATCH (e:Entity)-[:MENTIONED_IN]->(ep:Episode)
                WITH e, count(ep) as mentions
                RETURN e.name as entity, e.entity_type as type, mentions
                ORDER BY mentions DESC
                LIMIT 8
            """)
            
            for record in result:
                print(f"   • {record['entity']} ({record['type']}): {record['mentions']} mentions")
            
            print()
            
            # Query 4: Episodes with most entity mentions
            print("4️⃣  Episodes with Most Entity Mentions:")
            result = session.run("""
                MATCH (ep:Episode)<-[:MENTIONED_IN]-(e:Entity)
                WITH ep, count(e) as entity_count
                RETURN ep.sequence_number as episode_num, 
                       substring(ep.content, 0, 80) as content_preview,
                       entity_count
                ORDER BY entity_count DESC
                LIMIT 5
            """)
            
            for record in result:
                preview = record['content_preview'].replace('\n', ' ')
                print(f"   • Episode #{record['episode_num']}: {record['entity_count']} entities")
                print(f"     \"{preview}...\"")
            
            print()
    
    def test_search_queries(self, search_terms: List[str]):
        """Test text-based search queries"""
        print(f"\n🔍 Search Query Tests")
        print("-" * 25)
        
        with self.driver.session() as session:
            for term in search_terms:
                print(f"🔎 Searching for: '{term}'")
                
                # Search entities by name/summary
                result = session.run("""
                    MATCH (e:Entity)
                    WHERE toLower(e.name) CONTAINS toLower($term)
                       OR (e.summary IS NOT NULL AND toLower(e.summary) CONTAINS toLower($term))
                    RETURN e.name as name, e.entity_type as type, 
                           CASE WHEN e.confidence IS NOT NULL THEN e.confidence ELSE 0.5 END as confidence
                    ORDER BY confidence DESC
                    LIMIT 5
                """, term=term)
                
                entities = list(result)
                if entities:
                    print("   📍 Matching Entities:")
                    for ent in entities:
                        print(f"     • {ent['name']} ({ent['type']}) - {ent['confidence']:.3f}")
                
                # Search episodes by content (using actual CONTAINS relationship)
                result = session.run("""
                    MATCH (d:Document)-[:CONTAINS]->(ep:Episode)
                    WHERE toLower(ep.content) CONTAINS toLower($term)
                    RETURN ep.uuid as uuid, d.title as document, 
                           ep.sequence_number as sequence,
                           substring(ep.content, 0, 100) as preview
                    ORDER BY ep.sequence_number
                    LIMIT 3
                """, term=term)
                
                episodes = list(result)
                if episodes:
                    print("   📄 Matching Episodes:")
                    for ep in episodes:
                        preview = ep['preview'].replace('\n', ' ')
                        print(f"     • {ep['document']} (#{ep['sequence']}): {preview}...")
                
                # Find entities mentioned in episodes containing the search term
                result = session.run("""
                    MATCH (d:Document)-[:CONTAINS]->(ep:Episode)
                    WHERE toLower(ep.content) CONTAINS toLower($term)
                    MATCH (e:Entity)-[:MENTIONED_IN]->(ep)
                    RETURN DISTINCT e.name as entity_name, e.entity_type as entity_type,
                           count(ep) as relevant_episodes
                    ORDER BY relevant_episodes DESC
                    LIMIT 3
                """, term=term)
                
                related_entities = list(result)
                if related_entities:
                    print("   🔗 Related Entities:")
                    for ent in related_entities:
                        print(f"     • {ent['entity_name']} ({ent['entity_type']}) - mentioned in {ent['relevant_episodes']} relevant episodes")
                
                if not entities and not episodes and not related_entities:
                    print("   ❌ No matches found")
                
                print()
    
    def test_analytical_queries(self):
        """Test analytical and aggregation queries"""
        print("\n📊 Analytical Queries")
        print("-" * 25)
        
        with self.driver.session() as session:
            # Analysis 1: Entity type distribution
            print("1️⃣  Entity Type Distribution:")
            result = session.run("""
                MATCH (e:Entity)
                RETURN e.entity_type as type, 
                       count(e) as count,
                       CASE WHEN avg(e.confidence) IS NOT NULL THEN avg(e.confidence) ELSE 0.5 END as avg_confidence,
                       CASE WHEN max(e.confidence) IS NOT NULL THEN max(e.confidence) ELSE 0.5 END as max_confidence
                ORDER BY count DESC
            """)
            
            for record in result:
                print(f"   • {record['type']}: {record['count']} entities (avg conf: {record['avg_confidence']:.3f})")
            
            print()
            
            # Analysis 2: Most mentioned entities
            print("2️⃣  Entity Mention Frequency:")
            result = session.run("""
                MATCH (e:Entity)-[:MENTIONED_IN]->(ep:Episode)
                WITH e, count(ep) as mentions
                RETURN e.entity_type as type,
                       avg(mentions) as avg_mentions_per_entity,
                       max(mentions) as max_mentions,
                       count(e) as entities_of_type
                ORDER BY avg_mentions_per_entity DESC
            """)
            
            for record in result:
                print(f"   • {record['type']}: avg {record['avg_mentions_per_entity']:.1f} mentions/entity (max: {record['max_mentions']}, {record['entities_of_type']} entities)")
            
            print()
            
            # Analysis 3: Document processing stats (using actual relationships)
            print("3️⃣  Document Processing Statistics:")
            result = session.run("""
                MATCH (d:Document)
                OPTIONAL MATCH (d)-[:CONTAINS]->(ep:Episode)
                OPTIONAL MATCH (ep)<-[:MENTIONED_IN]-(e:Entity)
                WITH d, count(DISTINCT ep) as episodes, count(DISTINCT e) as entities
                RETURN d.title as document,
                       episodes,
                       entities,
                       CASE WHEN episodes > 0 THEN toFloat(entities) / episodes ELSE 0 END as entities_per_episode
                ORDER BY entities DESC
            """)
            
            for record in result:
                print(f"   • {record['document']}")
                print(f"     - Episodes: {record['episodes']}")
                print(f"     - Entities: {record['entities']}")
                print(f"     - Entities per episode: {record['entities_per_episode']:.2f}")
            
            print()
            
            # Analysis 4: Episode content analysis
            print("4️⃣  Episode Content Analysis:")
            result = session.run("""
                MATCH (ep:Episode)
                OPTIONAL MATCH (ep)<-[:MENTIONED_IN]-(e:Entity)
                WITH ep, size(ep.content) as content_length, count(e) as entity_mentions
                RETURN avg(content_length) as avg_episode_length,
                       min(content_length) as min_episode_length,
                       max(content_length) as max_episode_length,
                       avg(entity_mentions) as avg_entities_per_episode,
                       count(ep) as total_episodes
            """)
            
            record = result.single()
            if record:
                print(f"   • Total episodes: {record['total_episodes']}")
                print(f"   • Average episode length: {record['avg_episode_length']:.0f} characters")
                print(f"   • Episode length range: {record['min_episode_length']} - {record['max_episode_length']} characters")
                print(f"   • Average entities per episode: {record['avg_entities_per_episode']:.1f}")
            
            print()
    
    def export_sample_data(self, output_file: str = "graph_retrieval_sample.json"):
        """Export sample data for analysis"""
        print(f"\n💾 Exporting Sample Data")
        print("-" * 30)
        
        sample_data = {}
        
        with self.driver.session() as session:
            # Sample entities with mention counts
            result = session.run("""
                MATCH (e:Entity)
                OPTIONAL MATCH (e)-[:MENTIONED_IN]->(ep:Episode)
                WITH e, count(ep) as mention_count
                RETURN e.uuid as uuid, e.name as name, e.entity_type as type,
                       e.summary as summary, 
                       CASE WHEN e.confidence IS NOT NULL THEN e.confidence ELSE 0.5 END as confidence,
                       mention_count
                ORDER BY mention_count DESC, confidence DESC
                LIMIT 50
            """)
            sample_data["entities"] = [dict(record) for record in result]
            
            # Sample entity mentions (not entity-to-entity relationships)
            result = session.run("""
                MATCH (e:Entity)-[:MENTIONED_IN]->(ep:Episode)
                MATCH (d:Document)-[:CONTAINS]->(ep)
                RETURN e.name as entity_name, e.entity_type as entity_type,
                       ep.sequence_number as episode_number,
                       d.title as document_title,
                       substring(ep.content, 0, 150) as episode_content
                ORDER BY ep.sequence_number
                LIMIT 30
            """)
            sample_data["entity_mentions"] = [dict(record) for record in result]
            
            # Sample episodes with entity counts
            result = session.run("""
                MATCH (d:Document)-[:CONTAINS]->(ep:Episode)
                OPTIONAL MATCH (ep)<-[:MENTIONED_IN]-(e:Entity)
                WITH d, ep, count(e) as entity_count
                RETURN d.title as document, ep.sequence_number as sequence,
                       ep.episode_type as type, 
                       substring(ep.content, 0, 200) as content_sample,
                       entity_count
                ORDER BY d.created_at DESC, ep.sequence_number
                LIMIT 20  
            """)
            sample_data["episodes"] = [dict(record) for record in result]
            
            # Document structure
            result = session.run("""
                MATCH (d:Document)
                OPTIONAL MATCH (d)-[:CONTAINS]->(ep:Episode)
                OPTIONAL MATCH (ep)<-[:MENTIONED_IN]-(e:Entity)
                WITH d, count(DISTINCT ep) as episode_count, count(DISTINCT e) as entity_count
                RETURN d.uuid as uuid, d.title as title, d.file_name as file_name,
                       d.document_type as document_type, episode_count, entity_count
            """)
            sample_data["documents"] = [dict(record) for record in result]
        
        # Save to JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✅ Sample data exported to: {output_file}")
        print(f"   • {len(sample_data['entities'])} entities")
        print(f"   • {len(sample_data['entity_mentions'])} entity mentions")
        print(f"   • {len(sample_data['episodes'])} episodes")
        print(f"   • {len(sample_data['documents'])} documents")

    def discover_actual_relationships(self):
        """Discover what relationship types actually exist in the database"""
        print("\n🔍 Discovering Actual Relationship Types")
        print("-" * 45)
        
        with self.driver.session() as session:
            # Get all relationship types that actually exist
            result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as relationship_type, count(r) as count
                ORDER BY count DESC
            """)
            
            actual_rels = []
            print("🔗 Found Relationship Types:")
            for record in result:
                rel_type = record["relationship_type"]
                count = record["count"]
                actual_rels.append((rel_type, count))
                print(f"   • {rel_type}: {count:,}")
            
            if not actual_rels:
                print("   ❌ No relationships found in database")
            
            return actual_rels


async def main():
    """Main function to run all retrieval tests"""
    
    print("🔍 Knowledge RAG - Graph Database Retrieval Testing")
    print("=" * 55)
    
    # Initialize retrieval tester
    retrieval = GraphRetrieval()
    
    # Connect to database
    if not retrieval.connect():
        print("❌ Cannot proceed without database connection")
        return False
    
    try:
        # Run all tests
        print("\n🚀 Starting Retrieval Tests...")
        
        # 1. Database overview
        overview = retrieval.get_database_overview()
        
        # 1.5. Discover actual relationships
        actual_rels = retrieval.discover_actual_relationships()
        
        # Check if we have data
        total_nodes = sum(overview.get("nodes", {}).values())
        if total_nodes == 0:
            print("\n⚠️  No data found in database!")
            print("Run the PDF pipeline first: python test_pdf_pipeline.py")
            return False
        
        # 2. Document retrieval
        documents = retrieval.test_document_retrieval()
        
        # 3. Entity retrieval
        entities = retrieval.test_entity_retrieval(limit=15)
        
        # 4. Relationship retrieval
        relationships = retrieval.test_relationship_retrieval(limit=10)
        
        # 5. Episode retrieval
        episodes = retrieval.test_episode_retrieval(limit=8)
        
        # 6. Graph traversal queries
        retrieval.test_graph_traversal_queries()
        
        # 7. Search queries (based on common terms from weak-to-strong paper)
        search_terms = [
            "supervision", "generalization", "model", "training", 
            "capabilities", "performance", "learning"
        ]
        retrieval.test_search_queries(search_terms)
        
        # 8. Analytical queries
        retrieval.test_analytical_queries()
        
        # 9. Export sample data
        retrieval.export_sample_data()
        
        # Final summary
        print("\n🎉 Retrieval Testing Complete!")
        print("=" * 40)
        print(f"✅ Database contains {total_nodes:,} total nodes")
        print(f"✅ Found {len(documents)} documents")
        print(f"✅ Retrieved {len(entities)} top entities")
        print(f"✅ Retrieved {len(relationships)} relationships")
        print(f"✅ Retrieved {len(episodes)} episodes")
        print(f"✅ Tested {len(search_terms)} search queries")
        print(f"✅ Sample data exported")
        
        print(f"\n🚀 Next Steps:")
        print(f"   • Visualize: Open http://localhost:7474 (Neo4j Browser)")
        print(f"   • Run custom queries in Neo4j Browser")
        print(f"   • Test semantic search: python test_search_knowledge.py")
        print(f"   • Analyze costs: python cost_analysis.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Retrieval testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up
        retrieval.close()


if __name__ == "__main__":
    print("Knowledge RAG - Graph Database Retrieval Testing")
    print("=" * 50)
    print()
    print("This script tests various retrieval patterns from Neo4j:")
    print("  • Document and episode retrieval")
    print("  • Entity and relationship queries")
    print("  • Graph traversal patterns")
    print("  • Text-based search")
    print("  • Analytical aggregations")
    print()
    
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 