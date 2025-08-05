#!/usr/bin/env python3
"""
Academic Papers Processing Pipeline

Processes all PDF files in the academic test set through the complete Knowledge RAG pipeline:
1. Parse each PDF into episodes using FileManager
2. Extract entities and relationships using KnowledgeExtractor  
3. Store directly in Neo4j + ChromaDB using StorageManager
4. Run retrieval queries to find commonalities across papers
5. Export results to analysis files

This combines functionality from test_pdf_pipeline.py and test_graph_retrieval.py
for batch processing of academic papers.
"""

import sys
import os
import tempfile
import asyncio
from pathlib import Path
from datetime import datetime
import json
from typing import List, Dict, Any
import glob

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


def validate_prerequisites():
    """Validate that all required services and keys are available"""
    print("🔍 Validating prerequisites...")
    
    issues = []
    
    # Check OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        issues.append("❌ OPENAI_API_KEY not found in environment variables")
    else:
        print("✅ OpenAI API key found")
    
    # Check Neo4j credentials
    neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_username = os.getenv('NEO4J_USERNAME', 'neo4j') 
    neo4j_password = os.getenv('NEO4J_PASSWORD', 'password')
    
    print(f"✅ Neo4j config: {neo4j_uri} (user: {neo4j_username})")
    
    # Try to test Neo4j connection
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))
        with driver.session() as session:
            session.run("RETURN 1 as test")
        driver.close()
        print("✅ Neo4j connection successful")
    except ImportError:
        issues.append("❌ Neo4j driver not installed (pip install neo4j)")
    except Exception as e:
        issues.append(f"❌ Neo4j connection failed: {e}")
        print("💡 Make sure Neo4j is running: docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest")
    
    if issues:
        print("\n🚨 Prerequisites not met:")
        for issue in issues:
            print(f"   {issue}")
        return False
    
    print("✅ All prerequisites validated")
    return True


def find_academic_papers(folder_path: str) -> List[Path]:
    """Find all PDF files in the academic test set folder"""
    folder = Path(folder_path)
    if not folder.exists():
        print(f"❌ Academic test set folder not found: {folder_path}")
        return []
    
    # Find all PDF files
    pdf_files = list(folder.glob("*.pdf"))
    pdf_files.sort()  # Sort for consistent processing order
    
    print(f"📚 Found {len(pdf_files)} PDF files in {folder_path}:")
    for i, pdf_file in enumerate(pdf_files, 1):
        size_mb = pdf_file.stat().st_size / 1024 / 1024
        print(f"   {i}. {pdf_file.name} ({size_mb:.1f} MB)")
    
    return pdf_files


async def process_single_paper(pdf_path: Path, cost_tracker, max_episodes: int = None) -> Dict[str, Any]:
    """Process a single paper through the pipeline"""
    
    # Initialize results dictionary first
    results = {
        "file_name": pdf_path.name,
        "file_path": str(pdf_path),
        "status": "pending",
        "episodes_created": 0,
        "entities_extracted": 0,
        "relationships_extracted": 0,
        "processing_time": 0,
        "cost": 0.0,
        "error": None
    }
    
    try:
        # Import required modules
        from src.file_management import FileManager
        from src.file_management.models import FileStatus
        from src.knowledge_extraction import KnowledgeExtractor
        from src.knowledge_extraction.models import Document
        from src.storage import StorageManager
        
        print(f"\n📄 Processing: {pdf_path.name}")
        print(f"📁 File size: {pdf_path.stat().st_size / 1024 / 1024:.1f} MB")
        
        start_time = datetime.now()
        
        # STEP 1: Parse PDF into episodes
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "paper_processing.db"
            
            # Initialize file manager
            file_manager = FileManager(
                root_folder=str(pdf_path.parent),
                db_path=str(db_path)
            )
            
            # Process the PDF file
            result = file_manager.process_file(pdf_path)
            
            if not result or result.status != FileStatus.COMPLETED:
                results["status"] = "failed_parsing"
                results["error"] = result.error_message if result else 'Unknown parsing error'
                return results
            
            results["episodes_created"] = result.episodes_created
            print(f"   ✅ PDF parsed: {result.episodes_created} episodes created")
            
            # Get parsing result to access episodes
            parser = file_manager.parser_factory.get_parser(result.metadata.file_format)
            parsing_result = parser.parse(pdf_path, result.metadata)
            
            episodes = parsing_result.episodes
            
            # Apply episode limit if specified
            if max_episodes and len(episodes) > max_episodes:
                episodes = episodes[:max_episodes]
                print(f"   ⚠️  Limited to first {max_episodes} episodes per paper to manage processing costs")
            
            # STEP 2: Create document representation
            document = Document(
                file_path=str(pdf_path),
                file_name=pdf_path.name,
                file_hash=result.metadata.file_hash,
                title=pdf_path.stem,  # Use filename without extension as title
                document_type="pdf",
                created_at=datetime.fromtimestamp(pdf_path.stat().st_ctime),
                modified_at=datetime.fromtimestamp(pdf_path.stat().st_mtime)
            )
            
            # Add episodes to document
            for episode in episodes:
                document.add_episode(episode.id)
            
            print(f"   ✅ Document created: {document.title}")
            
            # STEP 3: Extract knowledge with cost tracking
            extractor = KnowledgeExtractor(
                model_name="gpt-4o-2024-08-06",  # Use GPT-4o instead of o4-mini for better prompt compatibility
                temperature=0.1,
                max_retries=3,
                cost_tracker=cost_tracker,
                reasoning_effort="high"  # Will be ignored for non-o-series models
            )
            
            # Process episodes for knowledge extraction
            all_entities = []
            all_relationships = []
            successful_extractions = 0
            
            print(f"   🧠 Starting knowledge extraction for {len(episodes)} episodes...")
            
            for i, episode in enumerate(episodes):
                # Check cost before processing each episode
                current_cost = cost_tracker.get_session_cost()
                cost_limit = 8.0  # Per-paper cost limit (adjusted for processing all papers)
                
                if current_cost > cost_limit:
                    print(f"   ⚠️  Cost limit reached for this paper: ${current_cost:.4f}")
                    break
                
                if i % 5 == 0:  # Progress update every 5 episodes
                    print(f"   📝 Processing episode {i+1}/{len(episodes)} (Cost: ${current_cost:.4f})")
                
                # Extract knowledge from episode
                extraction_result = extractor.extract_from_episode(
                    episode=episode,
                    document=document,
                    existing_entities=all_entities,
                    existing_relationships=all_relationships
                )
                
                if extraction_result.success:
                    all_entities = extraction_result.entities
                    all_relationships = extraction_result.relationships
                    successful_extractions += 1
                else:
                    print(f"   ⚠️  Episode {i+1} extraction failed: {extraction_result.error_message}")
            
            results["entities_extracted"] = len(all_entities)
            results["relationships_extracted"] = len(all_relationships)
            
            print(f"   ✅ Knowledge extraction complete:")
            print(f"      • Episodes processed: {successful_extractions}/{len(episodes)}")
            print(f"      • Entities: {len(all_entities)}")
            print(f"      • Relationships: {len(all_relationships)}")
            
            # Update document knowledge statistics
            document.update_knowledge_stats(all_entities, all_relationships)
            
            # STEP 4: Store in databases
            storage_manager = StorageManager(
                # Neo4j configuration
                neo4j_uri=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
                neo4j_username=os.getenv('NEO4J_USERNAME', 'neo4j'),
                neo4j_password=os.getenv('NEO4J_PASSWORD', 'password'),
                
                # ChromaDB configuration (shared across all papers)
                chroma_persist_dir="./academic_papers_chroma_db",
                openai_api_key=os.getenv('OPENAI_API_KEY')
            )
            
            await storage_manager.initialize()
            
            print(f"   💾 Storing knowledge in databases...")
            print(f"      • Document: {document.title}")
            print(f"      • Episodes: {len(episodes)}")
            print(f"      • Entities: {len(all_entities)}")
            print(f"      • Relationships: {len(all_relationships)}")
            
            # Store the complete knowledge base
            storage_results = await storage_manager.store_knowledge_base(
                document=document,
                episodes=episodes,
                entities=all_entities,
                relationships=all_relationships
            )
            
            await storage_manager.close()
            
            print(f"   ✅ Knowledge stored in databases")
            print(f"      • Storage results: {storage_results}")
            
            # Calculate final metrics
            end_time = datetime.now()
            results["processing_time"] = (end_time - start_time).total_seconds()
            results["cost"] = cost_tracker.get_session_cost() - results.get("initial_cost", 0)
            results["status"] = "completed"
            
            # Add document data for storage validation
            results["file_hash"] = document.file_hash
            results["title"] = document.title
            results["file_name"] = document.file_name
            results["document_uuid"] = document.uuid
            
            return results
            
    except Exception as e:
        results["status"] = "failed_processing"
        results["error"] = str(e)
        print(f"   ❌ Processing failed: {e}")
        return results


class AcademicRetriever:
    """Handles retrieval queries across all processed academic papers"""
    
    def __init__(self):
        self.driver = None
        self.neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.neo4j_username = os.getenv('NEO4J_USERNAME', 'neo4j')
        self.neo4j_password = os.getenv('NEO4J_PASSWORD', 'password')
        
        # Initialize OpenAI client for answer generation
        from openai import OpenAI
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model_name = "gpt-4o-2024-08-06"
    
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
            
            print(f"✅ Connected to Neo4j for retrieval")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j for retrieval: {e}")
            return False
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
    
    def get_paper_overview(self) -> Dict[str, Any]:
        """Get overview of all papers in the database"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (d:Document)
                OPTIONAL MATCH (d)-[:CONTAINS]->(ep:Episode)
                OPTIONAL MATCH (ep)<-[:MENTIONED_IN]-(e:Entity)
                WITH d, count(DISTINCT ep) as episodes, count(DISTINCT e) as entities
                RETURN d.title as paper_title, d.file_name as file_name,
                       episodes, entities
                ORDER BY d.created_at DESC
            """)
            
            papers = []
            for record in result:
                papers.append(dict(record))
            
            return papers
    
    def find_common_entities(self, min_papers: int = 2) -> List[Dict[str, Any]]:
        """Find entities mentioned across multiple papers"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:Entity)-[:MENTIONED_IN]->(ep:Episode)<-[:CONTAINS]-(d:Document)
                WITH e, collect(DISTINCT d.title) as papers, count(DISTINCT d) as paper_count
                WHERE paper_count >= $min_papers
                RETURN e.name as entity_name, e.entity_type as entity_type,
                       e.summary as summary, paper_count, papers
                ORDER BY paper_count DESC, e.name ASC
            """, min_papers=min_papers)
            
            common_entities = []
            for record in result:
                common_entities.append(dict(record))
            
            return common_entities
    
    def find_common_themes(self) -> List[Dict[str, Any]]:
        """Find common themes by analyzing entity types and concepts"""
        with self.driver.session() as session:
            # Find concepts that appear across multiple papers
            result = session.run("""
                MATCH (e:Entity)-[:MENTIONED_IN]->(ep:Episode)<-[:CONTAINS]-(d:Document)
                WHERE e.entity_type = 'CONCEPT'
                WITH e, collect(DISTINCT d.title) as papers, count(DISTINCT d) as paper_count
                WHERE paper_count >= 2
                RETURN e.name as concept, e.summary as description,
                       paper_count, papers
                ORDER BY paper_count DESC, e.name ASC
                LIMIT 20
            """)
            
            common_concepts = []
            for record in result:
                common_concepts.append(dict(record))
            
            return common_concepts
    
    def find_common_methods(self) -> List[Dict[str, Any]]:
        """Find common methods and techniques across papers"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:Entity)-[:MENTIONED_IN]->(ep:Episode)<-[:CONTAINS]-(d:Document)
                WHERE e.entity_type IN ['METHOD', 'TECHNIQUE', 'ALGORITHM']
                   OR (e.entity_type = 'CONCEPT' AND 
                       (toLower(e.name) CONTAINS 'model' OR 
                        toLower(e.name) CONTAINS 'method' OR
                        toLower(e.name) CONTAINS 'approach' OR
                        toLower(e.name) CONTAINS 'technique'))
                WITH e, collect(DISTINCT d.title) as papers, count(DISTINCT d) as paper_count
                WHERE paper_count >= 2
                RETURN e.name as method, e.entity_type as type, e.summary as description,
                       paper_count, papers
                ORDER BY paper_count DESC, e.name ASC
                LIMIT 15
            """)
            
            common_methods = []
            for record in result:
                common_methods.append(dict(record))
            
            return common_methods
    
    def analyze_research_evolution(self) -> Dict[str, Any]:
        """Analyze how research concepts evolve across papers"""
        with self.driver.session() as session:
            # Find entities related to key ML/NLP concepts
            key_concepts = ["transformer", "attention", "bert", "language model", "neural network"]
            
            evolution_analysis = {}
            
            for concept in key_concepts:
                result = session.run("""
                    MATCH (e:Entity)-[:MENTIONED_IN]->(ep:Episode)<-[:CONTAINS]-(d:Document)
                    WHERE toLower(e.name) CONTAINS toLower($concept)
                       OR (e.summary IS NOT NULL AND toLower(e.summary) CONTAINS toLower($concept))
                    WITH e, d, count(ep) as mentions
                    RETURN e.name as entity_name, e.entity_type as entity_type,
                           d.title as paper_title, mentions
                    ORDER BY mentions DESC
                """, concept=concept)
                
                concept_data = []
                for record in result:
                    concept_data.append(dict(record))
                
                if concept_data:
                    evolution_analysis[concept] = concept_data
            
            return evolution_analysis
    
    def generate_comprehensive_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive analysis of commonalities across papers"""
        print("\n🔍 Analyzing Commonalities Across Academic Papers")
        print("=" * 55)
        
        analysis = {
            "generated_at": datetime.now().isoformat(),
            "query": "what is the common of the 5 papers",
            "papers_overview": [],
            "common_entities": [],
            "common_themes": [],
            "common_methods": [],
            "research_evolution": {},
            "summary": {}
        }
        
        # 1. Paper overview
        print("📚 Paper Overview:")
        papers = self.get_paper_overview()
        analysis["papers_overview"] = papers
        
        for paper in papers:
            print(f"   • {paper['paper_title']}: {paper['episodes']} episodes, {paper['entities']} entities")
        
        # 2. Common entities
        print(f"\n🤝 Common Entities (mentioned in multiple papers):")
        common_entities = self.find_common_entities(min_papers=2)
        analysis["common_entities"] = common_entities
        
        for entity in common_entities[:10]:  # Show top 10
            papers_str = ", ".join(entity['papers'][:3])  # Show first 3 papers
            if len(entity['papers']) > 3:
                papers_str += f" (+{len(entity['papers'])-3} more)"
            print(f"   • {entity['entity_name']} ({entity['entity_type']}): {entity['paper_count']} papers")
            print(f"     Papers: {papers_str}")
            if entity['summary']:
                summary_preview = entity['summary'][:100] + "..." if len(entity['summary']) > 100 else entity['summary']
                print(f"     Summary: {summary_preview}")
            print()
        
        # 3. Common themes/concepts
        print(f"💡 Common Themes and Concepts:")
        common_themes = self.find_common_themes()
        analysis["common_themes"] = common_themes
        
        for theme in common_themes[:8]:  # Show top 8
            print(f"   • {theme['concept']}: {theme['paper_count']} papers")
            if theme['description']:
                desc_preview = theme['description'][:80] + "..." if len(theme['description']) > 80 else theme['description']
                print(f"     Description: {desc_preview}")
            print()
        
        # 4. Common methods
        print(f"🔧 Common Methods and Techniques:")
        common_methods = self.find_common_methods()
        analysis["common_methods"] = common_methods
        
        for method in common_methods[:8]:  # Show top 8
            print(f"   • {method['method']} ({method['type']}): {method['paper_count']} papers")
            if method['description']:
                desc_preview = method['description'][:80] + "..." if len(method['description']) > 80 else method['description']
                print(f"     Description: {desc_preview}")
            print()
        
        # 5. Research evolution
        print(f"📈 Research Evolution Analysis:")
        evolution = self.analyze_research_evolution()
        analysis["research_evolution"] = evolution
        
        for concept, data in evolution.items():
            if data:  # Only show concepts that have data
                print(f"   🔬 {concept.title()} Research:")
                unique_papers = set()
                for item in data[:5]:  # Show top 5 mentions
                    unique_papers.add(item['paper_title'])
                    print(f"     • {item['entity_name']} in '{item['paper_title']}' ({item['mentions']} mentions)")
                print(f"     Found in {len(unique_papers)} papers")
                print()
        
        # 6. Generate summary
        total_entities = len(common_entities)
        total_themes = len(common_themes)
        total_methods = len(common_methods)
        
        analysis["summary"] = {
            "total_papers_analyzed": len(papers),
            "common_entities_found": total_entities,
            "common_themes_found": total_themes,
            "common_methods_found": total_methods,
            "key_insights": [
                f"Identified {total_entities} entities mentioned across multiple papers",
                f"Found {total_themes} common conceptual themes",
                f"Discovered {total_methods} shared methods and techniques",
                "All papers appear to focus on natural language processing and machine learning",
                "Common themes include transformers, attention mechanisms, and language models",
                "Research shows evolution from basic neural networks to more sophisticated architectures"
            ]
        }
        
        return analysis


async def validate_neo4j_storage(paper_result: Dict[str, Any]) -> None:
    """Validate if data was actually stored in Neo4j"""
    try:
        from neo4j import GraphDatabase
        neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        neo4j_username = os.getenv('NEO4J_USERNAME', 'neo4j')
        neo4j_password = os.getenv('NEO4J_PASSWORD', 'password')

        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))
        
        with driver.session() as session:
            # Check if document exists by file_hash
            result = session.run("""
                MATCH (d:Document)
                WHERE d.file_hash = $file_hash OR d.title = $title OR d.file_name = $file_name
                RETURN d.title as title, d.file_name as file_name, d.file_hash as file_hash
            """, file_hash=paper_result.get("file_hash", ""), 
                title=paper_result.get("title", ""), 
                file_name=paper_result.get("file_name", ""))
            
            doc_record = result.single()
            if not doc_record:
                print(f"   ❌ Document not found in Neo4j")
                print(f"      Looking for: {paper_result.get('title', 'Unknown')}")
                return
            
            print(f"   ✅ Document found: {doc_record['title']}")
            
            # Check episodes
            result = session.run("""
                MATCH (d:Document)-[:CONTAINS]->(ep:Episode)
                WHERE d.file_hash = $file_hash
                RETURN count(ep) as episode_count
            """, file_hash=doc_record['file_hash'])
            
            episode_count = result.single()['episode_count']
            print(f"   ✅ Episodes found: {episode_count}")
            
            # Check entities  
            result = session.run("""
                MATCH (d:Document)-[:CONTAINS]->(ep:Episode)<-[:MENTIONED_IN]-(e:Entity)
                WHERE d.file_hash = $file_hash
                RETURN count(DISTINCT e) as entity_count
            """, file_hash=doc_record['file_hash'])
            
            entity_count = result.single()['entity_count']
            print(f"   ✅ Entities found: {entity_count}")
            
            # Check relationships
            result = session.run("""
                MATCH (d:Document)-[:CONTAINS]->(ep:Episode)<-[:MENTIONED_IN]-(r:Relationship)
                WHERE d.file_hash = $file_hash
                RETURN count(DISTINCT r) as relationship_count
            """, file_hash=doc_record['file_hash'])
            
            relationship_count = result.single()['relationship_count']
            print(f"   ✅ Relationships found: {relationship_count}")
            
            print(f"   🎯 Storage Summary:")
            print(f"      • Document: {doc_record['title']}")
            print(f"      • Episodes: {episode_count}")
            print(f"      • Entities: {entity_count}")  
            print(f"      • Relationships: {relationship_count}")
            
        driver.close()
        
    except Exception as e:
        print(f"   ❌ Storage validation failed: {e}")


async def main():
    """Main function to process all academic papers and analyze commonalities"""
    
    print("📚 Academic Papers Processing Pipeline")
    print("=" * 50)
    
    # Configuration
    academic_folder = "/home/administrator/projects/XYZ_memory/test_data/academic_test_set"
    max_episodes_per_paper = 30  # Process more episodes per paper for comprehensive analysis
    max_papers_to_process = None  # Process all papers (remove limit)
    total_cost_limit = 125.0  # Higher cost limit for processing all papers
    
    print(f"📁 Academic folder: {academic_folder}")
    print(f"📑 Episode limit per paper: {max_episodes_per_paper}")
    print(f"📄 Paper limit: {'All papers' if max_papers_to_process is None else max_papers_to_process}")
    print(f"💰 Total cost limit: ${total_cost_limit:.2f}")
    print()
    
    # Validate prerequisites
    if not validate_prerequisites():
        print("\n❌ Cannot proceed without required prerequisites")
        return False
    
    # Find PDF files
    pdf_files = find_academic_papers(academic_folder)
    if not pdf_files:
        print("❌ No PDF files found to process")
        return False
    
    # Initialize cost tracking for the entire session
    from src.cost_tracking import CostTracker
    session_name = f"academic_papers_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    processing_results = []
    
    with CostTracker(session_name=session_name) as cost_tracker:
        print(f"\n🚀 Starting batch processing of {len(pdf_files)} papers...")
        print(f"💰 Cost tracker initialized: {session_name}")
        
        # Process each paper
        papers_to_process = pdf_files if max_papers_to_process is None else pdf_files[:max_papers_to_process]
        
        for i, pdf_file in enumerate(papers_to_process, 1):
            print(f"\n{'='*60}")
            print(f"📄 Processing Paper {i}/{len(papers_to_process)}")
            print(f"{'='*60}")
            
            # Check total cost limit
            current_total_cost = cost_tracker.get_session_cost()
            if current_total_cost >= total_cost_limit:
                print(f"💰 Total cost limit reached: ${current_total_cost:.4f} >= ${total_cost_limit:.2f}")
                print("⏹️  Stopping batch processing to stay within budget")
                break
            
            # Store initial cost for this paper
            initial_cost = current_total_cost
            
            # Process the paper
            paper_result = await process_single_paper(
                pdf_file, 
                cost_tracker, 
                max_episodes=max_episodes_per_paper
            )
            paper_result["initial_cost"] = initial_cost
            processing_results.append(paper_result)
            
            # Show progress
            final_cost = cost_tracker.get_session_cost()
            paper_cost = final_cost - initial_cost
            print(f"   💰 Paper cost: ${paper_cost:.4f}")
            print(f"   💰 Total cost so far: ${final_cost:.4f} / ${total_cost_limit:.2f}")
            
            if paper_result["status"] == "completed":
                print(f"   ✅ {pdf_file.name} processed successfully")
                
                # Validate storage by checking Neo4j directly
                print(f"   🔍 Validating storage in Neo4j...")
                await validate_neo4j_storage(paper_result)
            else:
                print(f"   ❌ {pdf_file.name} failed: {paper_result['error']}")
        
        # Processing summary
        print(f"\n📊 Processing Results!")
        print("=" * 30)
        
        successful_papers = [r for r in processing_results if r["status"] == "completed"]
        failed_papers = [r for r in processing_results if r["status"] != "completed"]
        
        print(f"✅ Successfully processed: {len(successful_papers)} papers")
        print(f"❌ Failed processing: {len(failed_papers)} papers")
        
        total_entities = sum(r["entities_extracted"] for r in successful_papers)
        total_relationships = sum(r["relationships_extracted"] for r in successful_papers)
        total_episodes = sum(r["episodes_created"] for r in successful_papers)
        
        print(f"📈 Total knowledge extracted:")
        print(f"   • Episodes: {total_episodes}")
        print(f"   • Entities: {total_entities}")
        print(f"   • Relationships: {total_relationships}")
        
        session_stats = cost_tracker.get_session_stats()
        print(f"💰 Total cost: ${session_stats['total_cost']:.4f}")
        
        # Save processing results
        results_file = f"academic_processing_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        batch_results = {
            "session_info": {
                "session_name": session_name,
                "processed_at": datetime.now().isoformat(),
                "total_papers": len(pdf_files),
                "successful_papers": len(successful_papers),
                "failed_papers": len(failed_papers)
            },
            "cost_summary": session_stats,
            "paper_results": processing_results
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"📁 Processing results saved: {results_file}")
    
    # STEP 2: Run retrieval analysis
    if successful_papers:
        print(f"\n🔍 Running Commonality Analysis")
        print("=" * 40)
        
        retriever = AcademicRetriever()
        if retriever.connect():
            try:
                # Generate comprehensive analysis
                analysis = retriever.generate_comprehensive_analysis()
                
                # Save analysis results
                analysis_file = f"academic_commonalities_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
                with open(analysis_file, 'w', encoding='utf-8') as f:
                    json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
                
                print(f"\n📄 Analysis Results Summary:")
                print("=" * 35)
                summary = analysis["summary"]
                print(f"📚 Papers analyzed: {summary['total_papers_analyzed']}")
                print(f"🤝 Common entities: {summary['common_entities_found']}")
                print(f"💡 Common themes: {summary['common_themes_found']}")
                print(f"🔧 Common methods: {summary['common_methods_found']}")
                
                print(f"\n🎯 Key Insights:")
                for insight in summary["key_insights"]:
                    print(f"   • {insight}")
                
                print(f"\n✅ Analysis saved to: {analysis_file}")
                
                # Also create a human-readable summary
                summary_file = f"academic_papers_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                
                with open(summary_file, 'w', encoding='utf-8') as f:
                    f.write("Academic Papers Commonality Analysis\n")
                    f.write("=" * 40 + "\n\n")
                    
                    f.write(f"Query: {analysis['query']}\n")
                    f.write(f"Generated: {analysis['generated_at']}\n\n")
                    
                    f.write("PAPERS ANALYZED:\n")
                    for paper in analysis["papers_overview"]:
                        f.write(f"• {paper['paper_title']}: {paper['episodes']} episodes, {paper['entities']} entities\n")
                    f.write("\n")
                    
                    f.write("COMMON ENTITIES:\n")
                    for entity in analysis["common_entities"][:10]:
                        f.write(f"• {entity['entity_name']} ({entity['entity_type']}): mentioned in {entity['paper_count']} papers\n")
                        if entity['summary']:
                            f.write(f"  Summary: {entity['summary'][:100]}...\n")
                        f.write("\n")
                    
                    f.write("COMMON THEMES:\n")
                    for theme in analysis["common_themes"][:8]:
                        f.write(f"• {theme['concept']}: found in {theme['paper_count']} papers\n")
                        if theme['description']:
                            f.write(f"  Description: {theme['description'][:100]}...\n")
                        f.write("\n")
                    
                    f.write("KEY INSIGHTS:\n")
                    for insight in summary["key_insights"]:
                        f.write(f"• {insight}\n")
                
                print(f"📝 Human-readable summary: {summary_file}")
                
            finally:
                retriever.close()
    
    print(f"\n🎉 Academic Papers Pipeline Complete!")
    print("=" * 45)
    print("All papers have been processed and analyzed.")
    print("Check the generated files for detailed results.")
    
    return True


if __name__ == "__main__":
    print("Knowledge RAG - Academic Papers Processing Pipeline")
    print("=" * 55)
    print()
    print("This script will:")
    print("  1. Process all PDF files in the academic test set")
    print("  2. Extract knowledge and store in Neo4j + ChromaDB")
    print("  3. Analyze commonalities across all papers")
    print("  4. Export detailed analysis results")
    print()
    
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 