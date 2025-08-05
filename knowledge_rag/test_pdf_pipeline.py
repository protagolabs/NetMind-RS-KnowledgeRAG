#!/usr/bin/env python3
"""
Complete PDF Processing Pipeline

Processes a PDF file through the complete Knowledge RAG pipeline:
1. Parse PDF into episodes using FileManager
2. Extract entities and relationships using KnowledgeExtractor  
3. Store directly in Neo4j + ChromaDB using StorageManager
4. Track costs and provide detailed analytics

This combines and extends the functionality from test_knowledge_extraction.py 
and test_store_from_json.py for a seamless PDF-to-storage workflow.
"""

import sys
import os
import tempfile
import asyncio
from pathlib import Path
from datetime import datetime
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


async def process_pdf_pipeline(pdf_path: str, max_episodes: int = None, cost_limit: float = 5.0):
    """
    Complete PDF processing pipeline
    
    Args:
        pdf_path: Path to the PDF file to process
        max_episodes: Maximum number of episodes to process (None for all)
        cost_limit: Maximum cost limit in USD
    """
    
    print("🚀 Knowledge RAG - Complete PDF Pipeline")
    print("=" * 60)
    
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        print(f"❌ PDF file not found: {pdf_path}")
        return False
    
    print(f"📄 Processing PDF: {pdf_file.name}")
    print(f"📁 File size: {pdf_file.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"💰 Cost limit: ${cost_limit:.2f}")
    if max_episodes:
        print(f"📑 Episode limit: {max_episodes}")
    
    try:
        # Import required modules
        from src.file_management import FileManager
        from src.file_management.models import FileStatus
        from src.knowledge_extraction import KnowledgeExtractor
        from src.knowledge_extraction.models import Document
        from src.storage import StorageManager
        from src.cost_tracking import CostTracker
        
        print("✅ All modules imported successfully")
        
        # STEP 1: Parse PDF into episodes
        print(f"\n📖 STEP 1: Parsing PDF")
        print("-" * 30)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "pdf_processing.db"
            
            # Initialize file manager
            file_manager = FileManager(
                root_folder=str(pdf_file.parent),
                db_path=str(db_path)
            )
            
            print("✅ FileManager initialized")
            
            # Process the PDF file
            result = file_manager.process_file(pdf_file)
            
            if not result or result.status != FileStatus.COMPLETED:
                print(f"❌ Failed to process PDF: {result.error_message if result else 'Unknown error'}")
                return False
            
            print(f"✅ PDF processed: {result.episodes_created} episodes created")
            print(f"📊 Processing stats:")
            print(f"   • File format: {result.metadata.file_format.value}")
            print(f"   • File hash: {result.metadata.file_hash[:12]}...")
            print(f"   • Episodes: {result.episodes_created}")
            
            # Get parsing result to access episodes
            parser = file_manager.parser_factory.get_parser(result.metadata.file_format)
            parsing_result = parser.parse(pdf_file, result.metadata)
            
            episodes = parsing_result.episodes
            print(f"📄 Retrieved {len(episodes)} episodes from PDF")
            
            # Show episode breakdown by type
            episode_types = {}
            for episode in episodes:
                ep_type = episode.episode_type
                if ep_type not in episode_types:
                    episode_types[ep_type] = []
                episode_types[ep_type].append(episode)
            
            print(f"\n📋 Episode Breakdown:")
            for ep_type, type_episodes in episode_types.items():
                avg_length = sum(len(ep.content) for ep in type_episodes) / len(type_episodes)
                print(f"   • {ep_type}: {len(type_episodes)} episodes (avg: {avg_length:.0f} chars)")
            
            # Apply episode limit if specified
            if max_episodes and len(episodes) > max_episodes:
                episodes = episodes[:max_episodes]
                print(f"⚠️  Limited to first {max_episodes} episodes for testing")
            
            # STEP 2: Create document representation
            print(f"\n📚 STEP 2: Creating Document")
            print("-" * 35)
            
            document = Document(
                file_path=str(pdf_file),
                file_name=pdf_file.name,
                file_hash=result.metadata.file_hash,
                title=pdf_file.stem,  # Use filename without extension as title
                document_type="pdf",
                created_at=datetime.fromtimestamp(pdf_file.stat().st_ctime),
                modified_at=datetime.fromtimestamp(pdf_file.stat().st_mtime)
            )
            
            # Add episodes to document
            for episode in episodes:
                document.add_episode(episode.id)
            
            print(f"✅ Document created: {document.title}")
            print(f"📄 Document UUID: {document.uuid}")
            print(f"📑 Episodes linked: {len(document.episode_ids)}")
            
            # STEP 3: Extract knowledge with cost tracking
            print(f"\n🧠 STEP 3: Knowledge Extraction")
            print("-" * 40)
            
            session_name = f"pdf_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            with CostTracker(session_name=session_name) as cost_tracker:
                print(f"💰 Cost tracker initialized: {session_name}")
                
                # Initialize knowledge extractor
                extractor = KnowledgeExtractor(
                    model_name="gpt-3.5-turbo",  # Use cost-effective model
                    temperature=0.1,
                    max_retries=3,
                    cost_tracker=cost_tracker
                )
                
                print("✅ KnowledgeExtractor initialized with cost tracking")
                
                # Process episodes for knowledge extraction
                all_entities = []
                all_relationships = []
                successful_extractions = 0
                
                print(f"\n🔬 Processing {len(episodes)} episodes for knowledge extraction...")
                
                for i, episode in enumerate(episodes):
                    print(f"\n--- Episode {i+1}/{len(episodes)} ---")
                    
                    # Get episode metadata
                    content_length = len(episode.content)
                    page_info = episode.metadata.get('page_number', 'Unknown')
                    
                    print(f"📄 Page: {page_info}")
                    print(f"📏 Length: {content_length:,} characters")
                    print(f"🏷️  Type: {episode.episode_type}")
                    
                    # Show content preview
                    content_preview = episode.content[:100].replace('\n', ' ')
                    print(f"📝 Preview: {content_preview}...")
                    
                    # Check cost limit before processing
                    current_cost = cost_tracker.get_session_cost()
                    cost_check = cost_tracker.check_cost_limit(cost_limit)
                    
                    if cost_check['exceeded']:
                        print(f"💰 Cost limit exceeded: ${current_cost:.4f} >= ${cost_limit:.2f}")
                        print(f"⏹️  Stopping extraction to stay within budget")
                        break
                    
                    print(f"💸 Current cost: ${current_cost:.4f} / ${cost_limit:.2f} ({cost_check['percentage_used']:.1f}%)")
                    
                    # Extract knowledge from episode
                    extraction_result = extractor.extract_from_episode(
                        episode=episode,
                        document=document,
                        existing_entities=all_entities,
                        existing_relationships=all_relationships
                    )
                    
                    if extraction_result.success:
                        print(f"✅ Success!")
                        print(f"   📊 Entities extracted: {extraction_result.entities_extracted}")
                        print(f"   🔗 Relationships extracted: {extraction_result.relationships_extracted}")
                        print(f"   🔄 Entities deduplicated: {extraction_result.entities_deduplicated}")
                        print(f"   🔄 Relationships deduplicated: {extraction_result.relationships_deduplicated}")
                        print(f"   ⏱️  Processing time: {extraction_result.processing_time:.2f}s")
                        
                        # Update knowledge base
                        all_entities = extraction_result.entities
                        all_relationships = extraction_result.relationships
                        successful_extractions += 1
                        
                        # Show running totals
                        session_cost = cost_tracker.get_session_cost()
                        print(f"   💰 Running cost: ${session_cost:.4f}")
                        print(f"   📈 Total entities: {len(all_entities)}")
                        print(f"   📈 Total relationships: {len(all_relationships)}")
                        
                    else:
                        print(f"❌ Failed: {extraction_result.error_message}")
                
                # Final extraction statistics
                print(f"\n📊 Knowledge Extraction Complete!")
                print("=" * 50)
                
                extractor_stats = extractor.get_stats()
                session_stats = cost_tracker.get_session_stats()
                
                print(f"📈 Extraction Results:")
                print(f"   • Episodes processed: {successful_extractions}/{len(episodes)}")
                print(f"   • Total entities: {len(all_entities)}")
                print(f"   • Total relationships: {len(all_relationships)}")
                print(f"   • API calls made: {extractor_stats['api_calls_made']}")
                print(f"   • Total processing time: {extractor_stats['total_processing_time']:.2f}s")
                print(f"   • Success rate: {(successful_extractions/len(episodes)*100):.1f}%")
                
                print(f"\n💰 Cost Analysis:")
                print(f"   • Session duration: {session_stats['duration_seconds']:.1f} seconds")
                print(f"   • Total cost: ${session_stats['total_cost']:.4f}")
                print(f"   • Total tokens: {session_stats['total_tokens']:,}")
                print(f"   • Cost per token: ${session_stats['total_cost'] / session_stats['total_tokens']:.6f}" if session_stats['total_tokens'] > 0 else "   • Cost per token: $0.000000")
                print(f"   • Average cost per call: ${session_stats['avg_cost_per_call']:.4f}")
                
                # Show sample extracted knowledge
                if all_entities:
                    print(f"\n🏷️  Sample Extracted Entities:")
                    entity_types = {}
                    for entity in all_entities:
                        entity_type = entity.entity_type.value
                        if entity_type not in entity_types:
                            entity_types[entity_type] = []
                        entity_types[entity_type].append(entity)
                    
                    for entity_type, type_entities in list(entity_types.items())[:5]:  # Show first 5 types
                        print(f"   {entity_type} ({len(type_entities)}):")
                        for entity in type_entities[:3]:  # Show first 3 of each type
                            confidence_str = f"({entity.confidence:.2f})" if entity.confidence > 0 else ""
                            print(f"     • {entity.name} {confidence_str}")
                        if len(type_entities) > 3:
                            print(f"     ... and {len(type_entities) - 3} more")
                        print()
                
                if all_relationships:
                    print(f"🔗 Sample Extracted Relationships:")
                    rel_count = 0
                    for relationship in all_relationships[:5]:  # Show first 5 relationships
                        fact_preview = relationship.fact[:60] + "..." if len(relationship.fact) > 60 else relationship.fact
                        confidence_str = f"({relationship.confidence:.2f})" if relationship.confidence > 0 else ""
                        print(f"   • {fact_preview} {confidence_str}")
                        rel_count += 1
                    if len(all_relationships) > 5:
                        print(f"   ... and {len(all_relationships) - 5} more")
                    print()
                
                # Update document knowledge statistics
                document.update_knowledge_stats(all_entities, all_relationships)
                
                # STEP 4: Store in databases
                print(f"\n🗄️  STEP 4: Storing in Databases")
                print("-" * 38)
                
                # Initialize storage manager
                storage_manager = StorageManager(
                    # Neo4j configuration
                    neo4j_uri=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
                    neo4j_username=os.getenv('NEO4J_USERNAME', 'neo4j'),
                    neo4j_password=os.getenv('NEO4J_PASSWORD', 'password'),
                    
                    # ChromaDB configuration
                    chroma_persist_dir="./pdf_pipeline_chroma_db",
                    openai_api_key=os.getenv('OPENAI_API_KEY')
                )
                
                print("🔌 Connecting to databases...")
                await storage_manager.initialize()
                print("✅ Connected to Neo4j and ChromaDB")
                
                # Store the complete knowledge base
                print("💾 Storing knowledge in databases...")
                storage_results = await storage_manager.store_knowledge_base(
                    document=document,
                    episodes=episodes,
                    entities=all_entities,
                    relationships=all_relationships
                )
                
                print("✅ Knowledge stored successfully:")
                for key, value in storage_results.items():
                    print(f"   • {key}: {value}")
                
                # Get storage statistics
                print("\n📊 Storage Statistics:")
                stats = await storage_manager.get_storage_statistics()
                
                print("   Neo4j Graph Database:")
                graph_stats = stats.get('graph_database', {})
                print(f"     • Documents: {graph_stats.get('total_documents', 0)}")
                print(f"     • Episodes: {graph_stats.get('total_episodes', 0)}")
                print(f"     • Entities: {graph_stats.get('total_entities', 0)}")
                print(f"     • Relationships: {graph_stats.get('total_relationships', 0)}")
                
                if graph_stats.get('entities_by_type'):
                    print("     • Entity types:")
                    for entity_type, count in graph_stats['entities_by_type'].items():
                        print(f"       - {entity_type}: {count}")
                
                print("   ChromaDB Vector Database:")
                vector_stats = stats.get('vector_database', {})
                entities_count = vector_stats.get('entities', {}).get('count', 0)
                episodes_count = vector_stats.get('episodes', {}).get('count', 0)
                print(f"     • Entity embeddings: {entities_count}")
                print(f"     • Episode embeddings: {episodes_count}")
                
                # STEP 5: Test search functionality
                print(f"\n🔍 STEP 5: Testing Search")
                print("-" * 28)
                
                from src.search import SearchEngine
                
                # Initialize search engine
                search_engine = SearchEngine(storage_manager)
                await search_engine.warm_up()
                print("✅ Search engine initialized")
                
                # Test searches based on document content
                test_queries = [
                    "weak supervision",
                    "strong capabilities", 
                    "generalization",
                    "machine learning",
                    "neural networks"
                ]
                
                print(f"🔎 Testing search with sample queries...")
                for query in test_queries:
                    try:
                        results = await search_engine.search_text(query, limit=3)
                        print(f"   '{query}': {results.total_results} results ({results.search_time_ms:.1f}ms)")
                        
                        if results.results:
                            top_result = results.results[0]
                            content_preview = top_result.get_display_content()[:60] + "..."
                            print(f"     Top: [{top_result.score:.3f}] {content_preview}")
                    except Exception as e:
                        print(f"   '{query}': Search failed - {e}")
                
                # Clean up connections
                await storage_manager.close()
                print("✅ Database connections closed")
                
                # STEP 6: Generate final report
                print(f"\n📋 STEP 6: Final Report")
                print("-" * 25)
                
                total_session_cost = cost_tracker.get_session_cost()
                final_cost_check = cost_tracker.check_cost_limit(cost_limit)
                
                print(f"🎉 PDF Pipeline Complete!")
                print("=" * 50)
                print(f"📄 Document: {pdf_file.name}")
                print(f"📑 Episodes processed: {successful_extractions}/{len(episodes)}")
                print(f"🧠 Knowledge extracted:")
                print(f"   • Entities: {len(all_entities)}")
                print(f"   • Relationships: {len(all_relationships)}")
                print(f"💾 Storage:")
                print(f"   • Neo4j nodes/edges: {storage_results.get('total_stored', 'N/A')}")
                print(f"   • ChromaDB embeddings: {storage_results.get('entity_embeddings_stored', 0) + storage_results.get('episode_embeddings_stored', 0)}")
                print(f"💰 Cost summary:")
                print(f"   • Total cost: ${total_session_cost:.4f}")
                print(f"   • Budget used: {final_cost_check['percentage_used']:.1f}%")
                print(f"   • Remaining budget: ${final_cost_check['remaining']:.4f}")
                
                # Export cost report
                try:
                    cost_report_path = cost_tracker.export_session_report()
                    print(f"📊 Cost report exported: {Path(cost_report_path).name}")
                except Exception as e:
                    print(f"⚠️  Could not export cost report: {e}")
                
                print(f"\n🚀 Next Steps:")
                print(f"   • View graph: http://localhost:7474 (Neo4j Browser)")
                print(f"   • Test search: python test_search_knowledge.py")
                print(f"   • View costs: python cost_analysis.py")
                
                return True
    
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you've installed required dependencies:")
        print("pip install -r requirements.txt")
        return False
    
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main function with command line argument handling"""
    
    # Default PDF path
    default_pdf = "/home/administrator/projects/XYZ_memory/paper_sets/paper_set_1/docs/Weak-to-Strong Generalization Eliciting Strong Capabilities with Weak Supervision.pdf"
    
    # Parse command line arguments
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else default_pdf
    max_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else None
    cost_limit = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
    
    print(f"🎯 PDF Pipeline Configuration:")
    print(f"   PDF: {pdf_path}")
    if max_episodes:
        print(f"   Episode limit: {max_episodes}")
    print(f"   Cost limit: ${cost_limit:.2f}")
    print()
    
    # Validate prerequisites
    if not validate_prerequisites():
        print("\n❌ Cannot proceed without required prerequisites")
        print("Please fix the issues above and try again")
        return False
    
    # Run the pipeline
    success = await process_pdf_pipeline(pdf_path, max_episodes, cost_limit)
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 PDF pipeline completed successfully!")
        print("\nThis demonstrated:")
        print("  ✅ Complete PDF parsing (pages, structure, content)")
        print("  ✅ Advanced knowledge extraction (entities, relationships)")
        print("  ✅ Deduplication and temporal processing")
        print("  ✅ Direct storage in Neo4j + ChromaDB")
        print("  ✅ Comprehensive cost tracking with aggregation")
        print("  ✅ Search functionality testing")
        print("  ✅ Full document traceability")
        print("\nYour PDF knowledge is now fully integrated into the RAG system! 🚀")
    else:
        print("❌ PDF pipeline failed")
        print("Check the error messages above for troubleshooting")
    
    return success


if __name__ == "__main__":
    print("Knowledge RAG - Complete PDF Processing Pipeline")
    print("=" * 55)
    print()
    print("Usage:")
    print("  python test_pdf_pipeline.py [pdf_path] [max_episodes] [cost_limit]")
    print()
    print("Examples:")
    print("  python test_pdf_pipeline.py  # Use default PDF")
    print("  python test_pdf_pipeline.py /path/to/paper.pdf  # Custom PDF")
    print("  python test_pdf_pipeline.py /path/to/paper.pdf 10 2.0  # Limit to 10 episodes, $2 budget")
    print()
    
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 