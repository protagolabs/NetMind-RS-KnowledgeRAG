#!/usr/bin/env python3
"""
Test Knowledge Extraction with Cost Tracking

Tests the knowledge extraction system with comprehensive cost tracking and analysis.
"""

import sys
import os
import tempfile
from pathlib import Path
from datetime import datetime
import json
import pickle

# Add the current directory to Python path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️  python-dotenv not installed, trying system environment variables")

def test_knowledge_extraction():
    print("🧠 Testing Knowledge Extraction with Cost Tracking")
    print("=" * 70)
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OpenAI API key not found in environment variables")
        print("Please set OPENAI_API_KEY in your .env file to run this test")
        return False
    
    try:
        # Import our modules
        from src.file_management import FileManager
        from src.file_management.models import FileStatus
        from src.knowledge_extraction import KnowledgeExtractor
        from src.knowledge_extraction.models import Document
        from src.cost_tracking import CostTracker
        
        print("✅ Successfully imported modules")
        
        # Path to the design document
        markdown_file = Path("Knowledge_RAG_Design_Document.md")
        
        if not markdown_file.exists():
            print(f"❌ Test file not found: {markdown_file}")
            print("Please make sure Knowledge_RAG_Design_Document.md exists in the current directory")
            return False
        
        print(f"📄 Processing test file: {markdown_file.name}")
        print(f"📁 File size: {markdown_file.stat().st_size:,} bytes")
        
        # Process the markdown file to get episodes
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_extraction.db"
            
            file_manager = FileManager(
                root_folder=str(markdown_file.parent),
                db_path=str(db_path)
            )
            
            print("✅ FileManager initialized")
            
            # Process the file
            result = file_manager.process_file(markdown_file)
            
            if not result or result.status != FileStatus.COMPLETED:
                print("❌ Failed to process file")
                return False
            
            print(f"✅ File processed: {result.episodes_created} episodes created")
            
            # Get the parsing result to access episodes
            parser = file_manager.parser_factory.get_parser(result.metadata.file_format)
            parsing_result = parser.parse(markdown_file, result.metadata)
            
            episodes = parsing_result.episodes
            print(f"📄 Retrieved {len(episodes)} episodes for testing")
            
            # Show episode breakdown
            print("\n📋 Episode Structure:")
            episode_structure = {}
            for episode in episodes:
                header_level = episode.metadata.get('header_level', 0)
                if header_level not in episode_structure:
                    episode_structure[header_level] = []
                episode_structure[header_level].append(episode)
            
            for level in sorted(episode_structure.keys()):
                episodes_at_level = episode_structure[level]
                level_name = {
                    1: "Main Sections (#)",
                    2: "Sections (##)", 
                    3: "Subsections (###)",
                    4: "Sub-subsections (####)",
                    5: "Detailed sections (#####)",
                    6: "Minor sections (######)"
                }.get(level, f"Level {level}")
                
                print(f"   {level_name}: {len(episodes_at_level)} episodes")
            
            # Create document representation
            document = Document(
                file_path=str(markdown_file),
                file_name=markdown_file.name,
                file_hash=result.metadata.file_hash,
                title="Knowledge RAG Design Document",
                document_type="markdown",
                created_at=datetime.fromtimestamp(markdown_file.stat().st_ctime),
                modified_at=datetime.fromtimestamp(markdown_file.stat().st_mtime)
            )
            
            # Add episodes to document
            for episode in episodes:
                document.add_episode(episode.id)
            
            print("✅ Document object created")
            
            # Initialize cost tracker with session name
            session_name = f"test_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            with CostTracker(session_name=session_name) as cost_tracker:
                print(f"💰 Cost tracker initialized: {session_name}")
                
                # Initialize knowledge extractor with cost tracking
                extractor = KnowledgeExtractor(
                    model_name="gpt-3.5-turbo",  # Use cheaper model for testing
                    temperature=0.1,
                    max_retries=3,
                    cost_tracker=cost_tracker
                )
                
                print("✅ KnowledgeExtractor initialized with cost tracking")
                
                # Process a few episodes for testing
                all_entities = []
                all_relationships = []
                
                # Test with first 3 episodes (or fewer if less available)
                test_episodes = episodes[:3]
                print(f"\n🧠 Starting knowledge extraction for {len(test_episodes)} episodes...")
                
                for i, episode in enumerate(test_episodes):
                    header_text = episode.metadata.get('header_text', 'No header')
                    content_length = len(episode.content)
                    
                    print(f"\n--- Processing Episode {i+1}/{len(test_episodes)} ---")
                    print(f"📄 Episode: '{header_text}'")
                    print(f"📏 Length: {content_length:,} characters")
                    print(f"🏷️  Type: {episode.episode_type}")
                    
                    # Estimate cost before extraction
                    if cost_tracker:
                        # Create sample messages for estimation
                        sample_messages = [
                            {"role": "system", "content": "You are an entity extraction AI."},
                            {"role": "user", "content": f"Extract entities from: {episode.content[:500]}..."}
                        ]
                        
                        estimate = cost_tracker.estimate_cost_for_messages(sample_messages, "gpt-3.5-turbo")
                        print(f"💸 Estimated cost: ${estimate.get('estimated_cost', 0):.4f}")
                    
                    # Extract knowledge
                    extraction_result = extractor.extract_from_episode(
                        episode=episode,
                        document=document,
                        existing_entities=all_entities,
                        existing_relationships=all_relationships
                    )
                    
                    if extraction_result.success:
                        print(f"✅ Success!")
                        print(f"   📊 Entities: {extraction_result.entities_extracted}")
                        print(f"   🔗 Relationships: {extraction_result.relationships_extracted}")
                        print(f"   🔄 Deduplicated entities: {extraction_result.entities_deduplicated}")
                        print(f"   🔄 Deduplicated relationships: {extraction_result.relationships_deduplicated}")
                        print(f"   ⏱️  Processing time: {extraction_result.processing_time:.2f}s")
                        
                        # Update knowledge base
                        all_entities = extraction_result.entities
                        all_relationships = extraction_result.relationships
                        
                        # Show current session costs
                        session_cost = cost_tracker.get_session_cost()
                        print(f"   💰 Session cost so far: ${session_cost:.4f}")
                        
                    else:
                        print(f"❌ Failed: {extraction_result.error_message}")
                
                # Final results
                print(f"\n🎉 Knowledge Extraction Complete!")
                print("=" * 50)
                
                # Show extraction statistics
                extractor_stats = extractor.get_stats()
                print(f"📈 Extraction Statistics:")
                print(f"   Episodes processed: {extractor_stats['episodes_processed']}")
                print(f"   Total entities: {len(all_entities)}")
                print(f"   Total relationships: {len(all_relationships)}")
                print(f"   Entities extracted: {extractor_stats['entities_extracted']}")
                print(f"   Relationships extracted: {extractor_stats['relationships_extracted']}")
                print(f"   Entities deduplicated: {extractor_stats['entities_deduplicated']}")
                print(f"   Relationships deduplicated: {extractor_stats['relationships_deduplicated']}")
                print(f"   API calls made: {extractor_stats['api_calls_made']}")
                print(f"   Total processing time: {extractor_stats['total_processing_time']:.2f}s")
                
                if extractor_stats['errors']:
                    print(f"   ⚠️  Errors: {len(extractor_stats['errors'])}")
                
                # Show cost tracking session summary (will be printed by context manager)
                # But let's also show some detailed cost info
                session_stats = cost_tracker.get_session_stats()
                print(f"\n💰 Cost Analysis:")
                print(f"   Session duration: {session_stats['duration_seconds']:.1f} seconds")
                print(f"   Total cost: ${session_stats['total_cost']:.4f}")
                print(f"   Total tokens: {session_stats['total_tokens']:,}")
                print(f"   Average cost per call: ${session_stats['avg_cost_per_call']:.4f}")
                print(f"   Success rate: {session_stats['success_rate']:.1f}%")
                print(f"   Calls per minute: {session_stats['calls_per_minute']:.1f}")
                
                if session_stats['total_tokens'] > 0:
                    cost_per_token = session_stats['total_cost'] / session_stats['total_tokens']
                    print(f"   Cost per token: ${cost_per_token:.6f}")
                
                # Check if we exceeded any reasonable cost limits
                cost_limit_check = cost_tracker.check_cost_limit(1.0)  # $1.00 limit
                if cost_limit_check['exceeded']:
                    print(f"   ⚠️  Cost limit exceeded: ${cost_limit_check['current_cost']:.4f} > ${cost_limit_check['limit']:.2f}")
                else:
                    print(f"   ✅ Within cost limit: ${cost_limit_check['current_cost']:.4f} / ${cost_limit_check['limit']:.2f} ({cost_limit_check['percentage_used']:.1f}%)")
                
                # Show sample extracted entities by type
                if all_entities:
                    print(f"\n🏷️  Sample Extracted Entities:")
                    entity_types = {}
                    for entity in all_entities:
                        entity_type = entity.entity_type.value
                        if entity_type not in entity_types:
                            entity_types[entity_type] = []
                        entity_types[entity_type].append(entity)
                    
                    for entity_type, type_entities in entity_types.items():
                        print(f"   {entity_type} ({len(type_entities)}):")
                        for entity in type_entities[:3]:  # Show first 3 of each type
                            confidence_str = f"({entity.confidence:.2f})" if entity.confidence > 0 else ""
                            aliases_str = f" [aliases: {', '.join(entity.aliases[:2])}]" if entity.aliases else ""
                            print(f"     • {entity.name} {confidence_str}{aliases_str}")
                        if len(type_entities) > 3:
                            print(f"     ... and {len(type_entities) - 3} more")
                        print()
                
                # Show sample extracted relationships by type
                if all_relationships:
                    print(f"🔗 Sample Extracted Relationships:")
                    relationship_types = {}
                    for relationship in all_relationships:
                        rel_type = relationship.relationship_type.value
                        if rel_type not in relationship_types:
                            relationship_types[rel_type] = []
                        relationship_types[rel_type].append(relationship)
                    
                    for rel_type, type_relationships in relationship_types.items():
                        print(f"   {rel_type} ({len(type_relationships)}):")
                        for relationship in type_relationships[:2]:  # Show first 2 of each type
                            confidence_str = f"({relationship.confidence:.2f})" if relationship.confidence > 0 else ""
                            fact_preview = relationship.fact[:50] + "..." if len(relationship.fact) > 50 else relationship.fact
                            print(f"     • {fact_preview} {confidence_str}")
                        if len(type_relationships) > 2:
                            print(f"     ... and {len(type_relationships) - 2} more")
                        print()
                
                # Export session report for analysis
                print(f"\n📊 Exporting detailed cost report...")
                try:
                    report_path = cost_tracker.export_session_report()
                    print(f"✅ Cost report exported to: {report_path}")
                except Exception as e:
                    print(f"⚠️  Failed to export cost report: {e}")
                
                # Save extracted entities and relationships to files
                print(f"\n💾 Saving extracted knowledge to files...")
                try:
                    # Create test output directory
                    test_output_dir = Path("test_extraction_output")
                    test_output_dir.mkdir(exist_ok=True)
                    
                    # Generate timestamp for unique filenames
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    
                    # Save as JSON (human readable)
                    test_knowledge = {
                        'metadata': {
                            'test_name': 'knowledge_extraction_test',
                            'timestamp': datetime.now().isoformat(),
                            'session_id': session_name,
                            'episodes_tested': len(test_episodes),
                            'extraction_statistics': extractor_stats,
                            'cost_statistics': session_stats
                        },
                        'document_info': {
                            'uuid': document.uuid,
                            'file_path': document.file_path,
                            'file_name': document.file_name,
                            'title': document.title
                        },
                        'entities': [
                            {
                                'uuid': entity.uuid,
                                'name': entity.name,
                                'entity_type': entity.entity_type.value,
                                'summary': entity.summary,
                                'aliases': entity.aliases,
                                'attributes': entity.attributes,
                                'confidence': entity.confidence,
                                'created_at': entity.created_at.isoformat(),
                                'first_mentioned_at': entity.first_mentioned_at.isoformat() if entity.first_mentioned_at else None,
                                'source_episodes': entity.source_episodes,
                                'source_documents': entity.source_documents
                            }
                            for entity in all_entities
                        ],
                        'relationships': [
                            {
                                'uuid': relationship.uuid,
                                'source_entity_id': relationship.source_entity_id,
                                'target_entity_id': relationship.target_entity_id,
                                'relationship_type': relationship.relationship_type.value,
                                'fact': relationship.fact,
                                'confidence': relationship.confidence,
                                'created_at': relationship.created_at.isoformat(),
                                'valid_at': relationship.valid_at.isoformat() if relationship.valid_at else None,
                                'invalid_at': relationship.invalid_at.isoformat() if relationship.invalid_at else None,
                                'attributes': relationship.attributes,
                                'source_episodes': relationship.source_episodes,
                                'source_documents': relationship.source_documents
                            }
                            for relationship in all_relationships
                        ],
                        'test_episodes': [
                            {
                                'id': episode.id,
                                'content_preview': episode.content[:200] + "..." if len(episode.content) > 200 else episode.content,
                                'episode_type': episode.episode_type,
                                'sequence_number': episode.sequence_number,
                                'timestamp': episode.timestamp.isoformat(),
                                'metadata': episode.metadata
                            }
                            for episode in test_episodes
                        ]
                    }
                    
                    # Save JSON file
                    json_file = test_output_dir / f"test_extraction_{timestamp}.json"
                    with open(json_file, 'w', encoding='utf-8') as f:
                        json.dump(test_knowledge, f, indent=2, ensure_ascii=False)
                    
                    print(f"✅ JSON saved: {json_file}")
                    
                    # Save Python objects (for easy loading)
                    pickle_file = test_output_dir / f"test_extraction_{timestamp}.pkl"
                    pickle_data = {
                        'document': document,
                        'entities': all_entities,
                        'relationships': all_relationships,
                        'test_episodes': test_episodes,
                        'extraction_statistics': extractor_stats,
                        'cost_statistics': session_stats,
                        'session_name': session_name
                    }
                    
                    with open(pickle_file, 'wb') as f:
                        pickle.dump(pickle_data, f)
                    
                    print(f"✅ Pickle saved: {pickle_file}")
                    
                    # Create a summary file
                    summary_file = test_output_dir / f"test_summary_{timestamp}.txt"
                    with open(summary_file, 'w', encoding='utf-8') as f:
                        f.write(f"Knowledge Extraction Test Summary\n")
                        f.write(f"=" * 40 + "\n\n")
                        f.write(f"Test Session: {session_name}\n")
                        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                        f.write(f"Episodes Tested: {len(test_episodes)}\n\n")
                        
                        f.write(f"Extraction Results:\n")
                        f.write(f"- Total Entities: {len(all_entities)}\n")
                        f.write(f"- Total Relationships: {len(all_relationships)}\n")
                        f.write(f"- API Calls: {extractor_stats['api_calls_made']}\n")
                        f.write(f"- Processing Time: {extractor_stats['total_processing_time']:.2f}s\n")
                        f.write(f"- Total Cost: ${session_stats['total_cost']:.4f}\n\n")
                        
                        f.write(f"Entity Types Found:\n")
                        entity_types = {}
                        for entity in all_entities:
                            entity_type = entity.entity_type.value
                            if entity_type not in entity_types:
                                entity_types[entity_type] = []
                            entity_types[entity_type].append(entity.name)
                        
                        for entity_type, names in entity_types.items():
                            f.write(f"- {entity_type}: {len(names)} entities\n")
                            for name in names[:5]:  # First 5
                                f.write(f"  • {name}\n")
                            if len(names) > 5:
                                f.write(f"  ... and {len(names) - 5} more\n")
                            f.write("\n")
                        
                        if all_relationships:
                            f.write(f"Relationship Types Found:\n")
                            rel_types = {}
                            for rel in all_relationships:
                                rel_type = rel.relationship_type.value
                                if rel_type not in rel_types:
                                    rel_types[rel_type] = []
                                rel_types[rel_type].append(rel.fact)
                            
                            for rel_type, facts in rel_types.items():
                                f.write(f"- {rel_type}: {len(facts)} relationships\n")
                                for fact in facts[:3]:  # First 3
                                    f.write(f"  • {fact[:80]}...\n" if len(fact) > 80 else f"  • {fact}\n")
                                if len(facts) > 3:
                                    f.write(f"  ... and {len(facts) - 3} more\n")
                                f.write("\n")
                    
                    print(f"✅ Summary saved: {summary_file}")
                    
                    print(f"\n📁 Test results saved to: {test_output_dir.absolute()}")
                    print(f"   Files created:")
                    print(f"     • {json_file.name} (human-readable JSON)")
                    print(f"     • {pickle_file.name} (Python objects)")
                    print(f"     • {summary_file.name} (text summary)")
                    
                except Exception as e:
                    print(f"⚠️  Failed to save test results: {e}")
                    import traceback
                    traceback.print_exc()
            
            # The cost tracker context manager will print final summary here
            
        print(f"\n🎯 Test completed successfully!")
        print(f"   • Processed {len(test_episodes)} episodes")
        print(f"   • Extracted {len(all_entities)} entities and {len(all_relationships)} relationships")
        print(f"   • Cost tracking enabled with detailed analysis")
        print(f"   • All API calls tracked and analyzed")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you've installed the required dependencies:")
        print("pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Error during knowledge extraction test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Knowledge RAG - Knowledge Extraction Test with Cost Tracking")
    print("=" * 70)
    
    success = test_knowledge_extraction()
    
    print("=" * 70)
    if success:
        print("🎉 Knowledge extraction test completed successfully!")
        print("\nThis demonstrates:")
        print("  • Complete knowledge extraction pipeline")
        print("  • Entity and relationship extraction")
        print("  • Deduplication and temporal processing")
        print("  • Comprehensive cost tracking and analysis")
        print("  • Real-time cost monitoring and limits")
        print("  • Detailed API usage statistics")
        print("  • Cost optimization insights")
    else:
        print("❌ Knowledge extraction test failed.")
        print("Check the error messages above for troubleshooting.")
    
    sys.exit(0 if success else 1) 