#!/usr/bin/env python3
"""
Knowledge Base Builder

Processes the Knowledge_RAG_Design_Document.md using improved markdown parsing,
runs the complete knowledge extraction pipeline with cost tracking, and stores 
results for future testing of storage and search systems.
"""

import sys
import os
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def build_knowledge_base():
    """Build complete knowledge base from markdown parsing and extraction with cost tracking"""
    
    print("🏗️  Knowledge Base Builder with Cost Tracking")
    print("=" * 70)
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OpenAI API key not found in environment variables")
        print("Please set OPENAI_API_KEY in your .env file to run this process")
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
            print(f"❌ Source file not found: {markdown_file}")
            return False
        
        print(f"📄 Processing source file: {markdown_file.name}")
        print(f"📁 File size: {markdown_file.stat().st_size:,} bytes")
        
        # Create output directory for results
        output_dir = Path("knowledge_base_output")
        output_dir.mkdir(exist_ok=True)
        
        # Initialize file manager
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "kb_builder.db"
            
            file_manager = FileManager(
                root_folder=str(markdown_file.parent),
                db_path=str(db_path)
            )
            
            print("✅ FileManager initialized")
            
            # Process the markdown file to get episodes
            print("🔄 Processing markdown file with smart chunking...")
            result = file_manager.process_file(markdown_file)
            
            if not result or result.status != FileStatus.COMPLETED:
                print("❌ Failed to process file")
                return False
            
            print(f"✅ File processed: {result.episodes_created} episodes created")
            
            # Get the parsing result to access episodes
            parser = file_manager.parser_factory.get_parser(result.metadata.file_format)
            parsing_result = parser.parse(markdown_file, result.metadata)
            
            episodes = parsing_result.episodes
            print(f"📄 Retrieved {len(episodes)} episodes for knowledge extraction")
            
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
            
            # Initialize cost tracker with descriptive session name
            session_name = f"kb_build_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            with CostTracker(session_name=session_name) as cost_tracker:
                print(f"💰 Cost tracker initialized: {session_name}")
                
                # Set cost limit warning
                cost_limit = 5.0  # $5.00 warning limit
                
                # Initialize knowledge extractor with cost tracking
                extractor = KnowledgeExtractor(
                    model_name="gpt-3.5-turbo",  # Use cheaper model for large extraction
                    temperature=0.1,
                    max_retries=3,
                    cost_tracker=cost_tracker
                )
                
                print("✅ KnowledgeExtractor initialized with cost tracking")
                
                # Process all episodes for knowledge extraction
                print(f"\n🧠 Starting knowledge extraction for {len(episodes)} episodes...")
                print("This may take several minutes due to API calls...")
                
                all_entities = []
                all_relationships = []
                extraction_results = []
                failed_episodes = []
                
                # Process episodes in batches to manage API costs and show progress
                batch_size = 5
                total_batches = (len(episodes) + batch_size - 1) // batch_size
                
                for batch_idx in range(total_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, len(episodes))
                    batch_episodes = episodes[start_idx:end_idx]
                    
                    print(f"\n--- Processing Batch {batch_idx + 1}/{total_batches} ---")
                    print(f"Episodes {start_idx + 1} to {end_idx}")
                    
                    # Check cost before proceeding with batch
                    cost_check = cost_tracker.check_cost_limit(cost_limit)
                    if cost_check['exceeded']:
                        print(f"⚠️  Cost limit of ${cost_limit:.2f} exceeded!")
                        print(f"   Current cost: ${cost_check['current_cost']:.4f}")
                        print("   Stopping processing to avoid excessive costs.")
                        break
                    elif cost_check['percentage_used'] > 50:  # 50% warning
                        print(f"⚠️  Cost warning: {cost_check['percentage_used']:.1f}% of limit used (${cost_check['current_cost']:.4f}/${cost_limit:.2f})")
                    
                    for i, episode in enumerate(batch_episodes):
                        episode_num = start_idx + i + 1
                        header_text = episode.metadata.get('header_text', 'No header')
                        
                        print(f"\n🔄 Episode {episode_num}/{len(episodes)}: {header_text}")
                        print(f"   Type: {episode.episode_type} | Length: {len(episode.content):,} chars")
                        
                        # Estimate cost before extraction
                        sample_messages = [
                            {"role": "system", "content": "You are an entity extraction AI."},
                            {"role": "user", "content": f"Extract entities from: {episode.content[:500]}..."}
                        ]
                        
                        estimate = cost_tracker.estimate_cost_for_messages(sample_messages, "gpt-3.5-turbo")
                        print(f"   💸 Estimated cost: ${estimate.get('estimated_cost', 0):.4f}")
                        
                        # Extract knowledge
                        extraction_result = extractor.extract_from_episode(
                            episode=episode,
                            document=document,
                            existing_entities=all_entities,
                            existing_relationships=all_relationships
                        )
                        
                        if extraction_result.success:
                            print(f"   ✅ Success: {extraction_result.entities_extracted} entities, "
                                  f"{extraction_result.relationships_extracted} relationships")
                            print(f"      Deduplicated: {extraction_result.entities_deduplicated} entities, "
                                  f"{extraction_result.relationships_deduplicated} relationships")
                            print(f"      Time: {extraction_result.processing_time:.2f}s")
                            
                            # Update knowledge base
                            all_entities = extraction_result.entities
                            all_relationships = extraction_result.relationships
                            extraction_results.append(extraction_result)
                            
                        else:
                            print(f"   ❌ Failed: {extraction_result.error_message}")
                            failed_episodes.append({
                                'episode_id': episode.id,
                                'header_text': header_text,
                                'error': extraction_result.error_message
                            })
                        
                        # Show running cost
                        current_cost = cost_tracker.get_session_cost()
                        print(f"   💰 Session cost: ${current_cost:.4f}")
                    
                    # Show progress
                    processed = min(end_idx, len(episodes))
                    print(f"\n📊 Progress: {processed}/{len(episodes)} episodes processed")
                    
                    # Show running totals and cost analysis
                    stats = extractor.get_stats()
                    session_stats = cost_tracker.get_session_stats()
                    
                    print(f"   Running totals: {len(all_entities)} entities, {len(all_relationships)} relationships")
                    print(f"   API calls: {stats['api_calls_made']}, Time: {stats['total_processing_time']:.1f}s")
                    print(f"   Cost analysis: ${session_stats['total_cost']:.4f} ({session_stats['total_tokens']:,} tokens)")
                    print(f"   Success rate: {session_stats['success_rate']:.1f}%, Avg per call: ${session_stats['avg_cost_per_call']:.4f}")
                
                # Final statistics
                print(f"\n🎉 Knowledge Extraction Complete!")
                print("=" * 60)
                
                final_stats = extractor.get_stats()
                session_stats = cost_tracker.get_session_stats()
                
                print(f"📈 Final Extraction Statistics:")
                print(f"   Episodes processed: {final_stats['episodes_processed']}")
                print(f"   Total entities: {len(all_entities)}")
                print(f"   Total relationships: {len(all_relationships)}")
                print(f"   Entities extracted: {final_stats['entities_extracted']}")
                print(f"   Relationships extracted: {final_stats['relationships_extracted']}")
                print(f"   Entities deduplicated: {final_stats['entities_deduplicated']}")
                print(f"   Relationships deduplicated: {final_stats['relationships_deduplicated']}")
                print(f"   Failed episodes: {len(failed_episodes)}")
                
                print(f"\n💰 Final Cost Analysis:")
                print(f"   Session duration: {session_stats['duration_seconds']:.1f} seconds")
                print(f"   Total cost: ${session_stats['total_cost']:.4f}")
                print(f"   Total tokens: {session_stats['total_tokens']:,}")
                print(f"   API calls made: {final_stats['api_calls_made']}")
                print(f"   Success rate: {session_stats['success_rate']:.1f}%")
                print(f"   Average cost per call: ${session_stats['avg_cost_per_call']:.4f}")
                print(f"   Cost per token: ${session_stats['total_cost'] / session_stats['total_tokens']:.6f}" if session_stats['total_tokens'] > 0 else "   Cost per token: $0.000000")
                print(f"   Total processing time: {final_stats['total_processing_time']:.2f}s")
                
                # Cost efficiency metrics
                if len(episodes) > 0:
                    cost_per_episode = session_stats['total_cost'] / final_stats['episodes_processed'] if final_stats['episodes_processed'] > 0 else 0
                    print(f"   Cost per episode: ${cost_per_episode:.4f}")
                
                if len(all_entities) > 0:
                    cost_per_entity = session_stats['total_cost'] / len(all_entities)
                    print(f"   Cost per entity: ${cost_per_entity:.4f}")
                
                if len(all_relationships) > 0:
                    cost_per_relationship = session_stats['total_cost'] / len(all_relationships)
                    print(f"   Cost per relationship: ${cost_per_relationship:.4f}")
                
                # Update document with final knowledge stats
                document.update_knowledge_stats(all_entities, all_relationships)
                
                # Save results to files
                print(f"\n💾 Saving knowledge base to {output_dir}/...")
                
                # Save as JSON (human readable)
                knowledge_base = {
                    'metadata': {
                        'source_file': str(markdown_file),
                        'created_at': datetime.now().isoformat(),
                        'total_episodes': len(episodes),
                        'processed_episodes': final_stats['episodes_processed'],
                        'failed_episodes': len(failed_episodes),
                        'extraction_statistics': final_stats,
                        'cost_statistics': session_stats
                    },
                    'document': {
                        'uuid': document.uuid,
                        'file_path': document.file_path,
                        'file_name': document.file_name,
                        'title': document.title,
                        'entity_count': document.entity_count,
                        'relationship_count': document.relationship_count,
                        'key_entities': document.key_entities,
                        'key_relationships': document.key_relationships
                    },
                    'entities': [
                        {
                            'uuid': entity.uuid,
                            'name': entity.name,
                            'entity_type': entity.entity_type.value,
                            'confidence': entity.confidence,
                            'aliases': entity.aliases,
                            'summary': entity.summary,
                            'attributes': entity.attributes,
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
                    'episodes': [
                        {
                            'id': episode.id,
                            'content': episode.content,
                            'episode_type': episode.episode_type,
                            'sequence_number': episode.sequence_number,
                            'timestamp': episode.timestamp.isoformat(),
                            'metadata': episode.metadata
                        }
                        for episode in episodes
                    ],
                    'failed_episodes': failed_episodes
                }
                
                # Save JSON file
                json_file = output_dir / "knowledge_base.json"
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(knowledge_base, f, indent=2, ensure_ascii=False)
                
                print(f"✅ JSON saved: {json_file}")
                
                # Save Python objects (for easy loading)
                pickle_file = output_dir / "knowledge_base.pkl"
                pickle_data = {
                    'document': document,
                    'entities': all_entities,
                    'relationships': all_relationships,
                    'episodes': episodes,
                    'extraction_results': extraction_results,
                    'extraction_statistics': final_stats,
                    'cost_statistics': session_stats
                }
                
                with open(pickle_file, 'wb') as f:
                    pickle.dump(pickle_data, f)
                
                print(f"✅ Pickle saved: {pickle_file}")
                
                # Save entity and relationship summaries
                save_summaries(all_entities, all_relationships, output_dir)
                
                # Create index files for easy access
                create_index_files(all_entities, all_relationships, episodes, output_dir)
                
                # Export detailed cost report
                try:
                    cost_report_path = cost_tracker.export_session_report()
                    print(f"✅ Cost report saved: {cost_report_path}")
                except Exception as e:
                    print(f"⚠️  Failed to export cost report: {e}")
                
                print(f"\n🎯 Knowledge Base Ready!")
                print(f"   Location: {output_dir.absolute()}")
                print(f"   Files created:")
                print(f"     • knowledge_base.json (complete data)")
                print(f"     • knowledge_base.pkl (Python objects)")
                print(f"     • entity_summary.json (entity overview)")
                print(f"     • relationship_summary.json (relationship overview)")
                print(f"     • entity_index.json (entity lookup)")
                print(f"     • relationship_index.json (relationship lookup)")
                print(f"     • episode_index.json (episode lookup)")
                print(f"     • Cost tracking data in cost_data/ directory")
                
                # Final cost summary will be printed by context manager
                
                return True
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you've installed the required dependencies:")
        print("pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Error during knowledge base building: {e}")
        import traceback
        traceback.print_exc()
        return False


def save_summaries(entities: List, relationships: List, output_dir: Path):
    """Save entity and relationship summaries"""
    from collections import defaultdict
    
    # Entity summary by type
    entities_by_type = defaultdict(list)
    for entity in entities:
        entities_by_type[entity.entity_type.value].append({
            'name': entity.name,
            'confidence': entity.confidence,
            'aliases': entity.aliases,
            'summary': entity.summary
        })
    
    entity_summary = {
        'total_entities': len(entities),
        'entity_types': {
            entity_type: {
                'count': len(entities),
                'entities': entities[:10]  # Top 10 by confidence
            }
            for entity_type, entities in entities_by_type.items()
        }
    }
    
    with open(output_dir / "entity_summary.json", 'w') as f:
        json.dump(entity_summary, f, indent=2)
    
    # Relationship summary by type
    relationships_by_type = defaultdict(list)
    for relationship in relationships:
        relationships_by_type[relationship.relationship_type.value].append({
            'fact': relationship.fact,
            'confidence': relationship.confidence,
            'source_entity_id': relationship.source_entity_id,
            'target_entity_id': relationship.target_entity_id
        })
    
    relationship_summary = {
        'total_relationships': len(relationships),
        'relationship_types': {
            rel_type: {
                'count': len(rels),
                'relationships': rels[:10]  # Top 10 by confidence
            }
            for rel_type, rels in relationships_by_type.items()
        }
    }
    
    with open(output_dir / "relationship_summary.json", 'w') as f:
        json.dump(relationship_summary, f, indent=2)


def create_index_files(entities: List, relationships: List, episodes: List, output_dir: Path):
    """Create index files for quick lookup"""
    
    # Entity index
    entity_index = {
        entity.uuid: {
            'name': entity.name,
            'type': entity.entity_type.value,
            'confidence': entity.confidence
        }
        for entity in entities
    }
    
    with open(output_dir / "entity_index.json", 'w') as f:
        json.dump(entity_index, f, indent=2)
    
    # Relationship index
    relationship_index = {
        relationship.uuid: {
            'fact': relationship.fact,
            'type': relationship.relationship_type.value,
            'source': relationship.source_entity_id,
            'target': relationship.target_entity_id,
            'confidence': relationship.confidence
        }
        for relationship in relationships
    }
    
    with open(output_dir / "relationship_index.json", 'w') as f:
        json.dump(relationship_index, f, indent=2)
    
    # Episode index
    episode_index = {
        episode.id: {
            'type': episode.episode_type,
            'sequence': episode.sequence_number,
            'header_text': episode.metadata.get('header_text', ''),
            'length': len(episode.content)
        }
        for episode in episodes
    }
    
    with open(output_dir / "episode_index.json", 'w') as f:
        json.dump(episode_index, f, indent=2)


def load_knowledge_base(output_dir: Path = None) -> Dict[str, Any]:
    """
    Utility function to load the knowledge base for future use
    
    Returns:
        Dictionary containing document, entities, relationships, episodes, etc.
    """
    if output_dir is None:
        output_dir = Path("knowledge_base_output")
    
    pickle_file = output_dir / "knowledge_base.pkl"
    if not pickle_file.exists():
        raise FileNotFoundError(f"Knowledge base not found at {pickle_file}")
    
    with open(pickle_file, 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    print("Knowledge RAG - Knowledge Base Builder with Cost Tracking")
    print("=" * 70)
    
    success = build_knowledge_base()
    
    print("=" * 70)
    if success:
        print("🎉 Knowledge base built successfully with comprehensive cost tracking!")
        print("\nYou now have a complete knowledge base extracted from the")
        print("Knowledge RAG design document with detailed cost analysis.")
        print("\nUse this data to test:")
        print("  • Graph database storage (Neo4j)")
        print("  • Vector embeddings and search")
        print("  • Retrieval algorithms")
        print("  • Community detection")
        print("\nCost tracking provides:")
        print("  • Detailed API usage statistics")
        print("  • Cost optimization insights")
        print("  • Budget monitoring capabilities")
        print("  • Model efficiency comparisons")
        print("\nTo load the knowledge base in your code:")
        print("  from build_knowledge_base import load_knowledge_base")
        print("  kb = load_knowledge_base()")
        print("\nTo analyze costs:")
        print("  python cost_analysis.py")
    else:
        print("❌ Knowledge base building failed.")
        print("Check the error messages above for troubleshooting.")
    
    sys.exit(0 if success else 1) 