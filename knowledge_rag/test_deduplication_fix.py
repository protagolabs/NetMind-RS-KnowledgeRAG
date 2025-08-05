#!/usr/bin/env python3
"""
Test script to verify entity deduplication fix
1. Clear Neo4j database
2. Process first paper with 5 episodes  
3. Verify entities are linked to multiple episodes
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

def clear_neo4j_database():
    """Clear all data from Neo4j database"""
    try:
        from neo4j import GraphDatabase
        neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        neo4j_username = os.getenv('NEO4J_USERNAME', 'neo4j')
        neo4j_password = os.getenv('NEO4J_PASSWORD', 'password')

        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))
        
        with driver.session() as session:
            # Delete all nodes and relationships
            session.run("MATCH (n) DETACH DELETE n")
            print("✅ Neo4j database cleared")
            
        driver.close()
        return True
        
    except Exception as e:
        print(f"❌ Failed to clear Neo4j database: {e}")
        return False

def check_entity_episode_connections():
    """Check how entities are connected to episodes after processing"""
    try:
        from neo4j import GraphDatabase
        neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        neo4j_username = os.getenv('NEO4J_USERNAME', 'neo4j')
        neo4j_password = os.getenv('NEO4J_PASSWORD', 'password')

        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))
        
        with driver.session() as session:
            print("\n🔍 Entity-Episode Connection Analysis:")
            print("=" * 50)
            
            # Check total counts
            result = session.run("MATCH (d:Document) RETURN count(d) as docs")
            docs = result.single()['docs']
            
            result = session.run("MATCH (ep:Episode) RETURN count(ep) as episodes")
            episodes = result.single()['episodes']
            
            result = session.run("MATCH (e:Entity) RETURN count(e) as entities")
            entities = result.single()['entities']
            
            print(f"📊 Database Summary:")
            print(f"   • Documents: {docs}")
            print(f"   • Episodes: {episodes}")
            print(f"   • Entities: {entities}")
            
            # Check entity distribution across episodes
            print(f"\n📈 Entity Distribution Across Episodes:")
            result = session.run("""
                MATCH (ep:Episode)<-[:MENTIONED_IN]-(e:Entity)
                RETURN ep.id as episode_id, count(e) as entity_count
                ORDER BY ep.id
            """)
            
            episode_entities = {}
            for record in result:
                episode_id = record['episode_id']
                entity_count = record['entity_count']
                episode_entities[episode_id] = entity_count
                print(f"   • Episode {episode_id[-8:]}: {entity_count} entities")
            
            # Check entities that appear in multiple episodes (this is the key test!)
            print(f"\n🔗 Entities Appearing in Multiple Episodes:")
            result = session.run("""
                MATCH (e:Entity)-[:MENTIONED_IN]->(ep:Episode)
                WITH e, collect(ep.id) as episodes, count(ep) as episode_count
                WHERE episode_count > 1
                RETURN e.name, e.entity_type, episodes, episode_count
                ORDER BY episode_count DESC, e.name
            """)
            
            multi_episode_entities = []
            for record in result:
                entity_name = record['e.name']
                entity_type = record['e.entity_type']
                episodes_list = record['episodes']
                episode_count = record['episode_count']
                
                multi_episode_entities.append({
                    'name': entity_name,
                    'type': entity_type,
                    'episodes': episodes_list,
                    'count': episode_count
                })
                
                episode_ids = [ep[-8:] for ep in episodes_list]  # Show last 8 chars
                print(f"   ✅ {entity_name} ({entity_type}): {episode_count} episodes {episode_ids}")
            
            if not multi_episode_entities:
                print("   ❌ No entities found in multiple episodes - deduplication may have failed!")
                return False
            
            # Success criteria
            success = True
            if len(multi_episode_entities) < 2:
                print("   ⚠️  Expected more entities to appear in multiple episodes")
                success = False
                
            # Check for "Transformer" specifically (should appear in multiple episodes)
            transformer_found = any(entity['name'].lower() == 'transformer' for entity in multi_episode_entities)
            if not transformer_found:
                print("   ⚠️  'Transformer' entity should appear in multiple episodes")
                success = False
            else:
                transformer_entity = next(entity for entity in multi_episode_entities if entity['name'].lower() == 'transformer')
                if transformer_entity['count'] >= 3:
                    print(f"   🎯 SUCCESS: 'Transformer' appears in {transformer_entity['count']} episodes!")
                else:
                    print(f"   ⚠️  'Transformer' only appears in {transformer_entity['count']} episodes (expected 3+)")
                    success = False
            
            print(f"\n🎯 Test Result: {'✅ PASSED' if success else '❌ FAILED'}")
            return success
            
        driver.close()
        
    except Exception as e:
        print(f"❌ Failed to check entity connections: {e}")
        return False

async def main():
    """Main test function"""
    print("🧪 Testing Entity Deduplication Fix")
    print("=" * 40)
    
    # Step 1: Clear database
    print("\n1️⃣ Clearing Neo4j database...")
    if not clear_neo4j_database():
        print("❌ Cannot proceed without clearing database")
        return False
    
    # Step 2: Run processing
    print("\n2️⃣ Processing academic paper...")
    print("Running: python process_academic_papers.py")
    
    # Import and run the processing
    import subprocess
    result = subprocess.run([
        sys.executable, "process_academic_papers.py"
    ], capture_output=True, text=True, cwd=Path(__file__).parent)
    
    if result.returncode != 0:
        print(f"❌ Processing failed:")
        print(result.stderr)
        return False
    
    print("✅ Processing completed")
    
    # Step 3: Check results
    print("\n3️⃣ Checking entity-episode connections...")
    success = check_entity_episode_connections()
    
    if success:
        print("\n🎉 DEDUPLICATION FIX VERIFIED!")
        print("Entities are now properly linked to multiple episodes.")
    else:
        print("\n❌ DEDUPLICATION FIX FAILED!")
        print("Entities are still not properly linked across episodes.")
    
    return success

if __name__ == "__main__":
    import asyncio
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 