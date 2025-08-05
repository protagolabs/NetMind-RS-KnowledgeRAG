#!/usr/bin/env python3
"""
Neo4j Knowledge Graph Visualization

Provides Cypher queries and instructions for visualizing the stored knowledge graph
in Neo4j Browser. Also includes a programmatic way to test the queries.
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# Neo4j Cypher Queries for Visualization
VISUALIZATION_QUERIES = {
    "overview": {
        "title": "📊 Knowledge Graph Overview",
        "description": "Get an overview of all nodes and relationships",
        "query": """
        // Overview of the entire knowledge graph
        MATCH (n)
        RETURN 
            labels(n)[0] as NodeType, 
            count(n) as Count
        ORDER BY Count DESC
        """,
        "browser_query": "MATCH (n) RETURN n LIMIT 50"
    },
    
    "document_structure": {
        "title": "📄 Document Structure",
        "description": "Show document with its episodes",
        "query": """
        // Document with its episodes
        MATCH (d:Document)-[:CONTAINS]->(e:Episode)
        RETURN d, e
        ORDER BY e.sequence_number
        """,
        "browser_query": "MATCH (d:Document)-[:CONTAINS]->(e:Episode) RETURN d, e ORDER BY e.sequence_number"
    },
    
    "entities_overview": {
        "title": "🏷️  Entities by Type",
        "description": "Show all entities grouped by type",
        "query": """
        // Entities grouped by type
        MATCH (e:Entity)
        RETURN 
            e.entity_type as EntityType,
            collect(e.name)[0..5] as SampleNames,
            count(e) as Count
        ORDER BY Count DESC
        """,
        "browser_query": "MATCH (e:Entity) RETURN e"
    },
    
    "entity_relationships": {
        "title": "🔗 Entity Relationships",
        "description": "Show entities and their relationships",
        "query": """
        // Entities and their relationships
        MATCH (e1:Entity)-[r:RELATES_TO]->(e2:Entity)
        RETURN e1, r, e2
        """,
        "browser_query": "MATCH (e1:Entity)-[r:RELATES_TO]->(e2:Entity) RETURN e1, r, e2"
    },
    
    "technology_entities": {
        "title": "💻 Technology Entities",
        "description": "Focus on technology-related entities",
        "query": """
        // Technology entities and their connections
        MATCH (e:Entity)
        WHERE e.entity_type = 'TECHNOLOGY'
        OPTIONAL MATCH (e)-[r:RELATES_TO]-(connected:Entity)
        RETURN e, r, connected
        """,
        "browser_query": "MATCH (e:Entity) WHERE e.entity_type = 'TECHNOLOGY' OPTIONAL MATCH (e)-[r:RELATES_TO]-(connected) RETURN e, r, connected"
    },
    
    "entity_episodes": {
        "title": "📝 Entity-Episode Connections",
        "description": "Show which entities are mentioned in which episodes",
        "query": """
        // Entities mentioned in episodes
        MATCH (e:Entity)-[:MENTIONED_IN]->(ep:Episode)
        MATCH (d:Document)-[:CONTAINS]->(ep)
        RETURN e, ep, d
        LIMIT 20
        """,
        "browser_query": "MATCH (e:Entity)-[:MENTIONED_IN]->(ep:Episode)<-[:CONTAINS]-(d:Document) RETURN e, ep, d LIMIT 20"
    },
    
    "full_graph": {
        "title": "🌐 Complete Knowledge Graph",
        "description": "Show the complete knowledge graph structure",
        "query": """
        // Complete knowledge graph
        MATCH (n)
        OPTIONAL MATCH (n)-[r]-(m)
        RETURN n, r, m
        LIMIT 100
        """,
        "browser_query": "MATCH (n) OPTIONAL MATCH (n)-[r]-(m) RETURN n, r, m LIMIT 100"
    },
    
    "high_confidence_entities": {
        "title": "⭐ High Confidence Entities",
        "description": "Show entities with high confidence scores",
        "query": """
        // High confidence entities
        MATCH (e:Entity)
        WHERE e.confidence >= 0.9
        OPTIONAL MATCH (e)-[r:RELATES_TO]-(connected:Entity)
        RETURN e, r, connected
        ORDER BY e.confidence DESC
        """,
        "browser_query": "MATCH (e:Entity) WHERE e.confidence >= 0.9 OPTIONAL MATCH (e)-[r:RELATES_TO]-(connected) RETURN e, r, connected ORDER BY e.confidence DESC"
    },
    
    "entity_details": {
        "title": "🔍 Entity Details",
        "description": "Detailed view of a specific entity (replace 'RAG' with entity name)",
        "query": """
        // Detailed view of a specific entity
        MATCH (e:Entity)
        WHERE e.name CONTAINS 'RAG'
        OPTIONAL MATCH (e)-[:MENTIONED_IN]->(ep:Episode)
        OPTIONAL MATCH (e)-[r:RELATES_TO]-(related:Entity)
        RETURN e, ep, r, related
        """,
        "browser_query": "MATCH (e:Entity) WHERE e.name CONTAINS 'RAG' OPTIONAL MATCH (e)-[:MENTIONED_IN]->(ep:Episode) OPTIONAL MATCH (e)-[r:RELATES_TO]-(related) RETURN e, ep, r, related"
    },
    
    "graph_statistics": {
        "title": "📈 Graph Statistics",
        "description": "Get detailed statistics about the knowledge graph",
        "query": """
        // Graph statistics
        MATCH (d:Document) WITH count(d) as documents
        MATCH (e:Episode) WITH documents, count(e) as episodes
        MATCH (n:Entity) WITH documents, episodes, count(n) as entities
        MATCH ()-[r:RELATES_TO]->() WITH documents, episodes, entities, count(r) as relationships
        MATCH ()-[m:MENTIONED_IN]->() WITH documents, episodes, entities, relationships, count(m) as mentions
        RETURN {
            documents: documents,
            episodes: episodes,
            entities: entities,
            relationships: relationships,
            mentions: mentions
        } as GraphStats
        """,
        "browser_query": None  # This is a data query, not for visualization
    }
}

# Neo4j Browser Styling
NEO4J_STYLE = """
// Neo4j Browser Styling (paste in Browser settings)
{
  "node": {
    "diameter": "50px",
    "color": "#A5ABB6",
    "border-color": "#9AA1AC",
    "border-width": "2px",
    "text-color-internal": "#FFFFFF",
    "font-size": "10px"
  },
  "relationship": {
    "color": "#A5ABB6",
    "shaft-width": "1px",
    "font-size": "8px",
    "padding": "3px",
    "text-color-external": "#000000",
    "text-color-internal": "#FFFFFF"
  },
  "node.Document": {
    "color": "#68BDF6",
    "border-color": "#5CA8DB",
    "text-color-internal": "#FFFFFF"
  },
  "node.Entity": {
    "color": "#FF756E",
    "border-color": "#E06760",
    "text-color-internal": "#FFFFFF"
  },
  "node.Episode": {
    "color": "#6DCE9E",
    "border-color": "#60B08C",
    "text-color-internal": "#FFFFFF"
  },
  "relationship.CONTAINS": {
    "color": "#68BDF6",
    "shaft-width": "3px"
  },
  "relationship.RELATES_TO": {
    "color": "#FF756E",
    "shaft-width": "2px"
  },
  "relationship.MENTIONED_IN": {
    "color": "#6DCE9E",
    "shaft-width": "1px"
  }
}
"""


def print_visualization_guide():
    """Print a comprehensive guide for Neo4j visualization"""
    
    print("🎨 Neo4j Knowledge Graph Visualization Guide")
    print("=" * 50)
    
    print("\n📌 Prerequisites:")
    print("1. Neo4j is running at http://localhost:7474")
    print("2. Knowledge has been stored using test_store_from_json.py")
    print("3. Login credentials: neo4j/password")
    
    print("\n🌐 Access Neo4j Browser:")
    print("• Open: http://localhost:7474")
    print("• Username: neo4j")
    print("• Password: password")
    
    print("\n🎭 Visualization Queries:")
    print("Copy and paste these queries in Neo4j Browser")
    print("-" * 50)
    
    for key, query_info in VISUALIZATION_QUERIES.items():
        print(f"\n{query_info['title']}")
        print(f"Description: {query_info['description']}")
        
        if query_info.get('browser_query'):
            print("📋 Browser Query (copy this):")
            print("```")
            print(query_info['browser_query'])
            print("```")
        else:
            print("📋 Query:")
            print("```")
            print(query_info['query'].strip())
            print("```")
        
        print("-" * 40)
    
    print(f"\n🎨 Optional: Custom Styling")
    print("To improve visualization, go to Browser Settings > Graph Style Sheet")
    print("Replace the content with:")
    print("```")
    print(NEO4J_STYLE)
    print("```")


async def test_queries_programmatically():
    """Test the queries programmatically to ensure they work"""
    
    try:
        from neo4j import GraphDatabase
        
        print("\n🧪 Testing Queries Programmatically")
        print("-" * 40)
        
        # Connect to Neo4j
        driver = GraphDatabase.driver(
            os.getenv('NEO4J_URI', 'bolt://localhost:7687'), 
            auth=(os.getenv('NEO4J_USERNAME', 'neo4j'), os.getenv('NEO4J_PASSWORD', 'password'))
        )
        
        with driver.session() as session:
            for key, query_info in VISUALIZATION_QUERIES.items():
                try:
                    print(f"Testing: {query_info['title']}")
                    
                    # Use the appropriate query
                    query = query_info.get('browser_query') or query_info['query']
                    result = session.run(query)
                    
                    # Count results
                    records = list(result)
                    print(f"✅ Success: {len(records)} records returned")
                    
                    # Show sample data for some queries
                    if key in ['overview', 'entities_overview', 'graph_statistics'] and records:
                        print(f"   Sample: {dict(records[0]) if records else 'No data'}")
                    
                except Exception as e:
                    print(f"❌ Failed: {e}")
                
                print()
        
        driver.close()
        print("✅ All queries tested successfully!")
        
    except ImportError:
        print("⚠️  Neo4j driver not available for programmatic testing")
        print("Install with: pip install neo4j")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("Make sure Neo4j is running and data has been stored")


def generate_visualization_notebook():
    """Generate a Jupyter notebook with visualization examples"""
    
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Knowledge RAG - Neo4j Visualization\n",
                    "\n",
                    "This notebook contains visualization queries for the Knowledge RAG system stored in Neo4j.\n"
                ]
            }
        ]
    }
    
    for key, query_info in VISUALIZATION_QUERIES.items():
        # Add markdown cell with description
        notebook_content["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"## {query_info['title']}\n\n{query_info['description']}\n"
            ]
        })
        
        # Add code cell with query
        query = query_info.get('browser_query') or query_info['query']
        notebook_content["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f'# {query_info["title"]}\n',
                f'query = """\n{query}\n"""\n\n',
                'with driver.session() as session:\n',
                '    result = session.run(query)\n',
                '    records = list(result)\n',
                '    print(f"Found {len(records)} records")\n',
                '    for record in records[:5]:  # Show first 5\n',
                '        print(record)'
            ]
        })
    
    # Save notebook
    import json
    with open('neo4j_visualization.ipynb', 'w') as f:
        json.dump(notebook_content, f, indent=2)
    
    print("📓 Generated Jupyter notebook: neo4j_visualization.ipynb")


def main():
    """Main function"""
    
    print("🎨 Knowledge RAG - Neo4j Visualization")
    print("=" * 45)
    
    # Print the comprehensive guide
    print_visualization_guide()
    
    # Test queries programmatically
    import asyncio
    asyncio.run(test_queries_programmatically())
    
    # Generate notebook
    try:
        generate_visualization_notebook()
    except Exception as e:
        print(f"⚠️  Could not generate notebook: {e}")
    
    print("\n🎯 Quick Start:")
    print("1. Open Neo4j Browser: http://localhost:7474")
    print("2. Login with neo4j/password")
    print("3. Try the 'Complete Knowledge Graph' query first:")
    print("   MATCH (n) OPTIONAL MATCH (n)-[r]-(m) RETURN n, r, m LIMIT 100")
    print("\n4. Then explore entity relationships:")
    print("   MATCH (e1:Entity)-[r:RELATES_TO]->(e2:Entity) RETURN e1, r, e2")


if __name__ == "__main__":
    main() 