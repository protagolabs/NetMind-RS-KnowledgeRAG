# PDF Processing Pipeline

Complete end-to-end pipeline for processing PDF documents through the Knowledge RAG system.

## 🎯 What It Does

The `test_pdf_pipeline.py` script processes a PDF file through the complete Knowledge RAG pipeline:

1. **📖 PDF Parsing**: Extracts text, structure, and metadata from PDF pages
2. **🧠 Knowledge Extraction**: Uses OpenAI to extract entities and relationships
3. **🗄️ Direct Storage**: Stores knowledge directly in Neo4j + ChromaDB (no JSON intermediate)
4. **💰 Cost Tracking**: Comprehensive cost monitoring with aggregated data
5. **🔍 Search Testing**: Validates search functionality on stored knowledge
6. **📊 Analytics**: Detailed processing and cost statistics

## 🚀 Quick Start

### Basic Usage
```bash
# Process the default PDF (Weak-to-Strong Generalization paper)
python test_pdf_pipeline.py
```

### Custom PDF
```bash
# Process any PDF file
python test_pdf_pipeline.py /path/to/your/paper.pdf
```

### With Limits
```bash
# Limit to 10 episodes and $2.00 budget
python test_pdf_pipeline.py /path/to/paper.pdf 10 2.0
```

## 📋 Prerequisites

### 1. Services Running
```bash
# Start Neo4j
docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest
```

### 2. Environment Variables
Create `.env` file with:
```bash
OPENAI_API_KEY=your_openai_api_key_here
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
```

### 3. Dependencies
```bash
pip install -r requirements.txt
```

## 💡 Features

### Smart PDF Processing
- ✅ **Page-by-page parsing** with PyMuPDF
- ✅ **Structure detection** (headings, paragraphs, tables)
- ✅ **Metadata extraction** (page numbers, layout info)
- ✅ **Text cleaning** and normalization

### Advanced Knowledge Extraction
- ✅ **Entity extraction** with type classification
- ✅ **Relationship extraction** with confidence scores
- ✅ **Deduplication** across document sections
- ✅ **Temporal processing** for fact validity
- ✅ **Provenance tracking** back to source pages

### Comprehensive Cost Control
- ✅ **Real-time cost monitoring** during processing
- ✅ **Budget limits** with automatic stopping
- ✅ **Token counting** with tiktoken accuracy
- ✅ **Aggregated cost tracking** (new feature!)
- ✅ **Cost optimization recommendations**

### Direct Storage Integration
- ✅ **Neo4j graph storage** for entities and relationships
- ✅ **ChromaDB vector storage** for semantic search
- ✅ **Full document traceability** to source pages
- ✅ **Bi-temporal tracking** for knowledge evolution

## 📊 Example Output

```
🚀 Knowledge RAG - Complete PDF Pipeline
============================================================
📄 Processing PDF: Weak-to-Strong Generalization...
📁 File size: 2.3 MB
💰 Cost limit: $5.00

📖 STEP 1: Parsing PDF
------------------------------
✅ PDF processed: 23 episodes created
📋 Episode Breakdown:
   • page_content: 23 episodes (avg: 1,847 chars)

🧠 STEP 3: Knowledge Extraction
----------------------------------------
--- Episode 1/23 ---
📄 Page: 1
📏 Length: 1,245 characters
📝 Preview: Weak-to-Strong Generalization: Eliciting Strong...
💸 Current cost: $0.0000 / $5.00 (0.0%)
✅ Success!
   📊 Entities extracted: 8
   🔗 Relationships extracted: 12
   💰 Running cost: $0.0234

🗄️  STEP 4: Storing in Databases
--------------------------------------
✅ Knowledge stored successfully:
   • documents_stored: 1
   • episodes_stored: 23
   • entities_stored: 156
   • relationships_stored: 203

🔍 STEP 5: Testing Search
----------------------------
🔎 Testing search with sample queries...
   'weak supervision': 15 results (23.4ms)
   'generalization': 23 results (18.7ms)

📋 STEP 6: Final Report
-------------------------
🎉 PDF Pipeline Complete!
📄 Document: Weak-to-Strong Generalization...
📑 Episodes processed: 23/23
🧠 Knowledge extracted:
   • Entities: 156
   • Relationships: 203
💰 Cost summary:
   • Total cost: $1.2567
   • Budget used: 25.1%
   • Remaining budget: $3.7433
```

## 🎛️ Configuration Options

### Command Line Arguments
```bash
python test_pdf_pipeline.py [pdf_path] [max_episodes] [cost_limit]
```

- `pdf_path`: Path to PDF file (default: the weak-to-strong paper)
- `max_episodes`: Maximum episodes to process (default: all)
- `cost_limit`: Budget limit in USD (default: $5.00)

### Environment Variables
- `OPENAI_API_KEY`: Required for knowledge extraction
- `NEO4J_URI`: Neo4j connection string
- `NEO4J_USERNAME`: Neo4j username
- `NEO4J_PASSWORD`: Neo4j password

## 🔧 Advanced Usage

### Processing Large Documents
```bash
# Process in chunks with higher budget
python test_pdf_pipeline.py large_paper.pdf 50 10.0
```

### Development/Testing
```bash
# Quick test with first 5 pages only
python test_pdf_pipeline.py test_paper.pdf 5 1.0
```

### Production Processing
```bash
# Full processing with generous budget
python test_pdf_pipeline.py production_doc.pdf 999 25.0
```

## 📈 What You Get

### Immediate Results
- **Neo4j Graph**: Structured knowledge ready for querying
- **ChromaDB Vectors**: Semantic search capabilities
- **Cost Report**: Detailed cost breakdown and recommendations
- **Search Validation**: Confirmed working search functionality

### Files Created
- `cost_data/sessions/session_pdf_pipeline_*.json` - Detailed cost tracking
- `cost_data/aggregated_costs.json` - Updated with session costs
- `pdf_pipeline_chroma_db/` - ChromaDB embeddings storage

### Next Steps After Processing
1. **Visualize**: Open http://localhost:7474 (Neo4j Browser)
2. **Search**: Run `python test_search_knowledge.py --interactive`
3. **Analyze Costs**: Run `python cost_analysis.py`

## 🚨 Troubleshooting

### PDF Not Found
```
❌ PDF file not found: /path/to/file.pdf
```
**Solution**: Check the file path and ensure the PDF exists

### Neo4j Connection Failed
```
❌ Neo4j connection failed: ServiceUnavailable
```
**Solution**: Start Neo4j with Docker command above

### OpenAI API Key Missing
```
❌ OPENAI_API_KEY not found in environment variables
```
**Solution**: Add your API key to the `.env` file

### Cost Limit Exceeded
```
💰 Cost limit exceeded: $5.23 >= $5.00
⏹️  Stopping extraction to stay within budget
```
**Solution**: Increase cost limit or process fewer episodes

## 🎉 Success Indicators

✅ **PDF Parsed**: Episodes created from PDF pages  
✅ **Knowledge Extracted**: Entities and relationships found  
✅ **Storage Complete**: Data stored in both databases  
✅ **Search Working**: Test queries return results  
✅ **Cost Tracked**: Session costs recorded and aggregated  

Your PDF is now fully integrated into the Knowledge RAG system! 🚀 