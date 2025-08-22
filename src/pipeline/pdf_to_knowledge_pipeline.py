"""
Complete pipeline for PDF to Knowledge Graph processing.

This pipeline:
1. Parses PDFs using enhanced document parser
2. Extracts entities and relationships using LLM
3. Stores in Neo4j and vector databases
4. Tracks API costs for each run
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import time

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Import our modules
from parsing.enhanced_document_parser import EnhancedDocumentParser
from extraction.llm_schema_extractor import LLMSchemaExtractor
from extraction.document_schemas import DocumentType
from storage import PaperStorageManager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PipelineConfig(BaseModel):
    """Configuration for the pipeline."""
    
    # Parsing config
    parser_type: str = Field(default="enhanced_marker", description="Parser to use")
    chunk_size: int = Field(default=500, description="Target chunk size in words")
    chunk_overlap: int = Field(default=50, description="Chunk overlap in words")
    parse_output_dir: str = Field(default="./parsed_docs", description="Output directory for parsed documents")
    use_section_chunking: bool = Field(default=False, description="Use section-based chunking")
    section_chunk_size: int = Field(default=2000, description="Words per chunk within sections")
    section_chunk_overlap: int = Field(default=200, description="Word overlap within sections")
    
    # Extraction config
    llm_model: str = Field(default="gpt-3.5-turbo", description="LLM model to use")
    llm_max_tokens: int = Field(default=2000, description="Max tokens for LLM")
    temperature: float = Field(default=0.1, description="LLM temperature")
    max_entities_per_chunk: int = Field(default=20, description="Max entities per chunk")
    max_relationships_per_chunk: int = Field(default=15, description="Max relationships per chunk")
    
    # Storage config
    neo4j_uri: str = Field(default="bolt://localhost:7687", description="Neo4j URI")
    neo4j_username: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str = Field(default="gfll9999", description="Neo4j password")
    neo4j_database: str = Field(default="neo4j", description="Neo4j database")
    chroma_persist_dir: str = Field(default="./chroma_db_papers", description="ChromaDB directory")
    embedding_model: str = Field(default="text-embedding-3-small", description="Embedding model")
    
    # Cost tracking
    track_costs: bool = Field(default=True, description="Track API costs")
    cost_file: str = Field(default="api_costs.json", description="Cost tracking file")


class CostTracker(BaseModel):
    """Track API costs for each run."""
    
    run_id: str = Field(description="Unique run ID")
    timestamp: str = Field(description="Run timestamp")
    pdf_file: str = Field(description="PDF file processed")
    
    # Token counts
    prompt_tokens: int = Field(default=0, description="Total prompt tokens")
    completion_tokens: int = Field(default=0, description="Total completion tokens")
    total_tokens: int = Field(default=0, description="Total tokens")
    
    # API calls
    extraction_calls: int = Field(default=0, description="Number of extraction API calls")
    embedding_calls: int = Field(default=0, description="Number of embedding API calls")
    
    # Costs (in USD)
    extraction_cost: float = Field(default=0.0, description="Cost for extraction")
    embedding_cost: float = Field(default=0.0, description="Cost for embeddings")
    total_cost: float = Field(default=0.0, description="Total cost")
    
    # Processing stats
    chunks_processed: int = Field(default=0, description="Number of chunks processed")
    entities_extracted: int = Field(default=0, description="Number of entities extracted")
    relationships_extracted: int = Field(default=0, description="Number of relationships extracted")
    processing_time: float = Field(default=0.0, description="Total processing time in seconds")


class PDFToKnowledgePipeline:
    """
    Complete pipeline for processing PDFs into knowledge graphs.
    """
    
    # Model pricing (as of 2024)
    MODEL_COSTS = {
        "gpt-3.5-turbo": {
            "prompt": 0.0005 / 1000,  # $0.0005 per 1K tokens
            "completion": 0.0015 / 1000  # $0.0015 per 1K tokens
        },
        "gpt-4": {
            "prompt": 0.03 / 1000,  # $0.03 per 1K tokens
            "completion": 0.06 / 1000  # $0.06 per 1K tokens
        },
        "text-embedding-3-small": {
            "embedding": 0.00002 / 1000  # $0.00002 per 1K tokens
        },
        "text-embedding-ada-002": {
            "embedding": 0.0001 / 1000  # $0.0001 per 1K tokens
        }
    }
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        self.cost_tracker: Optional[CostTracker] = None
        
        # Initialize components
        self.parser = EnhancedDocumentParser(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            use_section_chunking=config.use_section_chunking,
            section_chunk_size=config.section_chunk_size,
            section_chunk_overlap=config.section_chunk_overlap
        )
        self.extractor = None  # Will be initialized when needed
        self.storage_manager = None  # Will be initialized when needed
        
        logger.info("PDFToKnowledgePipeline initialized")
    
    async def initialize_storage(self):
        """Initialize storage manager."""
        if not self.storage_manager:
            self.storage_manager = PaperStorageManager(
                neo4j_uri=self.config.neo4j_uri,
                neo4j_username=self.config.neo4j_username,
                neo4j_password=self.config.neo4j_password,
                neo4j_database=self.config.neo4j_database,
                chroma_persist_dir=self.config.chroma_persist_dir,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                embedding_model=self.config.embedding_model
            )
            await self.storage_manager.initialize()
            logger.info("Storage manager initialized")
    
    def initialize_extractor(self):
        """Initialize LLM extractor."""
        if not self.extractor:
            self.extractor = LLMSchemaExtractor(
                api_key=os.getenv("OPENAI_API_KEY"),
                model=self.config.llm_model,
                temperature=self.config.temperature,
                max_tokens=self.config.llm_max_tokens
            )
            logger.info(f"LLM extractor initialized with model: {self.config.llm_model}")
    
    def start_cost_tracking(self, pdf_file: str) -> CostTracker:
        """
        Start cost tracking for a run.
        
        Args:
            pdf_file: PDF file being processed
            
        Returns:
            CostTracker instance
        """
        self.cost_tracker = CostTracker(
            run_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            timestamp=datetime.now().isoformat(),
            pdf_file=pdf_file
        )
        return self.cost_tracker
    
    def update_extraction_costs(self, usage: Dict[str, Any]):
        """
        Update cost tracker with extraction usage.
        
        Args:
            usage: OpenAI usage dictionary
        """
        if not self.cost_tracker:
            return
        
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        
        self.cost_tracker.prompt_tokens += prompt_tokens
        self.cost_tracker.completion_tokens += completion_tokens
        self.cost_tracker.total_tokens += usage.get("total_tokens", 0)
        self.cost_tracker.extraction_calls += 1
        
        # Calculate cost
        model_costs = self.MODEL_COSTS.get(self.config.llm_model, self.MODEL_COSTS["gpt-3.5-turbo"])
        cost = (prompt_tokens * model_costs["prompt"] + 
                completion_tokens * model_costs["completion"])
        self.cost_tracker.extraction_cost += cost
        self.cost_tracker.total_cost += cost
    
    def update_embedding_costs(self, token_count: int):
        """
        Update cost tracker with embedding usage.
        
        Args:
            token_count: Number of tokens embedded
        """
        if not self.cost_tracker:
            return
        
        self.cost_tracker.embedding_calls += 1
        
        # Calculate cost
        model_costs = self.MODEL_COSTS.get(self.config.embedding_model, self.MODEL_COSTS["text-embedding-3-small"])
        cost = token_count * model_costs["embedding"]
        self.cost_tracker.embedding_cost += cost
        self.cost_tracker.total_cost += cost
    
    def save_cost_tracking(self):
        """Save cost tracking to file."""
        if not self.cost_tracker or not self.config.track_costs:
            return
        
        cost_file = Path(self.config.cost_file)
        
        # Load existing costs if file exists
        if cost_file.exists():
            with open(cost_file, 'r') as f:
                costs = json.load(f)
        else:
            costs = []
        
        # Add current run
        costs.append(self.cost_tracker.dict())
        
        # Save updated costs
        with open(cost_file, 'w') as f:
            json.dump(costs, f, indent=2)
        
        logger.info(f"Cost tracking saved to {cost_file}")
    
    async def process_pdf(
        self,
        pdf_path: str,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a PDF through the complete pipeline.
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Optional output directory for intermediate files
            
        Returns:
            Processing results
        """
        start_time = time.time()
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        # Start cost tracking
        if self.config.track_costs:
            self.start_cost_tracking(str(pdf_path))
        
        # Set output directory
        if output_dir:
            output_dir = Path(output_dir)
        else:
            output_dir = pdf_path.parent
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            "pdf_file": str(pdf_path),
            "status": "started",
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Step 1: Parse PDF
            logger.info(f"Parsing PDF: {pdf_path}")
            parsed_doc = self.parser.parse_pdf(str(pdf_path))
            
            # Save parsed document
            parsed_output = output_dir / f"{pdf_path.stem}.enhanced.json"
            
            # Convert Pydantic model to JSON and save
            with open(parsed_output, 'w', encoding='utf-8') as f:
                json.dump(parsed_doc.dict(), f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Parsed document saved to: {parsed_output}")
            
            results["parsing"] = {
                "status": "success",
                "output_file": str(parsed_output),
                "chunks": len(parsed_doc.text_chunks),
                "tables": len(parsed_doc.table_chunks),
                "images": len(parsed_doc.image_chunks)
            }
            
            # Step 2: Extract entities and relationships
            logger.info("Extracting entities and relationships...")
            self.initialize_extractor()
            
            extraction_result = self.extractor.extract_from_document(
                parsed_doc,
                override_type=DocumentType.ACADEMIC_PAPER,
            )
            
            logger.info(f"llm extractor done")
            
            # Update extraction costs
            # if self.cost_tracker and hasattr(self.extractor, 'last_usage'):
            #     for usage in self.extractor.last_usage:
            #         self.update_extraction_costs(usage)
            
            if self.cost_tracker:
                self.cost_tracker.chunks_processed = len(parsed_doc.text_chunks)
                self.cost_tracker.entities_extracted = extraction_result["entity_count"]
                self.cost_tracker.relationships_extracted = extraction_result["relationship_count"]
            
            results["extraction"] = {
                "status": "success",
                "output_file": extraction_result["output_file"],
                "entities": extraction_result["entity_count"],
                "relationships": extraction_result["relationship_count"],
                "document_type": extraction_result["document_type"]
            }
            
            logger.info(f"Extraction result done")
            
            # Step 3: Store in databases
            logger.info("Storing in knowledge graph and vector database...")
            await self.initialize_storage()
            
            storage_result = await self.storage_manager.store_paper(
                json_path=extraction_result["output_file"]
            )
            
            # Estimate embedding costs (approximate)
            if self.cost_tracker:
                # Rough estimate: 100 tokens per entity/chunk
                estimated_tokens = (storage_result["entities_count"] + 
                                  storage_result["chunks_count"]) * 100
                self.update_embedding_costs(estimated_tokens)
            
            results["storage"] = {
                "status": "success",
                "graph_stored": storage_result["graph_stored"],
                "embeddings_stored": storage_result["embeddings_stored"],
                "entities_count": storage_result["entities_count"],
                "relationships_count": storage_result["relationships_count"],
                "chunks_count": storage_result["chunks_count"]
            }
            
            # Calculate processing time
            processing_time = time.time() - start_time
            if self.cost_tracker:
                self.cost_tracker.processing_time = processing_time
            
            results["status"] = "completed"
            results["processing_time"] = processing_time
            
            # Save cost tracking
            if self.config.track_costs:
                self.save_cost_tracking()
                results["cost_tracking"] = {
                    "total_cost": self.cost_tracker.total_cost,
                    "extraction_cost": self.cost_tracker.extraction_cost,
                    "embedding_cost": self.cost_tracker.embedding_cost,
                    "total_tokens": self.cost_tracker.total_tokens
                }
            
            logger.info(f"Pipeline completed successfully in {processing_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            results["status"] = "failed"
            results["error"] = str(e)
            
            # Still save cost tracking on failure
            if self.config.track_costs and self.cost_tracker:
                self.cost_tracker.processing_time = time.time() - start_time
                self.save_cost_tracking()
        
        return results
    
    async def process_multiple_pdfs(
        self,
        pdf_paths: List[str],
        output_dir: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple PDFs.
        
        Args:
            pdf_paths: List of PDF paths
            output_dir: Optional output directory
            
        Returns:
            List of processing results
        """
        results = []
        
        for pdf_path in pdf_paths:
            logger.info(f"Processing {pdf_path}...")
            result = await self.process_pdf(pdf_path, output_dir)
            results.append(result)
        
        return results
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """
        Get cost summary from tracking file.
        
        Returns:
            Cost summary statistics
        """
        cost_file = Path(self.config.cost_file)
        
        if not cost_file.exists():
            return {"error": "No cost tracking file found"}
        
        with open(cost_file, 'r') as f:
            costs = json.load(f)
        
        if not costs:
            return {"error": "No cost data available"}
        
        total_cost = sum(c["total_cost"] for c in costs)
        total_tokens = sum(c["total_tokens"] for c in costs)
        total_time = sum(c["processing_time"] for c in costs)
        
        return {
            "total_runs": len(costs),
            "total_cost_usd": round(total_cost, 4),
            "total_tokens": total_tokens,
            "total_processing_time": round(total_time, 2),
            "average_cost_per_run": round(total_cost / len(costs), 4),
            "average_tokens_per_run": total_tokens // len(costs),
            "recent_runs": costs[-5:]  # Last 5 runs
        }


async def main():
    """Main function for testing the pipeline."""
    
    # Configure pipeline
    config = PipelineConfig(
        llm_model="gpt-3.5-turbo",
        temperature=0.1,
        llm_max_tokens=10000,
        track_costs=True,
        cost_file="api_costs.json"
    )
    
    # Create pipeline
    pipeline = PDFToKnowledgePipeline(config)
    
    # Test PDF
    test_pdf = "/home/administrator/projects/NetMind-RS-KnowledgeRAG/paper_sets/paper_set_1/docs/AutoPrompt Eliciting Knowledge from Language Models with Automatically Generated Prompts.pdf"
    
    if Path(test_pdf).exists():
        logger.info(f"Processing: {test_pdf}")
        
        # Process PDF
        results = await pipeline.process_pdf(test_pdf)
        
        # Print results
        print("\n" + "="*50)
        print("Pipeline Results")
        print("="*50)
        print(json.dumps(results, indent=2))
        
        # Print cost summary
        if config.track_costs:
            print("\n" + "="*50)
            print("Cost Summary")
            print("="*50)
            summary = pipeline.get_cost_summary()
            print(json.dumps(summary, indent=2))
    else:
        logger.error(f"Test PDF not found: {test_pdf}")
    
    # Close storage connections
    if pipeline.storage_manager:
        await pipeline.storage_manager.close()


if __name__ == "__main__":
    asyncio.run(main())