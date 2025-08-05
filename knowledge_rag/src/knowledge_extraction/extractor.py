"""
Knowledge Extraction Pipeline

Main orchestrator for extracting entities, relationships, and temporal information
from episodic content using LLMs. Includes deduplication and conflict resolution.
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from openai import OpenAI

from .models import Entity, Relationship, Document, ExtractionResult, EntityType, RelationshipType
from .prompts import (
    ENTITY_EXTRACTION_STRUCTURED_PROMPT,
    RELATIONSHIP_EXTRACTION_STRUCTURED_PROMPT,
    ENTITY_DEDUPLICATION_STRUCTURED_PROMPT,
    RELATIONSHIP_DEDUPLICATION_STRUCTURED_PROMPT,
    TEMPORAL_EXTRACTION_STRUCTURED_PROMPT,
    create_relationship_extraction_schema
)

# Import Episode from file_management
try:
    from ..file_management.models import Episode
except ImportError:
    # Fallback for testing
    from dataclasses import dataclass
    @dataclass
    class Episode:
        id: str
        content: str


class KnowledgeExtractor:
    """Main knowledge extraction orchestrator"""
    
    def __init__(
        self, 
        openai_api_key: Optional[str] = None,
        model_name: str = "gpt-4",
        max_retries: int = 3,
        temperature: float = 0.1,
        cost_tracker: Optional[Any] = None,
        use_structured_output: bool = True,
        reasoning_effort: str = "medium"
    ):
        self.logger = logging.getLogger(__name__)
        
        # OpenAI configuration
        self.api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        
        self.model_name = model_name
        self.max_retries = max_retries
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort  # For o-series models: low, medium, high
        
        # Structured output configuration
        self.use_structured_output = use_structured_output
        self.supports_structured_output = self._check_structured_output_support(model_name)
        
        if self.use_structured_output and self.supports_structured_output:
            self.logger.info(f"Structured output enabled for model: {model_name}")
        elif self.use_structured_output and not self.supports_structured_output:
            self.logger.warning(f"Model {model_name} doesn't support structured output, falling back to regular prompts")
            self.use_structured_output = False
        
        # Cost tracking
        self.cost_tracker = cost_tracker
        if self.cost_tracker:
            self.logger.info("Cost tracking enabled")
        
        # Supported entity and relationship types
        self.entity_types = [t.value for t in EntityType]
        self.relationship_types = [t.value for t in RelationshipType]
        
        # Statistics tracking
        self.stats = {
            'episodes_processed': 0,
            'entities_extracted': 0,
            'relationships_extracted': 0,
            'entities_deduplicated': 0,
            'relationships_deduplicated': 0,
            'api_calls_made': 0,
            'total_processing_time': 0.0,
            'errors': []
        }
        
        # Create debug directory for detailed logging
        self.debug_dir = Path("debug_extraction")
        self.debug_dir.mkdir(exist_ok=True)
        self.debug_session = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Enable detailed file logging
        debug_handler = logging.FileHandler(self.debug_dir / f"extraction_debug_{self.debug_session}.log")
        debug_handler.setLevel(logging.DEBUG)
        debug_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        debug_handler.setFormatter(debug_formatter)
        self.logger.addHandler(debug_handler)
        
        # Also enable console logging for INFO level messages
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        self.logger.setLevel(logging.INFO)  # Ensure logger level allows INFO messages
        
        self.logger.info(f"KnowledgeExtractor initialized with model: {model_name}")
        self.logger.info(f"Debug logging enabled - session: {self.debug_session}")
        self.logger.info(f"Debug directory: {self.debug_dir.absolute()}")
    
    def _check_structured_output_support(self, model_name: str) -> bool:
        """Check if the model supports structured outputs"""
        # Models that support structured outputs (as of 2024-2025)
        # Note: o-series models have limited structured output support
        structured_output_models = [
            # GPT-4o models (well-tested structured output support)
            'gpt-4o', 'gpt-4o-2024-05-13', 'gpt-4o-2024-08-06',
            'gpt-4o-mini', 'gpt-4o-mini-2024-07-18',
            # GPT-4 Turbo models
            'gpt-4-turbo', 'gpt-4-turbo-2024-04-09',
            # GPT-3.5 specific versions
            'gpt-3.5-turbo-0125'  # Only specific version, not generic gpt-3.5-turbo
            # Note: Removing o-series models for now due to compatibility issues
            # 'o4-mini', 'o4-mini-2025-04-16',
        ]
        
        # Exact match only - don't use partial matching for model names
        return model_name in structured_output_models
    
    def _call_openai_api(self, messages: List[Dict[str, str]], operation_type: str = "general", response_format: Optional[Dict] = None) -> str:
        """
        Make OpenAI API call with cost tracking and structured output support
        
        Args:
            messages: List of message dictionaries
            operation_type: Type of operation for context (e.g., "relationship_extraction")
            response_format: Optional response format schema for structured outputs
            
        Returns:
            Response text content
        """
        start_time = time.time()
        processing_time = 0.0  # Initialize processing_time
        
        # Create debug file names
        call_id = f"{operation_type}_{int(time.time())}_{hash(str(messages)) % 10000}"
        prompt_file = self.debug_dir / f"prompt_{call_id}.json"
        response_file = self.debug_dir / f"response_{call_id}.txt"
        api_params_file = self.debug_dir / f"api_params_{call_id}.json"
        
        try:
            # Prepare API call parameters
            api_params = {
                "model": self.model_name,
                "messages": messages
            }
            
            # Configure parameters based on model type
            if self.model_name.startswith('o'):  # o-series models (o1, o3, o4-mini, etc.)
                # o-series models don't support temperature, top_p, etc.
                # They use reasoning_effort instead for controlling behavior
                api_params["max_completion_tokens"] = 4000
                api_params["reasoning_effort"] = self.reasoning_effort  # Use the configured reasoning_effort
                # Always request JSON format for o-series models
                api_params["response_format"] = {"type": "json_object"}
            else:  # GPT models
                # Regular GPT models support standard sampling parameters
                api_params["temperature"] = self.temperature
                api_params["max_tokens"] = 4000
            
            # Add structured output if supported and requested
            # Note: o-series models may have different structured output behavior
            # Temporarily disable structured output for relationship extraction due to schema validation issues
            if (self.use_structured_output and 
                self.supports_structured_output and 
                response_format and 
                operation_type == "relationship_extraction" and
                not self.model_name.startswith('o') and
                False):  # Temporarily disabled
                api_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "relationship_extraction_response",
                        "schema": response_format,
                        "strict": True
                    }
                }
                self.logger.debug("Using structured output for relationship extraction")
            elif (self.use_structured_output and response_format and 
                  operation_type == "relationship_extraction" and
                  self.model_name.startswith('o')):
                # For o-series models, JSON format is already set above
                self.logger.debug("Using JSON object format for o-series model (already set)")
            elif (operation_type == "relationship_extraction"):
                # Use simple JSON object format for relationship extraction to avoid schema issues
                api_params["response_format"] = {"type": "json_object"}
                self.logger.debug("Using simple JSON object format for relationship extraction")
            
            # Save debug information to files
            debug_info = {
                "operation_type": operation_type,
                "model_name": self.model_name,
                "messages": messages,
                "api_params": api_params,
                "timestamp": datetime.now().isoformat(),
                "session": self.debug_session
            }
            
            with open(prompt_file, 'w', encoding='utf-8') as f:
                json.dump(debug_info, f, indent=2, ensure_ascii=False)
            
            with open(api_params_file, 'w', encoding='utf-8') as f:
                json.dump(api_params, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"🔍 DEBUG: Saved prompt to {prompt_file}")
            self.logger.info(f"🔍 DEBUG: API params - Model: {self.model_name}, Operation: {operation_type}")
            self.logger.info(f"🔍 DEBUG: Message count: {len(messages)}, Content length: {sum(len(str(m.get('content', ''))) for m in messages)}")
            
            # Log first part of messages for debugging
            for i, msg in enumerate(messages):
                content = str(msg.get('content', ''))
                self.logger.debug(f"🔍 DEBUG: Message {i} ({msg.get('role', 'unknown')}): {content[:200]}...")
            
            # Make the API call
            self.logger.info(f"🚀 Making API call to {self.model_name} for {operation_type}")
            response = self.client.chat.completions.create(**api_params)
            
            # Extract response content
            response_text = response.choices[0].message.content
            
            # Save response to file (even if empty)
            response_debug = {
                "response_text": response_text,
                "response_length": len(response_text) if response_text else 0,
                "response_type": type(response_text).__name__,
                "is_empty": not response_text or response_text.strip() == "",
                "api_call_success": True,
                "timestamp": datetime.now().isoformat(),
                "processing_time": time.time() - start_time
            }
            
            with open(response_file, 'w', encoding='utf-8') as f:
                json.dump(response_debug, f, indent=2, ensure_ascii=False)
                f.write("\n\n=== RAW RESPONSE ===\n")
                f.write(repr(response_text))
                f.write("\n\n=== FORMATTED RESPONSE ===\n")
                f.write(str(response_text) if response_text else "None")
            
            self.logger.info(f"💾 DEBUG: Saved response to {response_file}")
            
            if not response_text or response_text.strip() == "":
                self.logger.error(f"❌ EMPTY RESPONSE from {self.model_name} for {operation_type}")
                self.logger.error(f"❌ Response was: {repr(response_text)} (length: {len(response_text) if response_text else 0})")
                self.logger.error(f"❌ Response type: {type(response_text)}")
                raise Exception(f"Empty response from model")
            
            response_text = response_text.strip()
            
            # Log response for debugging (first 200 chars)
            self.logger.info(f"✅ API response preview ({len(response_text)} chars): {response_text[:200]}...")
            
            # Additional debug logging for o-series models
            if self.model_name.startswith('o'):
                self.logger.info(f"🤖 o-series model response length: {len(response_text)} chars")
                if len(response_text) < 50:
                    self.logger.warning(f"⚠️ Very short response from o-series model: {repr(response_text)}")
                
                # Try to detect if it's valid JSON
                try:
                    json.loads(response_text)
                    self.logger.info("✅ Response appears to be valid JSON")
                except json.JSONDecodeError as e:
                    self.logger.warning(f"⚠️ Response is not valid JSON: {e}")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Track cost if tracker is available
            if self.cost_tracker:
                # Track the API call with correct parameters
                self.cost_tracker.track_api_call(
                    messages=messages,
                    response_text=response_text,
                    model_name=self.model_name,
                    processing_time=processing_time,
                    success=True
                )
            
            # Update stats
            self.stats['api_calls_made'] += 1
            self.stats['total_processing_time'] += processing_time
            
            self.logger.info(f"✅ API call completed in {processing_time:.2f}s")
            return response_text
            
        except Exception as e:
            # Calculate processing time for failed calls
            processing_time = time.time() - start_time
            self.stats['total_processing_time'] += processing_time
            self.stats['errors'].append(str(e))
            
            # Save error information to response file
            error_debug = {
                "error": str(e),
                "error_type": type(e).__name__,
                "api_call_success": False,
                "timestamp": datetime.now().isoformat(),
                "processing_time": processing_time
            }
            
            with open(response_file, 'w', encoding='utf-8') as f:
                json.dump(error_debug, f, indent=2, ensure_ascii=False)
            
            # Track failed API call with correct parameters
            if self.cost_tracker:
                self.cost_tracker.track_api_call(
                    messages=messages,
                    response_text="",
                    model_name=self.model_name,
                    processing_time=processing_time,
                    success=False,
                    error_message=str(e)
                )
            
            self.logger.error(f"❌ OpenAI API call failed: {e}")
            self.logger.error(f"💾 Error details saved to {response_file}")
            raise
    
    def extract_from_episode(
        self,
        episode: Episode,
        document: Document,
        existing_entities: Optional[List[Entity]] = None,
        existing_relationships: Optional[List[Relationship]] = None
    ) -> ExtractionResult:
        """
        Extract knowledge from a single episode
        
        Args:
            episode: Episode to extract knowledge from
            document: Document containing the episode
            existing_entities: Previously extracted entities for deduplication
            existing_relationships: Previously extracted relationships for deduplication
            
        Returns:
            ExtractionResult with extracted and deduplicated knowledge
        """
        start_time = time.time()
        
        try:
            # Set cost tracking context if available
            if self.cost_tracker:
                self.cost_tracker.set_context(
                    episode_id=episode.id,
                    document_id=document.uuid,
                    operation="knowledge_extraction"
                )
            
            self.logger.info(f"Starting knowledge extraction for episode {episode.id}")
            
            # Step 1: Extract entities
            self.logger.debug("Step 1: Extracting entities")
            entities = self._extract_entities(episode)
            
            # Step 2: Extract relationships between entities
            self.logger.debug("Step 2: Extracting relationships")
            relationships = self._extract_relationships(episode, entities)
            
            # Step 3: Extract temporal information for relationships
            self.logger.debug("Step 3: Extracting temporal information")
            # Skip temporal extraction for o-series models as it's causing empty responses
            if not self.model_name.startswith('o'):
                relationships = self._extract_temporal_info(episode, relationships)
            else:
                self.logger.info(f"Skipping temporal extraction for o-series model: {self.model_name}")
                # For o-series models, just set current time as valid_at for relationships that don't have it
                for rel in relationships:
                    if rel.valid_at is None:
                        rel.valid_at = datetime.now()
            
            # Step 4: Deduplicate entities with existing ones
            entities_deduplicated = 0
            if existing_entities:
                self.logger.debug("Step 4: Deduplicating entities")
                entities, entities_deduplicated = self._deduplicate_entities(entities, existing_entities)
            
            # Step 5: Deduplicate relationships with existing ones
            relationships_deduplicated = 0
            if existing_relationships:
                self.logger.debug("Step 5: Deduplicating relationships")
                relationships, relationships_deduplicated = self._deduplicate_relationships(relationships, existing_relationships)
            
            # Step 6: Update entity and relationship metadata
            self.logger.debug("Step 6: Updating provenance")
            self._update_provenance(entities, relationships, episode.id, document.uuid)
            
            # Step 7: Create result and update stats
            processing_time = time.time() - start_time
            
            result = ExtractionResult(
                entities=entities,
                relationships=relationships,
                processing_time=processing_time,
                success=True,
                entities_extracted=len(entities),
                relationships_extracted=len(relationships),
                entities_deduplicated=entities_deduplicated,
                relationships_deduplicated=relationships_deduplicated,
                source_episode_id=episode.id,
                source_document_id=document.uuid
            )
            
            # Update internal statistics
            self._update_stats(result)
            
            self.logger.info(
                f"Knowledge extraction completed for episode {episode.id}: "
                f"{len(entities)} entities, {len(relationships)} relationships "
                f"(took {processing_time:.2f}s)"
            )
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Knowledge extraction failed for episode {episode.id}: {str(e)}"
            self.logger.error(error_msg)
            
            # Record error in stats
            self.stats['errors'].append({
                'episode_id': episode.id,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            
            return ExtractionResult(
                entities=[],
                relationships=[],
                processing_time=processing_time,
                success=False,
                error_message=error_msg,
                source_episode_id=episode.id,
                source_document_id=document.uuid
            )
    
    def _extract_entities(self, episode: Episode) -> List[Entity]:
        """Extract entities from episode content"""
        extraction_id = f"entity_{episode.id}_{int(time.time())}"
        extraction_file = self.debug_dir / f"entity_extraction_{extraction_id}.json"
        
        try:
            self.logger.info(f"🧠 Starting entity extraction for episode {episode.id}")
            
            # Set operation context for cost tracking
            if self.cost_tracker:
                self.cost_tracker.set_context(
                    episode_id=episode.id,
                    operation="entity_extraction"
                )
            
            # Prepare prompt
            context = {
                'episode_content': episode.content,
                'entity_types': self.entity_types
            }
            
            messages = ENTITY_EXTRACTION_STRUCTURED_PROMPT(context)
            
            # Save extraction attempt details
            extraction_debug = {
                "extraction_id": extraction_id,
                "episode_id": episode.id,
                "episode_content_length": len(episode.content),
                "episode_content_preview": episode.content[:500],
                "entity_types": self.entity_types,
                "model_name": self.model_name,
                "context": context,
                "messages": messages,
                "timestamp": datetime.now().isoformat(),
                "status": "attempting"
            }
            
            with open(extraction_file, 'w', encoding='utf-8') as f:
                json.dump(extraction_debug, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"💾 DEBUG: Saved entity extraction attempt to {extraction_file}")
            
            # Make API call with cost tracking
            response_text = self._call_openai_api(
                messages=messages,
                operation_type="entity_extraction"
            )
            
            self.logger.info(f"📨 Got response for entity extraction: {len(response_text)} chars")
            
            # Update extraction debug with response
            extraction_debug.update({
                "status": "response_received",
                "response_text": response_text,
                "response_length": len(response_text),
                "parsing_attempt": "starting"
            })
            
            # Parse JSON response
            entities_data = []
            try:
                # First try to parse the entire response
                response_data = json.loads(response_text)
                entities_data = response_data.get('entities', [])
                extraction_debug.update({
                    "parsing_attempt": "success",
                    "response_data": response_data,
                    "entities_found": len(entities_data)
                })
                self.logger.info(f"✅ Successfully parsed entity JSON: {len(entities_data)} entities found")
            except json.JSONDecodeError as e:
                self.logger.error(f"❌ Failed to parse entity extraction JSON: {e}")
                self.logger.error(f"❌ Response text: {response_text[:500]}...")
                
                extraction_debug.update({
                    "parsing_attempt": "failed",
                    "parse_error": str(e),
                    "trying_fallback": True
                })
                
                # Try to extract JSON from response if it's embedded in other text
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    try:
                        response_data = json.loads(json_match.group())
                        entities_data = response_data.get('entities', [])
                        extraction_debug.update({
                            "fallback_parsing": "success",
                            "fallback_response_data": response_data,
                            "entities_found": len(entities_data)
                        })
                        self.logger.info("✅ Successfully extracted JSON from embedded response")
                    except json.JSONDecodeError:
                        extraction_debug.update({"fallback_parsing": "failed"})
                        self.logger.error("❌ Could not extract valid JSON from response")
                        
                        # Save the final state and return empty
                        with open(extraction_file, 'w', encoding='utf-8') as f:
                            json.dump(extraction_debug, f, indent=2, ensure_ascii=False)
                        return []
                else:
                    extraction_debug.update({"fallback_parsing": "no_json_found"})
                    self.logger.error("❌ No JSON found in response")
                    
                    # Save the final state and return empty
                    with open(extraction_file, 'w', encoding='utf-8') as f:
                        json.dump(extraction_debug, f, indent=2, ensure_ascii=False)
                    return []
            
            # Convert to Entity objects
            entities = []
            entity_conversion_debug = []
            
            self.logger.info(f"🔄 Converting {len(entities_data)} entity data to Entity objects")
            
            for i, entity_data in enumerate(entities_data):
                entity_debug = {
                    "index": i,
                    "raw_data": entity_data,
                    "status": "processing"
                }
                
                try:
                    entity_type = EntityType(entity_data.get('entity_type', 'UNKNOWN'))
                    
                    entity = Entity(
                        name=entity_data.get('name', ''),
                        entity_type=entity_type,
                        summary=entity_data.get('summary', ''),
                        aliases=entity_data.get('aliases', []),
                        attributes=entity_data.get('attributes', {}),
                        confidence=entity_data.get('confidence', 0.0),
                        first_mentioned_at=datetime.now()
                    )
                    
                    entities.append(entity)
                    entity_debug.update({
                        "status": "success",
                        "entity_uuid": entity.uuid,
                        "entity_name": entity.name,
                        "entity_type": entity.entity_type.value
                    })
                    
                except (ValueError, KeyError) as e:
                    entity_debug.update({
                        "status": "failed",
                        "error": str(e)
                    })
                    self.logger.warning(f"⚠️ Skipping invalid entity data: {e}")
                    continue
                
                entity_conversion_debug.append(entity_debug)
            
            # Final extraction results
            extraction_debug.update({
                "status": "completed",
                "final_entity_count": len(entities),
                "entity_conversion_debug": entity_conversion_debug,
                "success": True
            })
            
            # Save final extraction debug
            with open(extraction_file, 'w', encoding='utf-8') as f:
                json.dump(extraction_debug, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"✅ Entity extraction completed: {len(entities)} entities extracted from episode {episode.id}")
            self.logger.info(f"💾 DEBUG: Final extraction results saved to {extraction_file}")
            
            return entities
            
        except Exception as e:
            self.logger.error(f"❌ Entity extraction failed: {e}")
            
            # Save error state
            extraction_debug.update({
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "success": False
            })
            
            with open(extraction_file, 'w', encoding='utf-8') as f:
                json.dump(extraction_debug, f, indent=2, ensure_ascii=False)
            
            self.logger.error(f"💾 DEBUG: Error details saved to {extraction_file}")
            return []
    
    def _extract_relationships(self, episode: Episode, entities: List[Entity]) -> List[Relationship]:
        """Extract relationships between entities"""
        if len(entities) < 2:
            self.logger.debug("Not enough entities for relationship extraction")
            return []
        
        try:
            # Set operation context for cost tracking
            if self.cost_tracker:
                self.cost_tracker.set_context(
                    episode_id=episode.id,
                    operation="relationship_extraction"
                )
            
            # For o-series models, use a simpler approach
            if self.model_name.startswith('o'):
                return self._extract_relationships_simple(episode, entities)
            
            # Prepare prompt
            context = {
                'episode_content': episode.content,
                'entities': [
                    {
                        'name': entity.name,
                        'entity_type': entity.entity_type.value,
                        'summary': entity.summary,
                        'aliases': entity.aliases,
                        'attributes': entity.attributes,
                        'confidence': entity.confidence
                    } for entity in entities
                ],
                'relationship_types': self.relationship_types
            }
            
            messages = RELATIONSHIP_EXTRACTION_STRUCTURED_PROMPT(context)
            
            # Prepare structured output schema if supported
            response_format = None
            if self.use_structured_output and self.supports_structured_output:
                response_format = create_relationship_extraction_schema()
            
            # Make API call with cost tracking and optional structured output
            response_text = self._call_openai_api(
                messages=messages, 
                operation_type="relationship_extraction",
                response_format=response_format
            )
            
            # Parse JSON response
            try:
                # First try to parse the entire response
                response_data = json.loads(response_text)
                relationships_data = response_data.get('relationships', [])
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse relationship extraction JSON: {e}")
                self.logger.error(f"Response text: {response_text[:500]}...")
                
                # Try to extract JSON from response if it's embedded in other text
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    try:
                        response_data = json.loads(json_match.group())
                        relationships_data = response_data.get('relationships', [])
                        self.logger.info("Successfully extracted JSON from embedded response")
                    except json.JSONDecodeError:
                        self.logger.error("Could not extract valid JSON from response")
                        return []
                else:
                    self.logger.error("No JSON found in response")
                    return []
            
            # Create entity name to ID mapping
            entity_map = {entity.name: entity.uuid for entity in entities}
            
            # Convert to Relationship objects
            relationships = []
            for rel_data in relationships_data:
                try:
                    # Find source and target entities
                    source_name = rel_data.get('source_entity', '')
                    target_name = rel_data.get('target_entity', '')
                    
                    source_entity_id = entity_map.get(source_name)
                    target_entity_id = entity_map.get(target_name)
                    
                    if not source_entity_id or not target_entity_id:
                        self.logger.warning(f"Could not find entities for relationship: {source_name} -> {target_name}")
                        continue
                    
                    relationship_type = RelationshipType.from_string(rel_data.get('relationship_type', 'UNKNOWN'))
                    
                    # Parse temporal information
                    valid_at = None
                    invalid_at = None
                    
                    if rel_data.get('valid_at'):
                        try:
                            valid_at = datetime.fromisoformat(rel_data['valid_at'].replace('Z', '+00:00'))
                        except ValueError:
                            pass
                    
                    if rel_data.get('invalid_at'):
                        try:
                            invalid_at = datetime.fromisoformat(rel_data['invalid_at'].replace('Z', '+00:00'))
                        except ValueError:
                            pass
                    
                    relationship = Relationship(
                        source_entity_id=source_entity_id,
                        target_entity_id=target_entity_id,
                        relationship_type=relationship_type,
                        fact=rel_data.get('fact', ''),
                        confidence=rel_data.get('confidence', 0.0),
                        valid_at=valid_at,
                        invalid_at=invalid_at,
                        attributes=rel_data.get('attributes', {})
                    )
                    
                    relationships.append(relationship)
                    
                except (ValueError, KeyError) as e:
                    self.logger.warning(f"Skipping invalid relationship data: {e}")
                    continue
            
            self.logger.debug(f"Extracted {len(relationships)} relationships from episode {episode.id}")
            return relationships
            
        except Exception as e:
            self.logger.error(f"Relationship extraction failed: {e}")
            return []
    
    def _extract_relationships_simple(self, episode: Episode, entities: List[Entity]) -> List[Relationship]:
        """
        Simplified relationship extraction for o-series models
        Uses a simpler prompt structure to avoid empty responses
        """
        try:
            self.logger.info(f"Using simplified relationship extraction for {self.model_name}")
            
            # Create simple entity list
            entity_names = [entity.name for entity in entities]
            entity_map = {entity.name: entity.uuid for entity in entities}
            
            # Create a very simple prompt
            messages = [
                {
                    "role": "system",
                    "content": "You are an AI that finds relationships between entities in text. Return only valid JSON with relationships array."
                },
                {
                    "role": "user", 
                    "content": f"""Find relationships between these entities in the text:

ENTITIES: {', '.join(entity_names)}

TEXT:
{episode.content[:2000]}...

Return JSON format:
{{"relationships": [{{"source_entity": "Name1", "target_entity": "Name2", "relationship_type": "RELATED_TO", "fact": "description", "confidence": 0.8}}]}}

Only return JSON, no other text."""
                }
            ]
            
            # Make API call
            response_text = self._call_openai_api(
                messages=messages,
                operation_type="relationship_extraction_simple"
            )
            
            if not response_text or response_text.strip() == "":
                self.logger.warning("Empty response from simple relationship extraction")
                return []
            
            # Parse response
            try:
                response_data = json.loads(response_text)
                relationships_data = response_data.get('relationships', [])
            except json.JSONDecodeError as e:
                self.logger.warning(f"Failed to parse simple relationship JSON: {e}")
                # Try to extract JSON
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    try:
                        response_data = json.loads(json_match.group())
                        relationships_data = response_data.get('relationships', [])
                    except json.JSONDecodeError:
                        self.logger.warning("Could not extract JSON from simple relationship response")
                        return []
                else:
                    return []
            
            # Convert to Relationship objects
            relationships = []
            for rel_data in relationships_data:
                try:
                    source_name = rel_data.get('source_entity', '')
                    target_name = rel_data.get('target_entity', '')
                    
                    source_entity_id = entity_map.get(source_name)
                    target_entity_id = entity_map.get(target_name)
                    
                    if not source_entity_id or not target_entity_id:
                        continue
                    
                    # Use a default relationship type for simple extraction
                    relationship_type = RelationshipType.from_string(rel_data.get('relationship_type', 'RELATED_TO'))
                    
                    relationship = Relationship(
                        source_entity_id=source_entity_id,
                        target_entity_id=target_entity_id,
                        relationship_type=relationship_type,
                        fact=rel_data.get('fact', ''),
                        confidence=rel_data.get('confidence', 0.7),
                        valid_at=datetime.now(),
                        invalid_at=None,
                        attributes={}
                    )
                    
                    relationships.append(relationship)
                    
                except (ValueError, KeyError) as e:
                    self.logger.warning(f"Skipping invalid simple relationship: {e}")
                    continue
            
            self.logger.debug(f"Simple extraction found {len(relationships)} relationships")
            return relationships
            
        except Exception as e:
            self.logger.error(f"Simple relationship extraction failed: {e}")
            return []
    
    def _extract_temporal_info(self, episode: Episode, relationships: List[Relationship]) -> List[Relationship]:
        """Extract temporal information for relationships"""
        if not relationships:
            return relationships
        
        try:
            # Set operation context for cost tracking
            if self.cost_tracker:
                self.cost_tracker.set_context(
                    episode_id=episode.id,
                    operation="temporal_extraction"
                )
            
            # Prepare prompt
            context = {
                'episode_content': episode.content,
                'relationships': [
                    {
                        'source_entity': rel.source_entity_id,
                        'target_entity': rel.target_entity_id,
                        'relationship_type': rel.relationship_type.value,
                        'fact': rel.fact,
                        'confidence': rel.confidence
                    } for rel in relationships
                ]
            }
            
            messages = TEMPORAL_EXTRACTION_STRUCTURED_PROMPT(context)
            
            # Make API call with cost tracking
            response_text = self._call_openai_api(
                messages=messages,
                operation_type="temporal_extraction"
            )
            
            # Parse JSON response
            try:
                response_data = json.loads(response_text)
                temporal_extractions = response_data.get('temporal_extractions', [])
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse temporal extraction JSON: {e}")
                
                # Try to extract JSON from response if it's embedded in other text (like ```json blocks)
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    try:
                        response_data = json.loads(json_match.group())
                        temporal_extractions = response_data.get('temporal_extractions', [])
                        self.logger.info("✅ Successfully extracted JSON from embedded temporal extraction response")
                    except json.JSONDecodeError:
                        self.logger.error("❌ Could not extract valid JSON from temporal extraction response")
                        return relationships
                else:
                    self.logger.error("❌ No JSON found in temporal extraction response")
                    return relationships
            
            # Update relationships with temporal information
            for extraction in temporal_extractions:
                try:
                    rel_index = extraction.get('relationship_index', 1) - 1  # Convert to 0-based
                    
                    if 0 <= rel_index < len(relationships):
                        relationship = relationships[rel_index]
                        
                        # Update temporal fields if provided
                        if extraction.get('valid_at'):
                            try:
                                relationship.valid_at = datetime.fromisoformat(
                                    extraction['valid_at'].replace('Z', '+00:00')
                                )
                            except ValueError:
                                pass
                        
                        if extraction.get('invalid_at'):
                            try:
                                relationship.invalid_at = datetime.fromisoformat(
                                    extraction['invalid_at'].replace('Z', '+00:00')
                                )
                            except ValueError:
                                pass
                        
                        # Update confidence if provided
                        if extraction.get('confidence'):
                            relationship.confidence = min(
                                relationship.confidence,
                                extraction['confidence']
                            )
                        
                except (ValueError, KeyError, IndexError) as e:
                    self.logger.warning(f"Skipping invalid temporal extraction: {e}")
                    continue
            
            self.logger.debug(f"Updated temporal information for {len(relationships)} relationships")
            return relationships
            
        except Exception as e:
            self.logger.error(f"Temporal extraction failed: {e}")
            return relationships
    
    def _deduplicate_entities(self, new_entities: List[Entity], existing_entities: List[Entity]) -> Tuple[List[Entity], int]:
        """Deduplicate entities using LLM"""
        if not new_entities or not existing_entities:
            return new_entities, 0
        
        try:
            # Set operation context for cost tracking
            if self.cost_tracker:
                self.cost_tracker.set_context(operation="entity_deduplication")
            
            # Combine entities for deduplication
            all_entities = existing_entities + new_entities
            
            # Prepare prompt
            context = {
                'entities': [
                    {
                        'name': entity.name,
                        'entity_type': entity.entity_type.value,
                        'summary': entity.summary,
                        'aliases': entity.aliases,
                        'attributes': entity.attributes,
                        'confidence': entity.confidence
                    } for entity in all_entities
                ]
            }
            messages = ENTITY_DEDUPLICATION_STRUCTURED_PROMPT(context)
            
            # Make API call with cost tracking
            response_text = self._call_openai_api(
                messages=messages,
                operation_type="entity_deduplication"
            )
            
            # Parse JSON response
            try:
                response_data = json.loads(response_text)
                duplicate_groups = response_data.get('duplicate_groups', [])
                unique_entities_indices = response_data.get('unique_entities', [])
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse entity deduplication JSON: {e}")
                
                # Try to extract JSON from response if it's embedded in other text (like ```json blocks)
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    try:
                        response_data = json.loads(json_match.group())
                        duplicate_groups = response_data.get('duplicate_groups', [])
                        unique_entities_indices = response_data.get('unique_entities', [])
                        self.logger.info("✅ Successfully extracted JSON from embedded entity deduplication response")
                    except json.JSONDecodeError:
                        self.logger.error("❌ Could not extract valid JSON from entity deduplication response")
                        return new_entities, 0
                else:
                    self.logger.error("❌ No JSON found in entity deduplication response")
                    return new_entities, 0
            
            # Process deduplication results
            final_entities = []
            entities_deduplicated = 0
            processed_indices = set()
            
            # Process duplicate groups
            for group_info in duplicate_groups:
                try:
                    duplicate_indices = group_info.get('duplicate_group', [])
                    
                    if len(duplicate_indices) < 2:
                        continue
                    
                    # Get entities to merge
                    entities_to_merge = []
                    for idx in duplicate_indices:
                        if 0 <= idx-1 < len(all_entities):  # Convert to 0-based
                            entities_to_merge.append(all_entities[idx-1])
                            processed_indices.add(idx-1)
                    
                    if entities_to_merge:
                        # Merge entities
                        merged_entity = self._merge_entities(entities_to_merge, group_info)
                        final_entities.append(merged_entity)
                        entities_deduplicated += len(entities_to_merge) - 1
                        
                except (ValueError, KeyError, IndexError) as e:
                    self.logger.warning(f"Error processing entity duplicate group: {e}")
                    continue
            
            # Add unique entities
            for idx in unique_entities_indices:
                try:
                    if 0 <= idx-1 < len(all_entities):  # Convert to 0-based
                        if idx-1 not in processed_indices:
                            final_entities.append(all_entities[idx-1])
                            processed_indices.add(idx-1)
                except (ValueError, IndexError) as e:
                    self.logger.warning(f"Error processing unique entity: {e}")
                    continue
            
            # Add any remaining unprocessed entities
            for i, entity in enumerate(all_entities):
                if i not in processed_indices:
                    final_entities.append(entity)
            
            self.logger.debug(f"Entity deduplication: {len(all_entities)} -> {len(final_entities)} (deduplicated: {entities_deduplicated})")
            return final_entities, entities_deduplicated
            
        except Exception as e:
            self.logger.error(f"Entity deduplication failed: {e}")
            return new_entities, 0
    
    def _deduplicate_relationships(self, new_relationships: List[Relationship], existing_relationships: List[Relationship]) -> Tuple[List[Relationship], int]:
        """Deduplicate relationships using LLM"""
        if not new_relationships or not existing_relationships:
            return new_relationships, 0
        
        try:
            # Set operation context for cost tracking
            if self.cost_tracker:
                self.cost_tracker.set_context(operation="relationship_deduplication")
            
            # Combine relationships for deduplication
            all_relationships = existing_relationships + new_relationships
            
            # Prepare prompt
            context = {
                'relationships': [
                    {
                        'source_entity': rel.source_entity_id,
                        'target_entity': rel.target_entity_id,
                        'relationship_type': rel.relationship_type.value,
                        'fact': rel.fact,
                        'confidence': rel.confidence,
                        'valid_at': rel.valid_at.isoformat() if rel.valid_at else 'N/A',
                        'invalid_at': rel.invalid_at.isoformat() if rel.invalid_at else 'N/A',
                        'attributes': rel.attributes
                    } for rel in all_relationships
                ]
            }
            messages = RELATIONSHIP_DEDUPLICATION_STRUCTURED_PROMPT(context)
            
            # Make API call with cost tracking
            response_text = self._call_openai_api(
                messages=messages,
                operation_type="relationship_deduplication"
            )
            
            # Parse JSON response
            try:
                response_data = json.loads(response_text)
                duplicate_groups = response_data.get('duplicate_groups', [])
                unique_relationships_indices = response_data.get('unique_relationships', [])
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse relationship deduplication JSON: {e}")
                
                # Try to extract JSON from response if it's embedded in other text (like ```json blocks)
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    try:
                        response_data = json.loads(json_match.group())
                        duplicate_groups = response_data.get('duplicate_groups', [])
                        unique_relationships_indices = response_data.get('unique_relationships', [])
                        self.logger.info("✅ Successfully extracted JSON from embedded relationship deduplication response")
                    except json.JSONDecodeError:
                        self.logger.error("❌ Could not extract valid JSON from relationship deduplication response")
                        return new_relationships, 0
                else:
                    self.logger.error("❌ No JSON found in relationship deduplication response")
                    return new_relationships, 0
            
            # Process deduplication results
            final_relationships = []
            relationships_deduplicated = 0
            processed_indices = set()
            
            # Process duplicate groups
            for group_info in duplicate_groups:
                try:
                    duplicate_indices = group_info.get('duplicate_group', [])
                    
                    if len(duplicate_indices) < 2:
                        continue
                    
                    # Get relationships to merge
                    relationships_to_merge = []
                    for idx in duplicate_indices:
                        if 0 <= idx-1 < len(all_relationships):  # Convert to 0-based
                            relationships_to_merge.append(all_relationships[idx-1])
                            processed_indices.add(idx-1)
                    
                    if relationships_to_merge:
                        # Merge relationships
                        merged_relationship = self._merge_relationships(relationships_to_merge, group_info)
                        final_relationships.append(merged_relationship)
                        relationships_deduplicated += len(relationships_to_merge) - 1
                        
                except (ValueError, KeyError, IndexError) as e:
                    self.logger.warning(f"Error processing relationship duplicate group: {e}")
                    continue
            
            # Add unique relationships
            for idx in unique_relationships_indices:
                try:
                    if 0 <= idx-1 < len(all_relationships):  # Convert to 0-based
                        if idx-1 not in processed_indices:
                            final_relationships.append(all_relationships[idx-1])
                            processed_indices.add(idx-1)
                except (ValueError, IndexError) as e:
                    self.logger.warning(f"Error processing unique relationship: {e}")
                    continue
            
            # Add any remaining unprocessed relationships
            for i, relationship in enumerate(all_relationships):
                if i not in processed_indices:
                    final_relationships.append(relationship)
            
            self.logger.debug(f"Relationship deduplication: {len(all_relationships)} -> {len(final_relationships)} (deduplicated: {relationships_deduplicated})")
            return final_relationships, relationships_deduplicated
            
        except Exception as e:
            self.logger.error(f"Relationship deduplication failed: {e}")
            return new_relationships, 0
    
    def _merge_entities(self, entities: List[Entity], merge_info: Dict[str, Any]) -> Entity:
        """Merge multiple entities into one"""
        if not entities:
            raise ValueError("No entities to merge")
        
        # Use the first entity as base
        base_entity = entities[0]
        
        # Merge information from merge_info and other entities
        merged_name = merge_info.get('primary_name', base_entity.name)
        merged_type = EntityType(merge_info.get('merged_entity_type', base_entity.entity_type.value))
        merged_aliases = list(set(merge_info.get('merged_aliases', [])))
        merged_summary = merge_info.get('merged_summary', base_entity.summary)
        merged_attributes = merge_info.get('merged_attributes', base_entity.attributes.copy())
        
        # Combine source episodes and documents
        all_source_episodes = []
        all_source_documents = []
        earliest_mention = base_entity.first_mentioned_at
        
        for entity in entities:
            all_source_episodes.extend(entity.source_episodes)
            all_source_documents.extend(entity.source_documents)
            
            if entity.first_mentioned_at and (not earliest_mention or entity.first_mentioned_at < earliest_mention):
                earliest_mention = entity.first_mentioned_at
        
        # Remove duplicates
        all_source_episodes = list(set(all_source_episodes))
        all_source_documents = list(set(all_source_documents))
        
        # Create merged entity
        merged_entity = Entity(
            name=merged_name,
            entity_type=merged_type,
            summary=merged_summary,
            aliases=merged_aliases,
            attributes=merged_attributes,
            confidence=merge_info.get('confidence', base_entity.confidence),
            first_mentioned_at=earliest_mention,
            last_updated_at=datetime.now(),
            source_episodes=all_source_episodes,
            source_documents=all_source_documents
        )
        
        return merged_entity
    
    def _merge_relationships(self, relationships: List[Relationship], merge_info: Dict[str, Any]) -> Relationship:
        """Merge multiple relationships into one"""
        if not relationships:
            raise ValueError("No relationships to merge")
        
        # Use the first relationship as base
        base_relationship = relationships[0]
        
        # Merge information from merge_info
        merged_fact = merge_info.get('merged_fact', base_relationship.fact)
        merged_type = RelationshipType.from_string(merge_info.get('merged_relationship_type', base_relationship.relationship_type.value))
        merged_confidence = merge_info.get('merged_confidence', base_relationship.confidence)
        merged_attributes = merge_info.get('merged_attributes', base_relationship.attributes.copy())
        
        # Parse temporal information
        merged_valid_at = base_relationship.valid_at
        merged_invalid_at = base_relationship.invalid_at
        
        if merge_info.get('merged_valid_at'):
            try:
                merged_valid_at = datetime.fromisoformat(merge_info['merged_valid_at'].replace('Z', '+00:00'))
            except ValueError:
                pass
        
        if merge_info.get('merged_invalid_at'):
            try:
                merged_invalid_at = datetime.fromisoformat(merge_info['merged_invalid_at'].replace('Z', '+00:00'))
            except ValueError:
                pass
        
        # Combine source episodes and documents
        all_source_episodes = []
        all_source_documents = []
        
        for relationship in relationships:
            all_source_episodes.extend(relationship.source_episodes)
            all_source_documents.extend(relationship.source_documents)
        
        # Remove duplicates
        all_source_episodes = list(set(all_source_episodes))
        all_source_documents = list(set(all_source_documents))
        
        # Create merged relationship
        merged_relationship = Relationship(
            source_entity_id=merge_info.get('merged_source_entity', base_relationship.source_entity_id),
            target_entity_id=merge_info.get('merged_target_entity', base_relationship.target_entity_id),
            relationship_type=merged_type,
            fact=merged_fact,
            confidence=merged_confidence,
            valid_at=merged_valid_at,
            invalid_at=merged_invalid_at,
            attributes=merged_attributes,
            source_episodes=all_source_episodes,
            source_documents=all_source_documents
        )
        
        return merged_relationship
    
    def _update_provenance(self, entities: List[Entity], relationships: List[Relationship], episode_id: str, document_id: str):
        """Update provenance information for entities and relationships"""
        for entity in entities:
            if episode_id not in entity.source_episodes:
                entity.source_episodes.append(episode_id)
            if document_id not in entity.source_documents:
                entity.source_documents.append(document_id)
            entity.last_updated_at = datetime.now()
        
        for relationship in relationships:
            if episode_id not in relationship.source_episodes:
                relationship.source_episodes.append(episode_id)
            if document_id not in relationship.source_documents:
                relationship.source_documents.append(document_id)
    
    def _update_stats(self, result: ExtractionResult):
        """Update internal statistics"""
        self.stats['episodes_processed'] += 1
        self.stats['entities_extracted'] += result.entities_extracted
        self.stats['relationships_extracted'] += result.relationships_extracted
        self.stats['entities_deduplicated'] += result.entities_deduplicated
        self.stats['relationships_deduplicated'] += result.relationships_deduplicated
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current extraction statistics"""
        return self.stats.copy() 