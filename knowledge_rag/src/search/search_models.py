"""
Search Models for Knowledge RAG System

Defines the data models and configuration for search queries and results
with support for hybrid search strategies and result ranking.
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from ..storage.models import SearchResult


class SearchType(Enum):
    """Types of search strategies available"""
    SEMANTIC_ENTITIES = "semantic_entities"
    SEMANTIC_EPISODES = "semantic_episodes" 
    GRAPH_ENTITIES = "graph_entities"
    GRAPH_EPISODES = "graph_episodes"
    GRAPH_TRAVERSAL = "graph_traversal"
    HYBRID = "hybrid"
    FULL_TEXT = "full_text"


class RankingStrategy(Enum):
    """Result ranking and fusion strategies"""
    RELEVANCE_SCORE = "relevance_score"  # Sort by individual relevance
    RRF = "rrf"  # Reciprocal Rank Fusion
    WEIGHTED_COMBINATION = "weighted_combination"
    GRAPH_CENTRALITY = "graph_centrality"
    TEMPORAL_RECENCY = "temporal_recency"


@dataclass
class SearchQuery:
    """Represents a search query with all parameters"""
    
    # Core query
    query_text: str
    
    # Search strategy
    search_types: List[SearchType] = field(default_factory=lambda: [SearchType.HYBRID])
    
    # Result filtering
    limit: int = 10
    min_score: float = 0.0
    entity_types: Optional[List[str]] = None
    document_filter: Optional[str] = None
    
    # Time-based filtering
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    # Graph traversal parameters
    graph_hops: int = 2
    traversal_entities: Optional[List[str]] = None
    
    # Result processing
    ranking_strategy: RankingStrategy = RankingStrategy.RRF
    include_context: bool = True
    deduplicate_results: bool = True
    
    # Metadata
    query_id: str = field(default_factory=lambda: f"query_{datetime.now().timestamp()}")
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SearchConfig:
    """Configuration for search engine behavior"""
    
    # Search weights for hybrid search
    semantic_weight: float = 0.4
    graph_weight: float = 0.3
    fulltext_weight: float = 0.3
    
    # Embedding parameters
    embedding_model: str = "text-embedding-3-small"
    similarity_threshold: float = 0.7
    
    # Graph traversal parameters
    max_graph_hops: int = 3
    max_traversal_results: int = 50
    
    # Result fusion parameters
    rrf_k: int = 60  # RRF parameter
    max_results_per_type: int = 20
    
    # Performance parameters
    search_timeout_seconds: int = 30
    parallel_search: bool = True
    
    # Quality parameters
    min_confidence_score: float = 0.1
    boost_recent_results: bool = True
    boost_high_confidence: bool = True


@dataclass
class SearchResults:
    """Container for search results with metadata"""
    
    # Core results
    results: List[SearchResult] = field(default_factory=list)
    
    # Search metadata
    query: SearchQuery = None
    total_results: int = 0
    search_time_ms: float = 0.0
    
    # Result breakdown by search type
    results_by_type: Dict[str, int] = field(default_factory=dict)
    
    # Quality metrics
    average_score: float = 0.0
    max_score: float = 0.0
    score_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Document traceability
    source_documents: Dict[str, int] = field(default_factory=dict)  # doc_uuid -> count
    document_coverage: float = 0.0  # Percentage of available documents covered
    
    # Debug information
    search_strategy_used: List[str] = field(default_factory=list)
    fusion_method: str = ""
    warnings: List[str] = field(default_factory=list)
    
    def add_result(self, result: SearchResult):
        """Add a search result and update metadata"""
        self.results.append(result)
        self.total_results = len(self.results)
        
        # Update score metrics
        if self.results:
            scores = [r.score for r in self.results]
            self.average_score = sum(scores) / len(scores)
            self.max_score = max(scores)
        
        # Update document coverage
        if result.document_uuid not in self.source_documents:
            self.source_documents[result.document_uuid] = 0
        self.source_documents[result.document_uuid] += 1
        
        # Update result type breakdown
        if result.result_type not in self.results_by_type:
            self.results_by_type[result.result_type] = 0
        self.results_by_type[result.result_type] += 1
    
    def get_top_results(self, n: int = 5) -> List[SearchResult]:
        """Get top N results by score"""
        return sorted(self.results, key=lambda x: x.score, reverse=True)[:n]
    
    def get_results_by_document(self) -> Dict[str, List[SearchResult]]:
        """Group results by source document"""
        by_doc = {}
        for result in self.results:
            doc_uuid = result.document_uuid
            if doc_uuid not in by_doc:
                by_doc[doc_uuid] = []
            by_doc[doc_uuid].append(result)
        return by_doc
    
    def get_results_by_type(self) -> Dict[str, List[SearchResult]]:
        """Group results by result type"""
        by_type = {}
        for result in self.results:
            result_type = result.result_type
            if result_type not in by_type:
                by_type[result_type] = []
            by_type[result_type].append(result)
        return by_type
    
    def filter_by_score(self, min_score: float) -> 'SearchResults':
        """Create new SearchResults with filtered results"""
        filtered_results = SearchResults()
        filtered_results.query = self.query
        filtered_results.fusion_method = self.fusion_method
        filtered_results.search_strategy_used = self.search_strategy_used.copy()
        
        for result in self.results:
            if result.score >= min_score:
                filtered_results.add_result(result)
        
        return filtered_results
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'total_results': self.total_results,
            'search_time_ms': self.search_time_ms,
            'average_score': self.average_score,
            'max_score': self.max_score,
            'results_by_type': self.results_by_type,
            'source_documents': len(self.source_documents),
            'search_strategy_used': self.search_strategy_used,
            'fusion_method': self.fusion_method,
            'results': [
                {
                    'content': result.get_display_content(),
                    'score': result.score,
                    'result_type': result.result_type,
                    'source_context': result.get_source_context(),
                    'document_name': result.document_name,
                    'entity_name': result.entity_name,
                    'entity_type': result.entity_type,
                    'relationship_fact': result.relationship_fact,
                    'metadata': result.metadata
                }
                for result in self.results
            ],
            'warnings': self.warnings
        }


@dataclass
class SearchAnalytics:
    """Analytics and metrics for search performance"""
    
    # Query statistics
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    
    # Performance metrics
    average_search_time_ms: float = 0.0
    fastest_search_ms: float = float('inf')
    slowest_search_ms: float = 0.0
    
    # Result quality metrics
    average_results_per_query: float = 0.0
    average_score_per_query: float = 0.0
    
    # Search type usage
    search_type_usage: Dict[str, int] = field(default_factory=dict)
    
    # Common queries
    popular_queries: Dict[str, int] = field(default_factory=dict)
    
    # Document access patterns
    most_accessed_documents: Dict[str, int] = field(default_factory=dict)
    
    # Error tracking
    common_errors: Dict[str, int] = field(default_factory=dict)
    
    def record_query(
        self, 
        query: SearchQuery, 
        results: SearchResults,
        success: bool = True,
        error: Optional[str] = None
    ):
        """Record a query for analytics"""
        self.total_queries += 1
        
        if success:
            self.successful_queries += 1
            
            # Update performance metrics
            if results.search_time_ms > 0:
                if self.average_search_time_ms == 0:
                    self.average_search_time_ms = results.search_time_ms
                else:
                    self.average_search_time_ms = (
                        self.average_search_time_ms + results.search_time_ms
                    ) / 2
                
                self.fastest_search_ms = min(self.fastest_search_ms, results.search_time_ms)
                self.slowest_search_ms = max(self.slowest_search_ms, results.search_time_ms)
            
            # Update result metrics
            self.average_results_per_query = (
                self.average_results_per_query + results.total_results
            ) / 2
            
            if results.average_score > 0:
                self.average_score_per_query = (
                    self.average_score_per_query + results.average_score
                ) / 2
            
            # Track search type usage
            for search_type in results.search_strategy_used:
                if search_type not in self.search_type_usage:
                    self.search_type_usage[search_type] = 0
                self.search_type_usage[search_type] += 1
            
            # Track popular queries
            query_text = query.query_text.lower()
            if query_text not in self.popular_queries:
                self.popular_queries[query_text] = 0
            self.popular_queries[query_text] += 1
            
            # Track document access
            for doc_uuid in results.source_documents:
                if doc_uuid not in self.most_accessed_documents:
                    self.most_accessed_documents[doc_uuid] = 0
                self.most_accessed_documents[doc_uuid] += results.source_documents[doc_uuid]
        
        else:
            self.failed_queries += 1
            if error:
                if error not in self.common_errors:
                    self.common_errors[error] = 0
                self.common_errors[error] += 1
    
    def get_success_rate(self) -> float:
        """Calculate query success rate"""
        if self.total_queries == 0:
            return 0.0
        return self.successful_queries / self.total_queries
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        return {
            'total_queries': self.total_queries,
            'success_rate': self.get_success_rate(),
            'average_search_time_ms': self.average_search_time_ms,
            'average_results_per_query': self.average_results_per_query,
            'average_score_per_query': self.average_score_per_query,
            'most_used_search_types': dict(
                sorted(self.search_type_usage.items(), key=lambda x: x[1], reverse=True)[:5]
            ),
            'top_queries': dict(
                sorted(self.popular_queries.items(), key=lambda x: x[1], reverse=True)[:10]
            )
        } 