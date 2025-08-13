"""
Document type schemas for knowledge extraction.

This module defines custom extraction schemas for different document types,
ensuring that the most relevant entities and relationships are extracted
based on the document's domain and purpose.
"""

from enum import Enum
from typing import List, Dict, Any, Optional, Set
from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """Enumeration of supported document types."""
    
    ACADEMIC_PAPER = "academic_paper"
    INDUSTRY_REPORT = "industry_report"
    FINANCIAL_REPORT = "financial_report"
    TECHNICAL_DOCUMENTATION = "technical_documentation"
    NEWS_ARTICLE = "news_article"
    LEGAL_DOCUMENT = "legal_document"
    MEDICAL_RECORD = "medical_record"
    PATENT = "patent"
    BOOK = "book"
    GENERAL = "general"


class EntitySchema(BaseModel):
    """Schema for entity extraction configuration."""
    
    entity_types: List[str] = Field(description="List of entity types to extract")
    priority_entities: List[str] = Field(description="High-priority entity types")
    custom_attributes: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Custom attributes to extract for specific entity types"
    )
    extraction_hints: Dict[str, str] = Field(
        default_factory=dict,
        description="Hints for extracting specific entity types"
    )


class RelationshipSchema(BaseModel):
    """Schema for relationship extraction configuration."""
    
    relationship_types: List[str] = Field(description="List of relationship types to extract")
    priority_relationships: List[str] = Field(description="High-priority relationship types")
    temporal_focus: bool = Field(default=False, description="Whether to focus on temporal relationships")
    causal_focus: bool = Field(default=False, description="Whether to focus on causal relationships")
    extraction_hints: Dict[str, str] = Field(
        default_factory=dict,
        description="Hints for extracting specific relationship types"
    )


class DocumentSchema(BaseModel):
    """Complete extraction schema for a document type."""
    
    document_type: DocumentType = Field(description="Type of document")
    description: str = Field(description="Description of the document type")
    entity_schema: EntitySchema = Field(description="Entity extraction configuration")
    relationship_schema: RelationshipSchema = Field(description="Relationship extraction configuration")
    extraction_focus: List[str] = Field(description="Key areas to focus on during extraction")
    keywords: List[str] = Field(default_factory=list, description="Keywords indicating this document type")


# Academic Paper Schema
ACADEMIC_PAPER_SCHEMA = DocumentSchema(
    document_type=DocumentType.ACADEMIC_PAPER,
    description="Scientific and research papers, conference proceedings, journal articles",
    entity_schema=EntitySchema(
        entity_types=[
            "CONCEPT", "METHODOLOGY", "ALGORITHM", "MODEL", "DATASET",
            "METRIC", "TECHNOLOGY", "FRAMEWORK", "THEORY", "HYPOTHESIS",
            "EXPERIMENT", "RESULT", "CONTRIBUTION", "LIMITATION"
        ],
        priority_entities=["METHODOLOGY", "ALGORITHM", "MODEL", "CONTRIBUTION"],
        custom_attributes={
            "METHODOLOGY": ["approach_type", "novelty", "complexity", "requirements"],
            "ALGORITHM": ["time_complexity", "space_complexity", "parameters", "optimization"],
            "MODEL": ["architecture", "parameters", "training_method", "performance"],
            "DATASET": ["size", "domain", "source", "preprocessing"],
            "RESULT": ["metric_values", "improvement", "statistical_significance"]
        },
        extraction_hints={
            "METHODOLOGY": "Look for methods, approaches, techniques, procedures",
            "CONTRIBUTION": "Focus on novel contributions, improvements, innovations",
            "LIMITATION": "Identify acknowledged limitations, future work, gaps"
        }
    ),
    relationship_schema=RelationshipSchema(
        relationship_types=[
            "IMPROVES_UPON", "EXTENDS", "APPLIES", "EVALUATES_WITH",
            "OUTPERFORMS", "BUILDS_ON", "CONTRADICTS", "VALIDATES",
            "USES_DATASET", "IMPLEMENTS", "COMPARES_TO", "INSPIRED_BY",
            "ADDRESSES_LIMITATION_OF", "ENABLES", "REQUIRES"
        ],
        priority_relationships=["IMPROVES_UPON", "OUTPERFORMS", "BUILDS_ON", "USES_DATASET"],
        temporal_focus=False,
        causal_focus=True,
        extraction_hints={
            "IMPROVES_UPON": "Look for performance improvements, better results",
            "OUTPERFORMS": "Find comparative results, benchmarks, evaluations",
            "BUILDS_ON": "Identify foundational work, prior art, references"
        }
    ),
    extraction_focus=[
        "Novel contributions and innovations",
        "Methodologies and algorithms",
        "Experimental results and evaluations",
        "Comparisons with existing work",
        "Theoretical foundations",
        "Limitations and future work"
    ],
    keywords=["abstract", "methodology", "experiment", "results", "conclusion", "references"]
)

# Industry Report Schema
INDUSTRY_REPORT_SCHEMA = DocumentSchema(
    document_type=DocumentType.INDUSTRY_REPORT,
    description="Market analysis, industry trends, business intelligence reports",
    entity_schema=EntitySchema(
        entity_types=[
            "COMPANY", "PRODUCT", "MARKET", "TECHNOLOGY", "TREND",
            "COMPETITOR", "CUSTOMER_SEGMENT", "PARTNERSHIP", "ACQUISITION",
            "REGULATION", "RISK", "OPPORTUNITY", "STRATEGY"
        ],
        priority_entities=["COMPANY", "PRODUCT", "MARKET", "TREND"],
        custom_attributes={
            "COMPANY": ["market_cap", "revenue", "growth_rate", "market_share", "headquarters"],
            "PRODUCT": ["category", "price_range", "target_market", "features", "launch_date"],
            "MARKET": ["size", "growth_rate", "segments", "geography", "maturity"],
            "TREND": ["impact", "timeframe", "drivers", "barriers"]
        },
        extraction_hints={
            "TREND": "Look for market trends, industry shifts, emerging patterns",
            "OPPORTUNITY": "Identify growth opportunities, market gaps, potential",
            "RISK": "Find challenges, threats, market risks, barriers"
        }
    ),
    relationship_schema=RelationshipSchema(
        relationship_types=[
            "COMPETES_WITH", "PARTNERS_WITH", "ACQUIRES", "SUPPLIES_TO",
            "TARGETS_MARKET", "DISRUPTS", "DOMINATES", "ENTERS_MARKET",
            "EXITS_MARKET", "INVESTS_IN", "DEVELOPS", "LAUNCHES",
            "DISCONTINUES", "REGULATES", "THREATENS"
        ],
        priority_relationships=["COMPETES_WITH", "PARTNERS_WITH", "ACQUIRES", "LAUNCHES"],
        temporal_focus=True,
        causal_focus=True,
        extraction_hints={
            "COMPETES_WITH": "Identify competitive relationships, market rivalry",
            "PARTNERS_WITH": "Find strategic partnerships, alliances, collaborations",
            "DISRUPTS": "Look for disruptive technologies, market disruptions"
        }
    ),
    extraction_focus=[
        "Market dynamics and competition",
        "Product launches and innovations",
        "Strategic partnerships and acquisitions",
        "Market trends and forecasts",
        "Regulatory environment",
        "Growth opportunities and risks"
    ],
    keywords=["market", "industry", "growth", "revenue", "competition", "forecast", "trends"]
)

# Financial Report Schema
FINANCIAL_REPORT_SCHEMA = DocumentSchema(
    document_type=DocumentType.FINANCIAL_REPORT,
    description="Financial statements, earnings reports, investment analysis",
    entity_schema=EntitySchema(
        entity_types=[
            "COMPANY", "FINANCIAL_METRIC", "REVENUE_STREAM", "EXPENSE",
            "ASSET", "LIABILITY", "INVESTMENT", "DIVIDEND", "FORECAST",
            "RISK_FACTOR", "SEGMENT", "SUBSIDIARY", "AUDITOR"
        ],
        priority_entities=["COMPANY", "FINANCIAL_METRIC", "REVENUE_STREAM", "RISK_FACTOR"],
        custom_attributes={
            "FINANCIAL_METRIC": ["value", "period", "change_percentage", "comparison"],
            "REVENUE_STREAM": ["amount", "growth_rate", "percentage_of_total"],
            "RISK_FACTOR": ["severity", "likelihood", "mitigation_strategy"],
            "FORECAST": ["period", "assumptions", "confidence_level"]
        },
        extraction_hints={
            "FINANCIAL_METRIC": "Extract metrics like revenue, profit, EBITDA, margins",
            "RISK_FACTOR": "Identify financial risks, market risks, operational risks",
            "FORECAST": "Find projections, guidance, estimates"
        }
    ),
    relationship_schema=RelationshipSchema(
        relationship_types=[
            "INCREASED_BY", "DECREASED_BY", "IMPACTS", "DRIVES",
            "OWNS", "OWES", "INVESTS_IN", "DIVESTS", "CONSOLIDATES",
            "REPORTS_TO", "AUDITED_BY", "REGULATED_BY"
        ],
        priority_relationships=["INCREASED_BY", "DECREASED_BY", "IMPACTS", "OWNS"],
        temporal_focus=True,
        causal_focus=True,
        extraction_hints={
            "IMPACTS": "Find causal relationships between metrics and factors",
            "DRIVES": "Identify key drivers of financial performance"
        }
    ),
    extraction_focus=[
        "Financial performance metrics",
        "Revenue and profitability",
        "Risk factors and mitigation",
        "Growth drivers and challenges",
        "Segment performance",
        "Forward-looking statements"
    ],
    keywords=["revenue", "profit", "earnings", "EBITDA", "margin", "growth", "risk", "forecast"]
)

# Technical Documentation Schema
TECHNICAL_DOCUMENTATION_SCHEMA = DocumentSchema(
    document_type=DocumentType.TECHNICAL_DOCUMENTATION,
    description="API documentation, user manuals, technical specifications",
    entity_schema=EntitySchema(
        entity_types=[
            "COMPONENT", "API", "FUNCTION", "PARAMETER", "CONFIGURATION",
            "DEPENDENCY", "VERSION", "FEATURE", "REQUIREMENT", "EXAMPLE",
            "ERROR", "LIMITATION", "BEST_PRACTICE"
        ],
        priority_entities=["COMPONENT", "API", "FUNCTION", "CONFIGURATION"],
        custom_attributes={
            "API": ["endpoint", "method", "authentication", "rate_limit"],
            "FUNCTION": ["parameters", "return_type", "exceptions", "complexity"],
            "CONFIGURATION": ["default_value", "valid_range", "required", "format"],
            "ERROR": ["error_code", "severity", "resolution"]
        },
        extraction_hints={
            "API": "Look for endpoints, methods, interfaces",
            "CONFIGURATION": "Find settings, parameters, options",
            "BEST_PRACTICE": "Identify recommendations, guidelines, tips"
        }
    ),
    relationship_schema=RelationshipSchema(
        relationship_types=[
            "DEPENDS_ON", "IMPLEMENTS", "EXTENDS", "OVERRIDES",
            "CALLS", "RETURNS", "THROWS", "CONFIGURES", "REQUIRES",
            "INCOMPATIBLE_WITH", "DEPRECATED_BY", "REPLACES"
        ],
        priority_relationships=["DEPENDS_ON", "IMPLEMENTS", "REQUIRES", "CALLS"],
        temporal_focus=False,
        causal_focus=False,
        extraction_hints={
            "DEPENDS_ON": "Find dependencies, prerequisites, requirements",
            "IMPLEMENTS": "Identify interface implementations, protocol conformance"
        }
    ),
    extraction_focus=[
        "System architecture and components",
        "APIs and interfaces",
        "Configuration and setup",
        "Dependencies and requirements",
        "Usage examples and best practices",
        "Error handling and troubleshooting"
    ],
    keywords=["API", "function", "method", "parameter", "configuration", "example", "usage"]
)

# Patent Schema
PATENT_SCHEMA = DocumentSchema(
    document_type=DocumentType.PATENT,
    description="Patent applications and granted patents",
    entity_schema=EntitySchema(
        entity_types=[
            "INVENTION", "CLAIM", "EMBODIMENT", "PRIOR_ART", "INVENTOR",
            "ASSIGNEE", "TECHNICAL_FIELD", "COMPONENT", "METHOD_STEP",
            "ADVANTAGE", "APPLICATION", "MATERIAL"
        ],
        priority_entities=["INVENTION", "CLAIM", "EMBODIMENT", "ADVANTAGE"],
        custom_attributes={
            "INVENTION": ["novelty", "utility", "non_obviousness"],
            "CLAIM": ["claim_type", "dependency", "scope"],
            "EMBODIMENT": ["preferred", "alternative", "example"],
            "PRIOR_ART": ["reference_number", "relevance"]
        },
        extraction_hints={
            "CLAIM": "Focus on independent and dependent claims",
            "ADVANTAGE": "Identify technical advantages, improvements",
            "PRIOR_ART": "Find referenced patents, publications"
        }
    ),
    relationship_schema=RelationshipSchema(
        relationship_types=[
            "COMPRISES", "IMPROVES_OVER", "DIFFERS_FROM", "COMBINES",
            "ENABLES", "PREVENTS", "REPLACES", "REFERENCES",
            "ASSIGNED_TO", "INVENTED_BY", "FILED_BY", "CITES"
        ],
        priority_relationships=["COMPRISES", "IMPROVES_OVER", "ENABLES", "CITES"],
        temporal_focus=False,
        causal_focus=True,
        extraction_hints={
            "COMPRISES": "Identify component relationships, system composition",
            "IMPROVES_OVER": "Find improvements over prior art"
        }
    ),
    extraction_focus=[
        "Novel inventive concepts",
        "Claims and their scope",
        "Technical advantages",
        "Embodiments and examples",
        "Prior art and differentiators",
        "Industrial applications"
    ],
    keywords=["claim", "embodiment", "invention", "comprises", "wherein", "thereof"]
)


# Schema Registry
SCHEMA_REGISTRY: Dict[DocumentType, DocumentSchema] = {
    DocumentType.ACADEMIC_PAPER: ACADEMIC_PAPER_SCHEMA,
    DocumentType.INDUSTRY_REPORT: INDUSTRY_REPORT_SCHEMA,
    DocumentType.FINANCIAL_REPORT: FINANCIAL_REPORT_SCHEMA,
    DocumentType.TECHNICAL_DOCUMENTATION: TECHNICAL_DOCUMENTATION_SCHEMA,
    DocumentType.PATENT: PATENT_SCHEMA,
}


def get_schema(document_type: DocumentType) -> DocumentSchema:
    """
    Get the extraction schema for a document type.
    
    Args:
        document_type: Type of document
        
    Returns:
        DocumentSchema for the specified type
    """
    return SCHEMA_REGISTRY.get(document_type, create_general_schema())


def create_general_schema() -> DocumentSchema:
    """
    Create a general-purpose schema for unclassified documents.
    
    Returns:
        General DocumentSchema
    """
    return DocumentSchema(
        document_type=DocumentType.GENERAL,
        description="General document without specific classification",
        entity_schema=EntitySchema(
            entity_types=[
                "PERSON", "ORGANIZATION", "LOCATION", "CONCEPT",
                "EVENT", "PRODUCT", "TECHNOLOGY", "DOCUMENT"
            ],
            priority_entities=["PERSON", "ORGANIZATION", "CONCEPT"],
            custom_attributes={},
            extraction_hints={}
        ),
        relationship_schema=RelationshipSchema(
            relationship_types=[
                "RELATED_TO", "PART_OF", "LOCATED_IN", "WORKS_FOR",
                "OWNS", "CREATED_BY", "MENTIONS", "REFERENCES"
            ],
            priority_relationships=["RELATED_TO", "PART_OF", "WORKS_FOR"],
            temporal_focus=False,
            causal_focus=False,
            extraction_hints={}
        ),
        extraction_focus=[
            "Key entities and their relationships",
            "Important concepts and ideas",
            "Events and timeline",
            "People and organizations"
        ],
        keywords=[]
    )