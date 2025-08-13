"""
Document type classifier for automatic schema selection.

This module provides classification functionality to identify document types
and automatically select the appropriate extraction schema.
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter

from extraction.document_schemas import DocumentType, DocumentSchema, get_schema


logger = logging.getLogger(__name__)


class DocumentClassifier:
    """
    Classifier to identify document types based on content and structure.
    
    Uses heuristic rules and keyword matching for initial classification,
    with optional LLM-based classification for higher accuracy.
    """
    
    def __init__(self, use_llm: bool = False, llm_model: str = "gpt-3.5-turbo"):
        """
        Initialize the document classifier.
        
        Args:
            use_llm: Whether to use LLM for classification
            llm_model: LLM model to use for classification
        """
        self.use_llm = use_llm
        self.llm_model = llm_model
        
        # Define classification patterns
        self._init_patterns()
    
    def _init_patterns(self):
        """Initialize regex patterns and keywords for classification."""
        
        # Academic paper patterns
        self.academic_patterns = {
            "abstract": r"\b(abstract|summary)\b",
            "methodology": r"\b(methodology|methods|approach|experiments?)\b",
            "results": r"\b(results|findings|evaluation|performance)\b",
            "references": r"\b(references|bibliography|citations)\b",
            "equations": r"\\(begin|end)\{equation\}|\$\$.*?\$\$",
            "citations": r"\[\d+\]|\(\w+,?\s*\d{4}\)",
            "sections": r"^#+\s*(introduction|related work|methodology|results|conclusion)",
        }
        
        # Industry report patterns
        self.industry_patterns = {
            "market": r"\b(market|industry|sector|segment)\b",
            "competition": r"\b(competitors?|rivalry|market share)\b",
            "forecast": r"\b(forecast|projection|outlook|trend)\b",
            "revenue": r"\b(revenue|sales|growth rate|CAGR)\b",
            "strategy": r"\b(strategy|strategic|positioning)\b",
            "swot": r"\b(strengths?|weaknesses?|opportunities|threats?)\b",
        }
        
        # Financial report patterns
        self.financial_patterns = {
            "financial_metrics": r"\b(revenue|profit|EBITDA|margin|earnings)\b",
            "financial_statements": r"\b(balance sheet|income statement|cash flow)\b",
            "accounting": r"\b(assets?|liabilities|equity|depreciation)\b",
            "currency": r"\$\d+|\d+\s*(million|billion|thousand)",
            "fiscal": r"\b(fiscal|quarter|Q[1-4]|FY\d{2,4})\b",
            "ratios": r"\b(P/E|ROI|ROE|debt.to.equity)\b",
        }
        
        # Technical documentation patterns
        self.technical_patterns = {
            "api": r"\b(API|endpoint|REST|GraphQL|request|response)\b",
            "code": r"```[\w]*\n.*?\n```|`[^`]+`",
            "functions": r"\b(function|method|class|interface|parameter)\b",
            "configuration": r"\b(config|configuration|settings?|options?)\b",
            "examples": r"\b(example|usage|sample|snippet)\b",
            "installation": r"\b(install|setup|requirements?|dependencies)\b",
        }
        
        # Patent patterns
        self.patent_patterns = {
            "claims": r"\b(claims?|wherein|thereof|comprising)\b",
            "embodiment": r"\b(embodiments?|implementations?|variations?)\b",
            "invention": r"\b(invention|inventive|novelty|utility)\b",
            "patent_refs": r"U\.S\.\s*Pat\.\s*No\.|Patent\s*\d+",
            "legal": r"\b(assignee|inventor|filed|granted)\b",
        }
    
    def classify_by_content(self, content: str) -> Tuple[DocumentType, float]:
        """
        Classify document based on content analysis.
        
        Args:
            content: Document text content
            
        Returns:
            Tuple of (DocumentType, confidence_score)
        """
        content_lower = content.lower()
        
        # Calculate scores for each document type
        scores = {}
        
        # Academic paper scoring
        academic_score = 0
        for pattern_name, pattern in self.academic_patterns.items():
            matches = len(re.findall(pattern, content_lower, re.MULTILINE | re.IGNORECASE))
            if pattern_name in ["abstract", "methodology", "results", "references"]:
                academic_score += matches * 3  # Higher weight for key sections
            elif pattern_name in ["equations", "citations"]:
                academic_score += matches * 2
            else:
                academic_score += matches
        scores[DocumentType.ACADEMIC_PAPER] = academic_score
        
        # Industry report scoring
        industry_score = 0
        for pattern_name, pattern in self.industry_patterns.items():
            matches = len(re.findall(pattern, content_lower, re.IGNORECASE))
            if pattern_name in ["market", "forecast", "competition"]:
                industry_score += matches * 3
            else:
                industry_score += matches
        scores[DocumentType.INDUSTRY_REPORT] = industry_score
        
        # Financial report scoring
        financial_score = 0
        for pattern_name, pattern in self.financial_patterns.items():
            matches = len(re.findall(pattern, content_lower, re.IGNORECASE))
            if pattern_name in ["financial_metrics", "financial_statements"]:
                financial_score += matches * 3
            elif pattern_name == "currency":
                financial_score += min(matches, 20) * 2  # Cap currency matches
            else:
                financial_score += matches
        scores[DocumentType.FINANCIAL_REPORT] = financial_score
        
        # Technical documentation scoring
        technical_score = 0
        for pattern_name, pattern in self.technical_patterns.items():
            matches = len(re.findall(pattern, content, re.MULTILINE))  # Don't ignore case for code
            if pattern_name in ["api", "code", "functions"]:
                technical_score += matches * 3
            else:
                technical_score += matches
        scores[DocumentType.TECHNICAL_DOCUMENTATION] = technical_score
        
        # Patent scoring
        patent_score = 0
        for pattern_name, pattern in self.patent_patterns.items():
            matches = len(re.findall(pattern, content_lower, re.IGNORECASE))
            if pattern_name in ["claims", "embodiment"]:
                patent_score += matches * 3
            else:
                patent_score += matches
        scores[DocumentType.PATENT] = patent_score
        
        # Find the document type with highest score
        max_score = max(scores.values())
        
        if max_score == 0:
            return DocumentType.GENERAL, 0.0
        
        # Get document type with highest score
        doc_type = max(scores, key=scores.get)
        
        # Calculate confidence (normalized score)
        total_score = sum(scores.values())
        confidence = scores[doc_type] / total_score if total_score > 0 else 0.0
        
        # Apply threshold
        if confidence < 0.3:  # Low confidence threshold
            return DocumentType.GENERAL, confidence
        
        return doc_type, confidence
    
    def classify_by_structure(self, content: str) -> Optional[DocumentType]:
        """
        Classify based on document structure.
        
        Args:
            content: Document text content
            
        Returns:
            DocumentType or None if structure doesn't match
        """
        lines = content.split('\n')
        
        # Check for academic paper structure
        has_abstract = any('abstract' in line.lower() for line in lines[:50])
        has_references = any('references' in line.lower() or 'bibliography' in line.lower() 
                           for line in lines[-100:])
        
        if has_abstract and has_references:
            return DocumentType.ACADEMIC_PAPER
        
        # Check for patent structure
        has_claims = any('claim' in line.lower() for line in lines)
        has_embodiment = any('embodiment' in line.lower() for line in lines)
        
        if has_claims and has_embodiment:
            return DocumentType.PATENT
        
        # Check for financial report structure
        has_financial_sections = any(
            term in content.lower() 
            for term in ['consolidated statements', 'financial position', 'comprehensive income']
        )
        
        if has_financial_sections:
            return DocumentType.FINANCIAL_REPORT
        
        return None
    
    def classify_by_filename(self, filename: str) -> Optional[DocumentType]:
        """
        Classify based on filename patterns.
        
        Args:
            filename: Name of the file
            
        Returns:
            DocumentType or None if no pattern matches
        """
        filename_lower = filename.lower()
        
        # Academic paper patterns
        if any(term in filename_lower for term in ['paper', 'article', 'journal', 'conference']):
            return DocumentType.ACADEMIC_PAPER
        
        # Financial report patterns
        if any(term in filename_lower for term in ['10-k', '10-q', 'annual_report', 'earnings']):
            return DocumentType.FINANCIAL_REPORT
        
        # Industry report patterns
        if any(term in filename_lower for term in ['market_report', 'industry_analysis', 'forecast']):
            return DocumentType.INDUSTRY_REPORT
        
        # Technical documentation patterns
        if any(term in filename_lower for term in ['api', 'documentation', 'manual', 'guide']):
            return DocumentType.TECHNICAL_DOCUMENTATION
        
        # Patent patterns
        if 'patent' in filename_lower or re.search(r'US\d+', filename):
            return DocumentType.PATENT
        
        return None
    
    async def classify_with_llm(self, content: str, context: Dict[str, Any]) -> Tuple[DocumentType, float]:
        """
        Classify document using LLM for higher accuracy.
        
        Args:
            content: Document text content (first 2000 chars)
            context: Additional context (filename, metadata)
            
        Returns:
            Tuple of (DocumentType, confidence_score)
        """
        # This would integrate with the LLM API
        # For now, returning placeholder
        logger.info("LLM classification not implemented yet")
        return self.classify_by_content(content)
    
    def classify(
        self,
        content: str,
        filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[DocumentType, float, DocumentSchema]:
        """
        Classify document and return appropriate schema.
        
        Args:
            content: Document text content
            filename: Optional filename for additional hints
            metadata: Optional metadata for classification
            
        Returns:
            Tuple of (DocumentType, confidence_score, DocumentSchema)
        """
        # Try filename classification first
        if filename:
            doc_type = self.classify_by_filename(filename)
            if doc_type:
                logger.info(f"Classified by filename as {doc_type}")
                return doc_type, 0.9, get_schema(doc_type)
        
        # Try structure-based classification
        struct_type = self.classify_by_structure(content)
        if struct_type:
            logger.info(f"Classified by structure as {struct_type}")
            # Verify with content classification
            content_type, confidence = self.classify_by_content(content)
            if content_type == struct_type:
                confidence = min(confidence + 0.2, 1.0)  # Boost confidence
            return struct_type, confidence, get_schema(struct_type)
        
        # Fall back to content-based classification
        doc_type, confidence = self.classify_by_content(content)
        logger.info(f"Classified by content as {doc_type} (confidence: {confidence:.2f})")
        
        return doc_type, confidence, get_schema(doc_type)
    
    def classify_chunks(
        self,
        chunks: List[Dict[str, Any]],
        filename: Optional[str] = None
    ) -> Tuple[DocumentType, float, DocumentSchema]:
        """
        Classify document based on chunks.
        
        Args:
            chunks: List of document chunks
            filename: Optional filename
            
        Returns:
            Tuple of (DocumentType, confidence_score, DocumentSchema)
        """
        # Combine first few chunks for classification
        sample_size = min(5, len(chunks))
        sample_content = " ".join(chunk.get("content", "") for chunk in chunks[:sample_size])
        
        return self.classify(sample_content, filename)


def quick_classify(content: str, filename: Optional[str] = None) -> DocumentType:
    """
    Quick classification helper function.
    
    Args:
        content: Document content
        filename: Optional filename
        
    Returns:
        DocumentType
    """
    classifier = DocumentClassifier()
    doc_type, _, _ = classifier.classify(content, filename)
    return doc_type