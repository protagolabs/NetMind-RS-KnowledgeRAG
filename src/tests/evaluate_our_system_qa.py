"""
Evaluate Our System's QA Results Using DeepEval Framework
==========================================================
This script evaluates the quality of our system's question-answering results
using multiple metrics including answer relevancy, faithfulness,
and contextual relevance.
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import statistics

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.evaluation.deep_eval import DeepEval
from dotenv import load_dotenv

load_dotenv()


class OurSystemQAEvaluator:
    """Evaluator for our Neo4j-based QA system results."""
    
    def __init__(self, results_file: str):
        """
        Initialize the evaluator.
        
        Args:
            results_file: Path to our system's results JSON file
        """
        self.results_file = results_file
        
    def prepare_context_from_references(
        self, 
        references: Dict[str, Any]
    ) -> List[str]:
        """
        Prepare context list from our system's references.
        
        Args:
            references: Dictionary containing entities, relationships, and chunks
            
        Returns:
            List of context strings for evaluation
        """
        context_list = []
        
        # Add entity descriptions as context
        for entity in references.get('entities', []):
            entity_context = f"{entity['entity']} ({entity['type']}): {entity['description']}"
            context_list.append(entity_context)
        
        # Add relationship descriptions as context
        for rel in references.get('relationships', []):
            if rel.get('description'):
                rel_context = f"{rel['source']} --[{rel['relationship']}]--> {rel['target']}: {rel['description']}"
            else:
                rel_context = f"{rel['source']} --[{rel['relationship']}]--> {rel['target']}"
            context_list.append(rel_context)
        
        # Add chunk contents as context
        for chunk in references.get('chunks', []):
            # Use content_preview which contains the actual text
            content = chunk.get('content_preview', '')
            if content:
                # Remove "..." if it's just a truncation marker
                if content.endswith('...'):
                    content = content[:-3]
                context_list.append(content)
        
        return context_list
    
    def evaluate_single_query(
        self,
        query_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate a single query result.
        
        Args:
            query_data: Dictionary containing query, answer, and references
            
        Returns:
            Dictionary of evaluation scores
        """
        question = query_data['query_text']
        generated_answer = query_data['answer']
        references = query_data.get('references', {})
        
        # Prepare context from references
        context_list = self.prepare_context_from_references(references)
        
        # Skip if no context or answer
        if not context_list or not generated_answer or query_data.get('success') is False:
            return {
                'answer_relevancy': 0.0,
                'faithfulness': 0.0,
                'contextual_relevance': 0.0,
                'contextual_precision': 0.0,
                'contextual_recall': 0.0,
                'skipped': True,
                'reason': 'No valid answer or context' if not context_list else 'Query failed'
            }
        
        # For our system, we don't have expected answers from the LightRAG file
        # We'll use the generated answer as both actual and expected for now
        # In a real scenario, you'd have ground truth answers
        
        # Create DeepEval instance
        evaluator = DeepEval(
            question=question,
            retrieval_context=context_list,
            actual_output=generated_answer,
            expected_output=generated_answer  # Using same as we don't have ground truth
        )
        
        # Run evaluation
        try:
            results = evaluator.evaluate()
            results['skipped'] = False
            return results
        except Exception as e:
            print(f"Evaluation error for {query_data['query_id']}: {e}")
            return {
                'answer_relevancy': 0.0,
                'faithfulness': 0.0,
                'contextual_relevance': 0.0,
                'contextual_precision': 0.0,
                'contextual_recall': 0.0,
                'skipped': True,
                'error': str(e)
            }
    
    def evaluate_results(
        self,
        output_file: str,
        max_items: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Evaluate all QA results from our system.
        
        Args:
            output_file: Path to save evaluation results
            max_items: Maximum items to evaluate
            
        Returns:
            Evaluation summary
        """
        # Load our system's results
        with open(self.results_file, 'r', encoding='utf-8') as f:
            system_results = json.load(f)
        
        queries = system_results['queries']
        if max_items:
            queries = queries[:max_items]
        
        evaluation_results = []
        
        print(f"Evaluating {len(queries)} query-answer pairs from our system...")
        print("=" * 80)
        
        for i, query_data in enumerate(queries, 1):
            query_id = query_data['query_id']
            query_text = query_data['query_text'][:80] + "..." if len(query_data['query_text']) > 80 else query_data['query_text']
            
            print(f"\nEvaluating Query {i}/{len(queries)}: {query_id}")
            print(f"  Query: {query_text}")
            
            # Evaluate this query
            scores = self.evaluate_single_query(query_data)
            
            # Add metadata
            evaluation_result = {
                'query_id': query_id,
                'query_type': query_data['query_type'],
                'focus': query_data['focus'],
                'query_text': query_data['query_text'],
                'answer_length': len(query_data['answer']),
                'context_tokens': query_data.get('context_tokens', 0),
                'response_time': query_data.get('response_time_seconds', 0),
                'scores': scores,
                'references_stats': {
                    'entities': query_data['references']['total_entities'],
                    'relationships': query_data['references']['total_relationships'],
                    'chunks': query_data['references']['total_chunks']
                }
            }
            
            evaluation_results.append(evaluation_result)
            
            # Print scores
            if not scores.get('skipped', False):
                print(f"  Answer Relevancy: {scores.get('answer_relevancy', 0):.3f}")
                print(f"  Faithfulness: {scores.get('faithfulness', 0):.3f}")
                print(f"  Contextual Relevance: {scores.get('contextual_relevance', 0):.3f}")
                print(f"  Contextual Precision: {scores.get('contextual_precision', 0):.3f}")
                print(f"  Contextual Recall: {scores.get('contextual_recall', 0):.3f}")
            else:
                reason = scores.get('reason', scores.get('error', 'Unknown'))
                print(f"  Skipped: {reason}")
        
        # Calculate summary statistics
        summary = self.calculate_summary(evaluation_results)
        
        # Save evaluation results
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'source_file': self.results_file,
            'system': system_results.get('system', 'NetMind-RS-KnowledgeRAG'),
            'database': system_results.get('database', 'testsampledoc2'),
            'total_evaluated': len(evaluation_results),
            'summary': summary,
            'detailed_results': evaluation_results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        self.print_summary(summary)
        print(f"\nDetailed results saved to: {output_file}")
        
        return summary
    
    def calculate_summary(self, evaluation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate summary statistics from evaluation results.
        
        Args:
            evaluation_results: List of evaluation results
            
        Returns:
            Summary statistics
        """
        # Filter out skipped results for metrics calculation
        valid_results = [r for r in evaluation_results if not r['scores'].get('skipped', False)]
        
        if not valid_results:
            return {
                'total_queries': len(evaluation_results),
                'valid_evaluations': 0,
                'skipped_evaluations': len(evaluation_results),
                'message': 'No valid evaluations completed'
            }
        
        # Calculate average scores
        metrics = ['answer_relevancy', 'faithfulness', 'contextual_relevance', 
                  'contextual_precision', 'contextual_recall']
        
        summary = {
            'total_queries': len(evaluation_results),
            'valid_evaluations': len(valid_results),
            'skipped_evaluations': len(evaluation_results) - len(valid_results),
            'average_scores': {},
            'score_distributions': {},
            'by_query_type': {},
            'performance_stats': {}
        }
        
        # Calculate averages
        for metric in metrics:
            scores = [r['scores'].get(metric, 0) for r in valid_results]
            if scores:
                summary['average_scores'][metric] = statistics.mean(scores)
                summary['score_distributions'][metric] = {
                    'min': min(scores),
                    'max': max(scores),
                    'median': statistics.median(scores),
                    'stdev': statistics.stdev(scores) if len(scores) > 1 else 0
                }
        
        # Group by query type
        query_types = {}
        for result in valid_results:
            q_type = result['query_type']
            if q_type not in query_types:
                query_types[q_type] = []
            query_types[q_type].append(result)
        
        for q_type, results in query_types.items():
            summary['by_query_type'][q_type] = {
                'count': len(results),
                'average_scores': {}
            }
            for metric in metrics:
                scores = [r['scores'].get(metric, 0) for r in results]
                if scores:
                    summary['by_query_type'][q_type]['average_scores'][metric] = statistics.mean(scores)
        
        # Calculate performance statistics
        response_times = [r['response_time'] for r in evaluation_results if r.get('response_time')]
        context_tokens = [r['context_tokens'] for r in evaluation_results if r.get('context_tokens')]
        
        if response_times:
            summary['performance_stats']['response_time'] = {
                'average': statistics.mean(response_times),
                'min': min(response_times),
                'max': max(response_times),
                'median': statistics.median(response_times)
            }
        
        if context_tokens:
            summary['performance_stats']['context_tokens'] = {
                'average': statistics.mean(context_tokens),
                'min': min(context_tokens),
                'max': max(context_tokens),
                'median': statistics.median(context_tokens)
            }
        
        # Average references used
        entity_counts = [r['references_stats']['entities'] for r in evaluation_results]
        rel_counts = [r['references_stats']['relationships'] for r in evaluation_results]
        chunk_counts = [r['references_stats']['chunks'] for r in evaluation_results]
        
        summary['performance_stats']['references'] = {
            'avg_entities': statistics.mean(entity_counts) if entity_counts else 0,
            'avg_relationships': statistics.mean(rel_counts) if rel_counts else 0,
            'avg_chunks': statistics.mean(chunk_counts) if chunk_counts else 0
        }
        
        return summary
    
    def print_summary(self, summary: Dict[str, Any]):
        """Print evaluation summary."""
        print(f"Total Queries: {summary['total_queries']}")
        print(f"Valid Evaluations: {summary['valid_evaluations']}")
        print(f"Skipped: {summary['skipped_evaluations']}")
        
        if summary['valid_evaluations'] > 0:
            print("\nAverage Scores:")
            for metric, score in summary['average_scores'].items():
                print(f"  {metric}: {score:.3f}")
            
            print("\nScore Distributions:")
            for metric, dist in summary['score_distributions'].items():
                print(f"  {metric}:")
                print(f"    Min: {dist['min']:.3f}, Max: {dist['max']:.3f}")
                print(f"    Median: {dist['median']:.3f}, StdDev: {dist['stdev']:.3f}")
            
            print("\nBy Query Type:")
            for q_type, data in summary['by_query_type'].items():
                print(f"  {q_type} ({data['count']} queries):")
                for metric, score in data['average_scores'].items():
                    print(f"    {metric}: {score:.3f}")
        
        if 'performance_stats' in summary:
            print("\nPerformance Statistics:")
            
            if 'response_time' in summary['performance_stats']:
                rt = summary['performance_stats']['response_time']
                print(f"  Response Time:")
                print(f"    Average: {rt['average']:.2f}s")
                print(f"    Min: {rt['min']:.2f}s, Max: {rt['max']:.2f}s")
                
            if 'context_tokens' in summary['performance_stats']:
                ct = summary['performance_stats']['context_tokens']
                print(f"  Context Tokens:")
                print(f"    Average: {ct['average']:.0f}")
                print(f"    Min: {ct['min']}, Max: {ct['max']}")
            
            if 'references' in summary['performance_stats']:
                refs = summary['performance_stats']['references']
                print(f"  Average References Used:")
                print(f"    Entities: {refs['avg_entities']:.1f}")
                print(f"    Relationships: {refs['avg_relationships']:.1f}")
                print(f"    Chunks: {refs['avg_chunks']:.1f}")


def main():
    """Main evaluation function."""
    
    # File paths
    results_file = "/home/administrator/projects/NetMind-RS-KnowledgeRAG/our_system_query_results.json"
    evaluation_output_file = "/home/administrator/projects/NetMind-RS-KnowledgeRAG/our_system_evaluation_results.json"
    
    # Check if results file exists
    if not Path(results_file).exists():
        print(f"Error: Results file not found: {results_file}")
        print("Please run test_queries_from_lightrag.py first to generate results.")
        return
    
    # Initialize evaluator
    evaluator = OurSystemQAEvaluator(results_file=results_file)
    
    # Evaluate the QA results
    print("\n" + "="*80)
    print("EVALUATING OUR SYSTEM'S QA RESULTS")
    print("="*80)
    print(f"Source: {Path(results_file).name}")
    print("="*80)
    
    summary = evaluator.evaluate_results(
        output_file=evaluation_output_file,
        max_items=None  # Evaluate all queries
    )
    
    print("\nEvaluation complete!")
    print(f"Results saved to: {evaluation_output_file}")


if __name__ == "__main__":
    # Run synchronously since we don't need async for evaluation
    main()