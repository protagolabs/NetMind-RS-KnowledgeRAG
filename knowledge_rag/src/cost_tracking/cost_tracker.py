"""
Main Cost Tracker

Orchestrates token counting, cost calculation, and storage for OpenAI API calls.
"""

import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import uuid

from .token_counter import TokenCounter
from .pricing_config import calculate_cost, get_cost_breakdown
from .cost_storage import CostStorage, ApiCallRecord
from .cost_reports import CostReporter


class CostTracker:
    """Main cost tracking coordinator"""
    
    def __init__(
        self,
        session_name: Optional[str] = None,
        db_path: str = "cost_data/api_costs.db",
        json_dir: str = "cost_data"
    ):
        self.logger = logging.getLogger(__name__)
        
        # Generate session ID
        self.session_id = str(uuid.uuid4())[:8]
        self.session_name = session_name or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_start = datetime.now()
        
        # Initialize components
        self.token_counter = TokenCounter()
        self.storage = CostStorage(db_path=db_path, json_dir=json_dir)
        self.reporter = CostReporter(self.storage)
        
        # Session tracking
        self.current_episode_id = ""
        self.current_document_id = ""
        self.current_operation = ""
        
        # Running totals for quick access
        self.session_cost = 0.0
        self.session_tokens = 0
        self.session_calls = 0
        self.session_errors = 0
        
        self.logger.info(f"Cost tracker initialized for session: {self.session_name} ({self.session_id})")
    
    def set_context(self, episode_id: str = "", document_id: str = "", operation: str = ""):
        """
        Set context for subsequent API calls
        
        Args:
            episode_id: ID of current episode being processed
            document_id: ID of current document being processed
            operation: Type of operation (e.g., 'entity_extraction')
        """
        self.current_episode_id = episode_id
        self.current_document_id = document_id
        self.current_operation = operation
        
        self.logger.debug(f"Context set: episode={episode_id}, doc={document_id}, op={operation}")
    
    def track_api_call(
        self,
        messages: List[Dict[str, str]],
        response_text: str,
        model_name: str,
        processing_time: float = 0.0,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Track a complete API call with cost calculation
        
        Args:
            messages: Input messages sent to API
            response_text: Response text from API
            model_name: OpenAI model used
            processing_time: Time taken for the API call
            success: Whether the call was successful
            error_message: Error message if call failed
            
        Returns:
            Dictionary with cost breakdown and tracking info
        """
        try:
            # Count tokens
            input_tokens = self.token_counter.count_tokens_in_messages(messages, model_name)
            output_tokens = self.token_counter.count_tokens_in_response(response_text, model_name) if response_text else 0
            
            # Calculate costs
            cost_breakdown = get_cost_breakdown(input_tokens, output_tokens, model_name)
            
            # Create record
            record = ApiCallRecord(
                session_id=self.session_id,
                model_name=model_name,
                operation_type=self.current_operation or "unknown",
                episode_id=self.current_episode_id,
                document_id=self.current_document_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                input_cost=cost_breakdown["input_cost"],
                output_cost=cost_breakdown["output_cost"],
                success=success,
                error_message=error_message,
                processing_time=processing_time
            )
            
            # Store record
            record_id = self.storage.store_api_call(record)
            
            # Update session totals
            if success:
                self.session_cost += cost_breakdown["total_cost"]
                self.session_tokens += cost_breakdown["total_tokens"]
            self.session_calls += 1
            if not success:
                self.session_errors += 1
            
            # Log the call
            self.logger.info(
                f"API call tracked: ${cost_breakdown['total_cost']:.4f} "
                f"({input_tokens}+{output_tokens} tokens) "
                f"[{model_name}] {'✓' if success else '✗'}"
            )
            
            return {
                "record_id": record_id,
                "cost_breakdown": cost_breakdown,
                "session_totals": {
                    "cost": self.session_cost,
                    "tokens": self.session_tokens,
                    "calls": self.session_calls,
                    "errors": self.session_errors
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to track API call: {e}")
            return {"error": str(e)}
    
    def estimate_cost_for_messages(self, messages: List[Dict[str, str]], model_name: str) -> Dict[str, Any]:
        """
        Estimate cost for messages before making API call
        
        Args:
            messages: Messages to estimate cost for
            model_name: OpenAI model name
            
        Returns:
            Dictionary with cost estimates
        """
        try:
            input_tokens = self.token_counter.count_tokens_in_messages(messages, model_name)
            
            # Estimate response tokens (rough guess based on input)
            estimated_response_tokens = min(input_tokens // 2, 500)  # Conservative estimate
            
            estimated_breakdown = get_cost_breakdown(input_tokens, estimated_response_tokens, model_name)
            
            return {
                "input_tokens": input_tokens,
                "estimated_response_tokens": estimated_response_tokens,
                "estimated_total_tokens": input_tokens + estimated_response_tokens,
                "estimated_cost": estimated_breakdown["total_cost"],
                "cost_breakdown": estimated_breakdown
            }
            
        except Exception as e:
            self.logger.error(f"Failed to estimate cost: {e}")
            return {"error": str(e)}
    
    def get_session_cost(self) -> float:
        """Get current session cost"""
        return self.session_cost
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics"""
        duration = (datetime.now() - self.session_start).total_seconds()
        
        return {
            "session_id": self.session_id,
            "session_name": self.session_name,
            "duration_seconds": duration,
            "total_cost": self.session_cost,
            "total_tokens": self.session_tokens,
            "total_calls": self.session_calls,
            "total_errors": self.session_errors,
            "success_rate": ((self.session_calls - self.session_errors) / self.session_calls * 100) if self.session_calls > 0 else 100,
            "avg_cost_per_call": self.session_cost / self.session_calls if self.session_calls > 0 else 0,
            "calls_per_minute": (self.session_calls / (duration / 60)) if duration > 0 else 0
        }
    
    def get_detailed_session_stats(self) -> Dict[str, Any]:
        """Get detailed session statistics from storage"""
        return self.storage.get_session_stats(self.session_id)
    
    def print_session_summary(self):
        """Print a formatted session summary"""
        stats = self.get_session_stats()
        
        print(f"\n💰 Cost Tracking Session Summary")
        print(f"{'='*50}")
        print(f"Session: {stats['session_name']} ({stats['session_id']})")
        print(f"Duration: {stats['duration_seconds']:.1f} seconds")
        print(f"")
        print(f"📊 Usage Statistics:")
        print(f"   Total API calls: {stats['total_calls']}")
        print(f"   Total tokens: {stats['total_tokens']:,}")
        print(f"   Success rate: {stats['success_rate']:.1f}%")
        print(f"   Calls per minute: {stats['calls_per_minute']:.1f}")
        print(f"")
        print(f"💵 Cost Breakdown:")
        print(f"   Total cost: ${stats['total_cost']:.4f}")
        print(f"   Average per call: ${stats['avg_cost_per_call']:.4f}")
        print(f"   Cost per token: ${stats['total_cost'] / stats['total_tokens']:.6f}" if stats['total_tokens'] > 0 else "   Cost per token: $0.000000")
        
        if stats['total_errors'] > 0:
            print(f"")
            print(f"⚠️  Errors: {stats['total_errors']} failed calls")
    
    def export_session_report(self, output_file: Optional[str] = None) -> str:
        """
        Export detailed session report
        
        Args:
            output_file: Output file path (optional)
            
        Returns:
            Path to exported report
        """
        return self.storage.export_session_to_json(self.session_id, output_file)
    
    def generate_session_report(self) -> Dict[str, Any]:
        """Generate comprehensive session report"""
        return self.reporter.generate_session_report(self.session_id)
    
    def check_cost_limit(self, limit: float) -> Dict[str, Any]:
        """
        Check if session cost has exceeded a limit
        
        Args:
            limit: Cost limit in USD
            
        Returns:
            Dictionary with limit check results
        """
        current_cost = self.get_session_cost()
        exceeded = current_cost >= limit
        
        result = {
            "limit": limit,
            "current_cost": current_cost,
            "exceeded": exceeded,
            "remaining": max(0, limit - current_cost),
            "percentage_used": (current_cost / limit * 100) if limit > 0 else 0
        }
        
        if exceeded:
            self.logger.warning(f"Cost limit exceeded: ${current_cost:.4f} >= ${limit:.4f}")
        elif current_cost >= limit * 0.8:  # 80% warning
            self.logger.warning(f"Cost limit warning: ${current_cost:.4f} (80% of ${limit:.4f})")
        
        return result
    
    def get_token_stats(self, text: str, model_name: str = "gpt-3.5-turbo") -> Dict[str, Any]:
        """Get token statistics for text"""
        return self.token_counter.get_token_stats(text, model_name)
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old cost tracking data"""
        self.storage.cleanup_old_records(days_to_keep)
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with summary"""
        if exc_type:
            self.logger.error(f"Session ended with error: {exc_val}")
        else:
            self.logger.info(f"Session completed successfully")
        
        self.print_session_summary()
        return False  # Don't suppress exceptions 