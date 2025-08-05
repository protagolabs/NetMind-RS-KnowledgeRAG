"""
Cost Reporting and Analytics

Generates various cost reports and analytics for API usage tracking.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict
import json
from pathlib import Path

from .cost_storage import CostStorage


class CostReporter:
    """Generates cost reports and analytics"""
    
    def __init__(self, storage: CostStorage):
        self.storage = storage
        self.logger = logging.getLogger(__name__)
    
    def generate_session_report(self, session_id: str) -> Dict[str, Any]:
        """
        Generate comprehensive report for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary with detailed session report
        """
        stats = self.storage.get_session_stats(session_id)
        if not stats:
            return {"error": f"No data found for session {session_id}"}
        
        # Calculate additional metrics
        total_cost = stats.get("total_cost", 0.0)
        total_calls = stats.get("total_calls", 0)
        success_rate = stats.get("success_rate", 0.0)
        
        # Cost efficiency metrics
        cost_per_token = total_cost / stats.get("total_tokens", 1)
        
        # Time analysis
        start_time = datetime.fromisoformat(stats.get("start_time", ""))
        end_time = datetime.fromisoformat(stats.get("end_time", ""))
        duration = (end_time - start_time).total_seconds()
        
        # Build comprehensive report
        report = {
            "session_id": session_id,
            "summary": {
                "total_cost": total_cost,
                "total_calls": total_calls,
                "total_tokens": stats.get("total_tokens", 0),
                "success_rate": success_rate,
                "duration_seconds": duration,
                "avg_cost_per_call": stats.get("avg_cost_per_call", 0.0),
                "cost_per_token": cost_per_token,
                "calls_per_minute": (total_calls / (duration / 60)) if duration > 0 else 0
            },
            "breakdown": {
                "by_operation": stats.get("by_operation", {}),
                "by_model": stats.get("by_model", {})
            },
            "performance": {
                "avg_processing_time": stats.get("avg_processing_time", 0.0),
                "success_rate": success_rate
            },
            "time_range": {
                "start_time": stats.get("start_time"),
                "end_time": stats.get("end_time"),
                "duration_minutes": duration / 60 if duration > 0 else 0
            }
        }
        
        # Add cost optimization recommendations
        report["recommendations"] = self._generate_recommendations(stats)
        
        return report
    
    def generate_daily_report(self, date: str) -> Dict[str, Any]:
        """
        Generate daily cost report
        
        Args:
            date: Date in YYYY-MM-DD format
            
        Returns:
            Dictionary with daily report
        """
        stats = self.storage.get_daily_stats(date)
        if not stats:
            return {"error": f"No data found for date {date}"}
        
        # Get recent sessions for the day
        recent_sessions = self.storage.get_recent_sessions(limit=50)
        daily_sessions = [
            session for session in recent_sessions
            if session.get("start_time", "").startswith(date)
        ]
        
        return {
            "date": date,
            "summary": stats,
            "sessions": daily_sessions,
            "session_count": len(daily_sessions)
        }
    
    def generate_cost_trend_report(self, days: int = 7) -> Dict[str, Any]:
        """
        Generate cost trend report for the last N days
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary with trend analysis
        """
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days-1)
        
        daily_stats = []
        total_cost = 0.0
        total_calls = 0
        
        for i in range(days):
            current_date = start_date + timedelta(days=i)
            date_str = current_date.strftime("%Y-%m-%d")
            
            day_stats = self.storage.get_daily_stats(date_str)
            if day_stats:
                daily_stats.append(day_stats)
                total_cost += day_stats.get("total_cost", 0.0)
                total_calls += day_stats.get("total_calls", 0)
            else:
                daily_stats.append({
                    "date": date_str,
                    "total_cost": 0.0,
                    "total_calls": 0,
                    "total_tokens": 0
                })
        
        # Calculate trends
        costs = [day.get("total_cost", 0.0) for day in daily_stats]
        calls = [day.get("total_calls", 0) for day in daily_stats]
        
        avg_daily_cost = total_cost / days if days > 0 else 0
        avg_daily_calls = total_calls / days if days > 0 else 0
        
        # Simple trend calculation (comparing first half vs second half)
        mid_point = days // 2
        first_half_avg = sum(costs[:mid_point]) / mid_point if mid_point > 0 else 0
        second_half_avg = sum(costs[mid_point:]) / (days - mid_point) if (days - mid_point) > 0 else 0
        
        cost_trend = "increasing" if second_half_avg > first_half_avg else "decreasing" if second_half_avg < first_half_avg else "stable"
        
        return {
            "period": {
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "days": days
            },
            "summary": {
                "total_cost": total_cost,
                "total_calls": total_calls,
                "avg_daily_cost": avg_daily_cost,
                "avg_daily_calls": avg_daily_calls,
                "cost_trend": cost_trend
            },
            "daily_breakdown": daily_stats,
            "analysis": {
                "highest_cost_day": max(daily_stats, key=lambda x: x.get("total_cost", 0)),
                "highest_usage_day": max(daily_stats, key=lambda x: x.get("total_calls", 0)),
                "trend_direction": cost_trend
            }
        }
    
    def generate_model_comparison_report(self, session_ids: List[str] = None) -> Dict[str, Any]:
        """
        Compare costs across different models
        
        Args:
            session_ids: List of session IDs to analyze (if None, analyze recent sessions)
            
        Returns:
            Dictionary with model comparison
        """
        if not session_ids:
            recent_sessions = self.storage.get_recent_sessions(limit=20)
            session_ids = [session["session_id"] for session in recent_sessions]
        
        model_stats = defaultdict(lambda: {
            "total_cost": 0.0,
            "total_calls": 0,
            "total_tokens": 0,
            "sessions": 0
        })
        
        for session_id in session_ids:
            session_stats = self.storage.get_session_stats(session_id)
            if not session_stats:
                continue
                
            for model, stats in session_stats.get("by_model", {}).items():
                model_stats[model]["total_cost"] += stats.get("cost", 0.0)
                model_stats[model]["total_calls"] += stats.get("calls", 0)
                model_stats[model]["total_tokens"] += stats.get("tokens", 0)
                model_stats[model]["sessions"] += 1
        
        # Calculate efficiency metrics
        for model, stats in model_stats.items():
            total_cost = stats["total_cost"]
            total_tokens = stats["total_tokens"]
            total_calls = stats["total_calls"]
            
            stats["cost_per_token"] = total_cost / total_tokens if total_tokens > 0 else 0
            stats["cost_per_call"] = total_cost / total_calls if total_calls > 0 else 0
            stats["tokens_per_call"] = total_tokens / total_calls if total_calls > 0 else 0
        
        # Sort by total cost
        sorted_models = sorted(
            model_stats.items(),
            key=lambda x: x[1]["total_cost"],
            reverse=True
        )
        
        return {
            "analysis_period": f"Last {len(session_ids)} sessions",
            "models_analyzed": len(model_stats),
            "model_breakdown": dict(sorted_models),
            "recommendations": self._generate_model_recommendations(dict(sorted_models))
        }
    
    def generate_operation_efficiency_report(self, session_ids: List[str] = None) -> Dict[str, Any]:
        """
        Analyze efficiency of different operations
        
        Args:
            session_ids: List of session IDs to analyze
            
        Returns:
            Dictionary with operation efficiency analysis
        """
        if not session_ids:
            recent_sessions = self.storage.get_recent_sessions(limit=20)
            session_ids = [session["session_id"] for session in recent_sessions]
        
        operation_stats = defaultdict(lambda: {
            "total_cost": 0.0,
            "total_calls": 0,
            "total_tokens": 0,
            "total_time": 0.0,
            "sessions": set()
        })
        
        for session_id in session_ids:
            session_stats = self.storage.get_session_stats(session_id)
            if not session_stats:
                continue
                
            for operation, stats in session_stats.get("by_operation", {}).items():
                operation_stats[operation]["total_cost"] += stats.get("cost", 0.0)
                operation_stats[operation]["total_calls"] += stats.get("calls", 0)
                operation_stats[operation]["total_tokens"] += stats.get("tokens", 0)
                operation_stats[operation]["sessions"].add(session_id)
        
        # Calculate efficiency metrics
        for operation, stats in operation_stats.items():
            stats["sessions"] = len(stats["sessions"])  # Convert set to count
            total_cost = stats["total_cost"]
            total_calls = stats["total_calls"]
            total_tokens = stats["total_tokens"]
            
            stats["cost_per_call"] = total_cost / total_calls if total_calls > 0 else 0
            stats["cost_per_token"] = total_cost / total_tokens if total_tokens > 0 else 0
            stats["tokens_per_call"] = total_tokens / total_calls if total_calls > 0 else 0
        
        # Sort by cost per call (efficiency)
        sorted_operations = sorted(
            operation_stats.items(),
            key=lambda x: x[1]["cost_per_call"],
            reverse=True
        )
        
        return {
            "analysis_period": f"Last {len(session_ids)} sessions",
            "operations_analyzed": len(operation_stats),
            "operation_breakdown": dict(sorted_operations),
            "efficiency_insights": self._generate_efficiency_insights(dict(sorted_operations))
        }
    
    def generate_aggregated_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive aggregated cost report
        
        Returns:
            Dictionary with aggregated cost analysis
        """
        try:
            aggregated_data = self.storage.get_aggregated_costs()
            
            if not aggregated_data or not aggregated_data.get("summary"):
                return {"error": "No aggregated cost data available"}
            
            summary = aggregated_data["summary"]
            daily_data = aggregated_data.get("daily", {})
            monthly_data = aggregated_data.get("monthly", {})
            
            # Calculate trends
            sorted_dates = sorted(daily_data.keys())
            recent_dates = sorted_dates[-7:] if len(sorted_dates) >= 7 else sorted_dates
            
            recent_daily_costs = [daily_data[date]["total_cost"] for date in recent_dates]
            avg_daily_cost = sum(recent_daily_costs) / len(recent_daily_costs) if recent_daily_costs else 0
            
            # Calculate monthly trends
            sorted_months = sorted(monthly_data.keys())
            recent_months = sorted_months[-3:] if len(sorted_months) >= 3 else sorted_months
            recent_monthly_costs = [monthly_data[month]["total_cost"] for month in recent_months]
            
            # Most/least expensive operations
            op_breakdown = aggregated_data.get("by_operation", {})
            sorted_operations = sorted(op_breakdown.items(), key=lambda x: x[1]["cost"], reverse=True)
            
            # Most/least expensive models
            model_breakdown = aggregated_data.get("by_model", {})
            sorted_models = sorted(model_breakdown.items(), key=lambda x: x[1]["cost"], reverse=True)
            
            report = {
                "report_type": "aggregated_costs",
                "generated_at": datetime.now().isoformat(),
                "summary": {
                    "total_cost": summary["total_cost"],
                    "total_tokens": summary["total_tokens"],
                    "total_sessions": summary["total_sessions"],
                    "first_session_date": summary.get("first_session_date"),
                    "cost_per_token": summary["total_cost"] / summary["total_tokens"] if summary["total_tokens"] > 0 else 0,
                    "avg_cost_per_session": summary["total_cost"] / summary["total_sessions"] if summary["total_sessions"] > 0 else 0,
                    "days_active": len(daily_data),
                    "months_active": len(monthly_data)
                },
                "trends": {
                    "avg_daily_cost_last_7_days": avg_daily_cost,
                    "daily_costs_trend": recent_daily_costs,
                    "monthly_costs_trend": recent_monthly_costs,
                    "most_expensive_day": max(daily_data.items(), key=lambda x: x[1]["total_cost"]) if daily_data else None,
                    "least_expensive_day": min(daily_data.items(), key=lambda x: x[1]["total_cost"]) if daily_data else None
                },
                "breakdowns": {
                    "by_operation": {
                        "most_expensive": sorted_operations[:5],
                        "breakdown": op_breakdown
                    },
                    "by_model": {
                        "most_expensive": sorted_models[:5],
                        "breakdown": model_breakdown
                    }
                },
                "recommendations": self._generate_cost_recommendations(aggregated_data)
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate aggregated report: {e}")
            return {"error": str(e)}
    
    def _generate_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate cost optimization recommendations"""
        recommendations = []
        
        total_cost = stats.get("total_cost", 0.0)
        success_rate = stats.get("success_rate", 100.0)
        
        if success_rate < 90:
            recommendations.append(
                f"Low success rate ({success_rate:.1f}%). Review error handling to reduce failed API calls."
            )
        
        if total_cost > 10.0:
            recommendations.append(
                "High session cost. Consider using cheaper models for simpler operations."
            )
        
        # Check model usage
        by_model = stats.get("by_model", {})
        if "gpt-4" in by_model and "gpt-3.5-turbo" in by_model:
            gpt4_cost = by_model["gpt-4"].get("cost", 0)
            gpt35_cost = by_model["gpt-3.5-turbo"].get("cost", 0)
            
            if gpt4_cost > gpt35_cost * 2:
                recommendations.append(
                    "GPT-4 usage is significantly higher than GPT-3.5. Consider using GPT-3.5 for simpler tasks."
                )
        
        return recommendations
    
    def _generate_model_recommendations(self, model_stats: Dict[str, Any]) -> List[str]:
        """Generate model-specific recommendations"""
        recommendations = []
        
        if not model_stats:
            return recommendations
        
        # Find most cost-effective model
        cost_per_token = {
            model: stats["cost_per_token"]
            for model, stats in model_stats.items()
            if stats["cost_per_token"] > 0
        }
        
        if cost_per_token:
            most_efficient = min(cost_per_token.items(), key=lambda x: x[1])
            recommendations.append(
                f"Most cost-effective model: {most_efficient[0]} at ${most_efficient[1]:.6f} per token"
            )
        
        return recommendations
    
    def _generate_efficiency_insights(self, operation_stats: Dict[str, Any]) -> List[str]:
        """Generate operation efficiency insights"""
        insights = []
        
        if not operation_stats:
            return insights
        
        # Find most expensive operation
        most_expensive = max(
            operation_stats.items(),
            key=lambda x: x[1]["cost_per_call"],
            default=("none", {})
        )
        
        if most_expensive[0] != "none":
            insights.append(
                f"Most expensive operation: {most_expensive[0]} at ${most_expensive[1]['cost_per_call']:.4f} per call"
            )
        
        # Find most efficient operation
        least_expensive = min(
            operation_stats.items(),
            key=lambda x: x[1]["cost_per_call"],
            default=("none", {})
        )
        
        if least_expensive[0] != "none":
            insights.append(
                f"Most efficient operation: {least_expensive[0]} at ${least_expensive[1]['cost_per_call']:.4f} per call"
            )
        
        return insights 
    
    def _generate_cost_recommendations(self, aggregated_data: Dict[str, Any]) -> List[str]:
        """Generate cost optimization recommendations based on aggregated data"""
        recommendations = []
        
        try:
            summary = aggregated_data.get("summary", {})
            by_operation = aggregated_data.get("by_operation", {})
            by_model = aggregated_data.get("by_model", {})
            
            total_cost = summary.get("total_cost", 0)
            total_tokens = summary.get("total_tokens", 0)
            
            # Token efficiency recommendations
            if total_tokens > 0:
                cost_per_token = total_cost / total_tokens
                if cost_per_token > 0.00005:  # High cost per token threshold
                    recommendations.append("Consider optimizing prompts to reduce token usage - current cost per token is above average")
            
            # Operation-specific recommendations
            if by_operation:
                most_expensive_op = max(by_operation.items(), key=lambda x: x[1]["cost"])
                op_name, op_stats = most_expensive_op
                
                if op_stats["cost"] > total_cost * 0.4:  # If one operation is >40% of total cost
                    recommendations.append(f"Operation '{op_name}' accounts for {(op_stats['cost']/total_cost*100):.1f}% of total costs - consider optimizing")
                
                # Check for inefficient operations (high cost per call)
                for op_name, op_stats in by_operation.items():
                    if op_stats["calls"] > 0:
                        cost_per_call = op_stats["cost"] / op_stats["calls"]
                        if cost_per_call > 0.01:  # High cost per call threshold
                            recommendations.append(f"Operation '{op_name}' has high cost per call (${cost_per_call:.4f}) - consider batch processing")
            
            # Model usage recommendations
            if by_model:
                model_costs = [(model, stats["cost"]) for model, stats in by_model.items()]
                if len(model_costs) > 1:
                    most_expensive_model = max(model_costs, key=lambda x: x[1])
                    if most_expensive_model[1] > total_cost * 0.8:  # If one model is >80% of cost
                        recommendations.append(f"Model '{most_expensive_model[0]}' dominates costs - consider using cheaper alternatives for simpler tasks")
            
            # Usage pattern recommendations
            daily_data = aggregated_data.get("daily", {})
            if len(daily_data) > 7:
                recent_days = sorted(daily_data.keys())[-7:]
                daily_costs = [daily_data[day]["total_cost"] for day in recent_days]
                avg_daily = sum(daily_costs) / len(daily_costs)
                max_daily = max(daily_costs)
                
                if max_daily > avg_daily * 3:  # High variance in daily costs
                    recommendations.append("High variance in daily costs detected - consider implementing cost limits or monitoring")
            
            if not recommendations:
                recommendations.append("Cost usage appears optimized - continue monitoring for trends")
                
        except Exception as e:
            recommendations.append(f"Unable to generate recommendations: {e}")
        
        return recommendations
    
    def print_aggregated_summary(self):
        """Print a formatted aggregated cost summary"""
        try:
            aggregated_data = self.storage.get_aggregated_costs()
            
            if not aggregated_data or not aggregated_data.get("summary"):
                print("❌ No aggregated cost data available")
                return
            
            summary = aggregated_data["summary"]
            daily_data = aggregated_data.get("daily", {})
            monthly_data = aggregated_data.get("monthly", {})
            by_operation = aggregated_data.get("by_operation", {})
            by_model = aggregated_data.get("by_model", {})
            
            print(f"\n💰 Aggregated Cost Summary")
            print(f"{'='*50}")
            print(f"Last Updated: {summary.get('last_updated', 'Unknown')}")
            print(f"Data Version: {summary.get('version', '1.0')}")
            print(f"")
            
            print(f"📊 Overall Statistics:")
            print(f"   Total Cost: ${summary['total_cost']:.4f}")
            print(f"   Total Tokens: {summary['total_tokens']:,}")
            print(f"   Total Sessions: {summary['total_sessions']}")
            print(f"   Days Active: {len(daily_data)}")
            print(f"   Months Active: {len(monthly_data)}")
            print(f"   First Session: {summary.get('first_session_date', 'Unknown')}")
            print(f"")
            
            if summary['total_tokens'] > 0:
                print(f"💵 Cost Efficiency:")
                print(f"   Cost per Token: ${summary['total_cost'] / summary['total_tokens']:.6f}")
            if summary['total_sessions'] > 0:
                print(f"   Cost per Session: ${summary['total_cost'] / summary['total_sessions']:.4f}")
            print(f"")
            
            # Recent activity
            if daily_data:
                sorted_dates = sorted(daily_data.keys())
                recent_dates = sorted_dates[-7:] if len(sorted_dates) >= 7 else sorted_dates
                recent_costs = [daily_data[date]["total_cost"] for date in recent_dates]
                avg_recent = sum(recent_costs) / len(recent_costs)
                
                print(f"📈 Recent Activity (Last {len(recent_dates)} days):")
                print(f"   Average Daily Cost: ${avg_recent:.4f}")
                print(f"   Cost Range: ${min(recent_costs):.4f} - ${max(recent_costs):.4f}")
                print(f"")
            
            # Top operations
            if by_operation:
                print(f"🔧 Top Operations by Cost:")
                sorted_ops = sorted(by_operation.items(), key=lambda x: x[1]["cost"], reverse=True)
                for i, (op_name, op_stats) in enumerate(sorted_ops[:5], 1):
                    percentage = (op_stats["cost"] / summary["total_cost"] * 100) if summary["total_cost"] > 0 else 0
                    print(f"   {i}. {op_name}: ${op_stats['cost']:.4f} ({percentage:.1f}%) - {op_stats['calls']} calls")
                print(f"")
            
            # Top models
            if by_model:
                print(f"🤖 Models by Cost:")
                sorted_models = sorted(by_model.items(), key=lambda x: x[1]["cost"], reverse=True)
                for model_name, model_stats in sorted_models:
                    percentage = (model_stats["cost"] / summary["total_cost"] * 100) if summary["total_cost"] > 0 else 0
                    print(f"   • {model_name}: ${model_stats['cost']:.4f} ({percentage:.1f}%) - {model_stats['calls']} calls")
                print(f"")
            
            # Monthly breakdown
            if monthly_data:
                print(f"📅 Monthly Breakdown:")
                sorted_months = sorted(monthly_data.keys(), reverse=True)
                for month in sorted_months[:6]:  # Show last 6 months
                    month_stats = monthly_data[month]
                    print(f"   {month}: ${month_stats['total_cost']:.4f} ({month_stats['total_sessions']} sessions, {month_stats['days_active']} days)")
                print(f"")
            
        except Exception as e:
            print(f"❌ Failed to print aggregated summary: {e}")
    
    def export_comprehensive_report(self, output_dir: str = "cost_reports") -> str:
        """
        Export comprehensive cost report to files
        
        Args:
            output_dir: Directory to save reports
            
        Returns:
            Path to the main report file
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate all reports
        reports = {
            "trend_report": self.generate_cost_trend_report(days=14),
            "model_comparison": self.generate_model_comparison_report(),
            "operation_efficiency": self.generate_operation_efficiency_report(),
            "recent_sessions": self.storage.get_recent_sessions(limit=10)
        }
        
        # Save individual reports
        for report_name, report_data in reports.items():
            file_path = output_path / f"{report_name}_{timestamp}.json"
            with open(file_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
        
        # Create comprehensive summary
        summary_report = {
            "generated_at": datetime.now().isoformat(),
            "reports_included": list(reports.keys()),
            "summary": {
                "trend_analysis": reports["trend_report"].get("summary", {}),
                "top_model": max(
                    reports["model_comparison"].get("model_breakdown", {}).items(),
                    key=lambda x: x[1]["total_cost"],
                    default=("none", {})
                )[0],
                "most_expensive_operation": max(
                    reports["operation_efficiency"].get("operation_breakdown", {}).items(),
                    key=lambda x: x[1]["cost_per_call"],
                    default=("none", {})
                )[0]
            }
        }
        
        main_report_path = output_path / f"comprehensive_cost_report_{timestamp}.json"
        with open(main_report_path, 'w') as f:
            json.dump({
                "summary": summary_report,
                "detailed_reports": reports
            }, f, indent=2, default=str)
        
        return str(main_report_path) 