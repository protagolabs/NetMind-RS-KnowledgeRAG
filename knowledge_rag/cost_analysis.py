#!/usr/bin/env python3
"""
Cost Analysis Tool

Analyzes stored API cost data and provides detailed reports including
aggregated costs across all sessions.
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️  python-dotenv not installed, trying system environment variables")

from src.cost_tracking import CostStorage, CostReporter


def main():
    """Main cost analysis function"""
    
    print("💰 Knowledge RAG - Cost Analysis")
    print("=" * 40)
    
    # Initialize cost tracking components
    storage = CostStorage()
    reporter = CostReporter(storage)
    
    try:
        # Display aggregated cost summary first
        print("\n🎯 AGGREGATED COST SUMMARY")
        print("-" * 40)
        reporter.print_aggregated_summary()
        
        # Generate detailed aggregated report
        print("\n📊 DETAILED AGGREGATED ANALYSIS")
        print("-" * 40)
        aggregated_report = reporter.generate_aggregated_report()
        
        if "error" not in aggregated_report:
            print(f"📈 Trends Analysis:")
            trends = aggregated_report.get("trends", {})
            print(f"   • Average daily cost (last 7 days): ${trends.get('avg_daily_cost_last_7_days', 0):.4f}")
            
            if trends.get("most_expensive_day"):
                expensive_day = trends["most_expensive_day"]
                print(f"   • Most expensive day: {expensive_day[0]} (${expensive_day[1]['total_cost']:.4f})")
            
            if trends.get("least_expensive_day"):
                cheap_day = trends["least_expensive_day"]
                print(f"   • Least expensive day: {cheap_day[0]} (${cheap_day[1]['total_cost']:.4f})")
            
            print(f"\n💡 Cost Recommendations:")
            recommendations = aggregated_report.get("recommendations", [])
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        else:
            print(f"❌ Error generating aggregated report: {aggregated_report['error']}")
        
        # Show recent sessions
        print(f"\n📋 RECENT SESSIONS")
        print("-" * 25)
        recent_sessions = storage.get_recent_sessions(limit=5)
        
        if recent_sessions:
            print(f"{'Session ID':<12} {'Cost':<10} {'Calls':<6} {'Start Time':<20}")
            print("-" * 55)
            
            for session in recent_sessions:
                session_id = session['session_id'][:10] + "..."
                cost = f"${session['total_cost']:.4f}"
                calls = str(session['total_calls'])
                start_time = session['start_time'][:19] if session.get('start_time') else 'Unknown'
                
                print(f"{session_id:<12} {cost:<10} {calls:<6} {start_time:<20}")
        else:
            print("No recent sessions found")
        
        # Detailed session analysis (if requested)
        print(f"\n🔍 DETAILED SESSION ANALYSIS")
        print("-" * 35)
        
        print("Select an option:")
        print("  1. Analyze specific session")
        print("  2. Cost trend analysis")
        print("  3. Model comparison report")
        print("  4. Operation efficiency report")
        print("  5. Export comprehensive report")
        print("  6. Rebuild aggregated costs")
        print("  0. Exit")
        
        try:
            choice = input("\nEnter your choice (0-6): ").strip()
            
            if choice == "1":
                # Analyze specific session
                session_id = input("Enter session ID (or partial ID): ").strip()
                if session_id:
                    # Find matching session
                    matching_sessions = [s for s in recent_sessions if session_id.lower() in s['session_id'].lower()]
                    
                    if matching_sessions:
                        full_session_id = matching_sessions[0]['session_id']
                        print(f"\n📊 Analyzing session: {full_session_id}")
                        session_report = reporter.generate_session_report(full_session_id)
                        
                        if session_report and "error" not in session_report:
                            stats = session_report.get("statistics", {})
                            print(f"   Duration: {session_report.get('duration_minutes', 0):.1f} minutes")
                            print(f"   Total cost: ${stats.get('total_cost', 0):.4f}")
                            print(f"   Total tokens: {stats.get('total_tokens', 0):,}")
                            print(f"   Success rate: {stats.get('success_rate', 0):.1f}%")
                            
                            # Show operation breakdown
                            by_operation = stats.get("by_operation", {})
                            if by_operation:
                                print(f"\n   Operations breakdown:")
                                for op, op_stats in by_operation.items():
                                    print(f"     • {op}: ${op_stats['cost']:.4f} ({op_stats['calls']} calls)")
                        else:
                            print("❌ Session not found or error occurred")
                    else:
                        print("❌ No matching sessions found")
            
            elif choice == "2":
                # Cost trend analysis
                print(f"\n📈 Cost Trend Analysis (14 days)")
                trend_report = reporter.generate_cost_trend_report(days=14)
                
                if trend_report and "error" not in trend_report:
                    summary = trend_report.get("summary", {})
                    print(f"   Total cost: ${summary.get('total_cost', 0):.4f}")
                    print(f"   Average daily cost: ${summary.get('avg_daily_cost', 0):.4f}")
                    print(f"   Days with activity: {summary.get('days_with_activity', 0)}")
                    
                    trend = summary.get('trend_direction', 'stable')
                    print(f"   Trend: {trend.upper()}")
                    
                    if summary.get('peak_day'):
                        print(f"   Peak day: {summary['peak_day']} (${summary.get('peak_cost', 0):.4f})")
                else:
                    print("❌ Unable to generate trend report")
            
            elif choice == "3":
                # Model comparison
                print(f"\n🤖 Model Comparison Report")
                model_report = reporter.generate_model_comparison_report()
                
                if model_report and "error" not in model_report:
                    model_breakdown = model_report.get("model_breakdown", {})
                    print(f"   Models analyzed: {len(model_breakdown)}")
                    
                    for model, stats in model_breakdown.items():
                        print(f"   • {model}:")
                        print(f"     - Total cost: ${stats['total_cost']:.4f}")
                        print(f"     - Total calls: {stats['total_calls']}")
                        print(f"     - Avg cost per call: ${stats['avg_cost_per_call']:.4f}")
                else:
                    print("❌ Unable to generate model comparison")
            
            elif choice == "4":
                # Operation efficiency
                print(f"\n⚡ Operation Efficiency Report")
                efficiency_report = reporter.generate_operation_efficiency_report()
                
                if efficiency_report and "error" not in efficiency_report:
                    op_breakdown = efficiency_report.get("operation_breakdown", {})
                    print(f"   Operations analyzed: {len(op_breakdown)}")
                    
                    for operation, stats in op_breakdown.items():
                        print(f"   • {operation}:")
                        print(f"     - Total cost: ${stats['total_cost']:.4f}")
                        print(f"     - Total calls: {stats['total_calls']}")
                        print(f"     - Cost per call: ${stats['cost_per_call']:.4f}")
                        print(f"     - Avg processing time: {stats['avg_processing_time']:.2f}s")
                else:
                    print("❌ Unable to generate efficiency report")
            
            elif choice == "5":
                # Export comprehensive report
                print(f"\n📄 Exporting Comprehensive Report...")
                try:
                    report_path = reporter.export_comprehensive_report()
                    print(f"✅ Report exported to: {report_path}")
                except Exception as e:
                    print(f"❌ Export failed: {e}")
            
            elif choice == "6":
                # Rebuild aggregated costs
                print(f"\n🔄 Rebuilding Aggregated Costs...")
                try:
                    rebuild_stats = storage.rebuild_aggregated_costs()
                    print(f"✅ Aggregated costs rebuilt:")
                    print(f"   • Sessions processed: {rebuild_stats['sessions_processed']}/{rebuild_stats['total_sessions_found']}")
                    print(f"   • Total cost: ${rebuild_stats['total_cost']:.4f}")
                    print(f"   • Total tokens: {rebuild_stats['total_tokens']:,}")
                    
                    if rebuild_stats['sessions_failed'] > 0:
                        print(f"   ⚠️  Failed sessions: {rebuild_stats['sessions_failed']}")
                except Exception as e:
                    print(f"❌ Rebuild failed: {e}")
            
            elif choice == "0":
                print("👋 Goodbye!")
            
            else:
                print("❌ Invalid choice")
        
        except KeyboardInterrupt:
            print(f"\n👋 Analysis interrupted by user")
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
    
    except Exception as e:
        print(f"❌ Failed to initialize cost analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 