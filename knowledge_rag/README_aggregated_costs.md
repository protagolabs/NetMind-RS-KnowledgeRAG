# Aggregated Cost Tracking System

This document explains the enhanced cost tracking system with aggregated data capabilities for the Knowledge RAG project.

## 🎯 Overview

The new aggregated cost tracking system provides:

1. **📊 Centralized Cost Summary**: One JSON file with all cost data aggregated by day, month, year
2. **📁 Organized Structure**: Session files moved to a dedicated `sessions/` subfolder
3. **⚡ Fast Access**: Quick cost summaries without database queries
4. **📈 Trend Analysis**: Built-in daily, monthly, and yearly trend tracking
5. **💡 Smart Recommendations**: Automated cost optimization suggestions

## 📁 New File Structure

```
cost_data/
├── aggregated_costs.json          # 📊 All aggregated cost data
├── sessions/                      # 📝 Individual session details
│   ├── session_abc123_20240724_103929.json
│   ├── session_def456_20240724_104512.json
│   └── session_ghi789_20240724_105203.json
├── api_costs.db                   # 🗄️ SQLite database (unchanged)
└── backup_YYYYMMDD_HHMMSS/       # 💾 Migration backups
```

## 📊 Aggregated Data Structure

The `aggregated_costs.json` contains:

### Summary Section
```json
{
  "summary": {
    "total_cost": 1.2345,
    "total_tokens": 50000,
    "total_sessions": 25,
    "first_session_date": "2024-07-01",
    "last_updated": "2024-07-24T10:39:29.527637",
    "version": "1.0"
  }
}
```

### Daily Breakdown
```json
{
  "daily": {
    "2024-07-24": {
      "total_cost": 0.0532,
      "total_tokens": 32210,
      "total_sessions": 3,
      "by_operation": {
        "entity_extraction": {"calls": 6, "cost": 0.0234, "tokens": 12000},
        "relationship_extraction": {"calls": 4, "cost": 0.0156, "tokens": 8500}
      },
      "by_model": {
        "gpt-3.5-turbo": {"calls": 10, "cost": 0.039, "tokens": 30710}
      }
    }
  }
}
```

### Monthly & Yearly Rollups
```json
{
  "monthly": {
    "2024-07": {
      "total_cost": 1.2345,
      "total_tokens": 50000,
      "total_sessions": 25,
      "days_active": 12,
      "by_operation": {...},
      "by_model": {...}
    }
  },
  "yearly": {
    "2024": {
      "total_cost": 1.2345,
      "total_tokens": 50000,
      "total_sessions": 25,
      "months_active": 1,
      "by_operation": {...},
      "by_model": {...}
    }
  }
}
```

## 🚀 Migration Process

If you have existing cost data, run the migration script:

```bash
python migrate_cost_data.py
```

**What it does:**
1. ✅ **Backs up** your existing data
2. ✅ **Moves** `session_*.json` files to `cost_data/sessions/`
3. ✅ **Rebuilds** `aggregated_costs.json` from all historical data
4. ✅ **Validates** the migration was successful

**Migration is safe:**
- Creates automatic backups
- Validates all data before proceeding
- Can be cancelled at any time
- Shows detailed progress

## 📊 Enhanced Cost Analysis

The updated `cost_analysis.py` now provides:

### 1. Aggregated Summary
```bash
python cost_analysis.py
```

Shows:
- 📊 **Overall Statistics**: Total cost, tokens, sessions, active days
- 📈 **Recent Trends**: Last 7 days average, cost ranges
- 🔧 **Top Operations**: Most expensive operations by cost
- 🤖 **Model Usage**: Cost breakdown by AI model
- 📅 **Monthly History**: Recent months' spending

### 2. Interactive Analysis
Choose from:
1. **Analyze specific session** - Detailed breakdown of any session
2. **Cost trend analysis** - 14-day trend with insights
3. **Model comparison** - Efficiency across different AI models
4. **Operation efficiency** - Performance by operation type
5. **Export reports** - Generate comprehensive reports
6. **Rebuild aggregated costs** - Refresh aggregated data

### 3. Cost Recommendations
Automatic suggestions like:
- 💡 "Consider optimizing prompts to reduce token usage"
- 💡 "Operation 'entity_extraction' accounts for 45% of costs - consider optimizing"
- 💡 "High variance in daily costs - consider cost limits"

## 🔧 Programmatic Access

### Basic Usage
```python
from src.cost_tracking import CostStorage

storage = CostStorage()

# Get all aggregated data
aggregated = storage.get_aggregated_costs()
print(f"Total cost: ${aggregated['summary']['total_cost']:.4f}")

# Get daily costs for a date range
daily_costs = storage.get_daily_aggregated_costs(
    start_date="2024-07-01", 
    end_date="2024-07-31"
)

# Get monthly costs for a specific year
monthly_costs = storage.get_monthly_aggregated_costs(year=2024)
```

### Reporting
```python
from src.cost_tracking import CostReporter

reporter = CostReporter(storage)

# Generate comprehensive aggregated report
report = reporter.generate_aggregated_report()

# Print formatted summary
reporter.print_aggregated_summary()
```

### Auto-Update
The aggregated data automatically updates when:
- ✅ New sessions complete (via `CostTracker` context manager)
- ✅ Sessions are exported to JSON
- ✅ Manual rebuild is triggered

## 📈 Performance Benefits

### Before (Session-Only)
- 🐌 **Slow queries**: Had to process all individual session files
- 🔍 **Limited insights**: Basic session-level data only
- 📁 **Cluttered structure**: All files in one directory
- ⏳ **No trends**: Had to calculate trends on-demand

### After (Aggregated)
- ⚡ **Instant summaries**: Pre-calculated aggregations
- 📊 **Rich insights**: Daily/monthly/yearly trends
- 📁 **Clean structure**: Organized session files
- 🎯 **Smart recommendations**: Built-in cost optimization

## 🛠️ Advanced Features

### Rebuild Aggregated Costs
If your aggregated data gets out of sync:

```python
from src.cost_tracking import CostStorage

storage = CostStorage()
rebuild_stats = storage.rebuild_aggregated_costs()
print(f"Processed {rebuild_stats['sessions_processed']} sessions")
```

### Custom Date Ranges
```python
# Get costs for specific periods
daily_july = storage.get_daily_aggregated_costs(
    start_date="2024-07-01",
    end_date="2024-07-31"
)

monthly_2024 = storage.get_monthly_aggregated_costs(year=2024)
```

### Integration with Existing Code
The new system is **fully backward compatible**:
- ✅ Existing `CostTracker` usage works unchanged
- ✅ All existing session data is preserved
- ✅ Database structure remains the same
- ✅ JSON export functionality enhanced, not replaced

## 🎉 Benefits Summary

### For Developers
- ✅ **Faster development**: Quick cost insights without complex queries
- ✅ **Better debugging**: Detailed operation-level cost tracking
- ✅ **Trend awareness**: Understand cost patterns over time

### For Operations
- ✅ **Budget tracking**: Know exactly where money is spent
- ✅ **Cost optimization**: Built-in recommendations for savings
- ✅ **Trend monitoring**: Spot cost increases early

### For Analysis
- ✅ **Rich reporting**: Daily/monthly/yearly breakdowns
- ✅ **Model comparison**: Find the most cost-effective AI models
- ✅ **Operation efficiency**: Identify expensive operations

## 🚀 Getting Started

1. **For new installations**: The aggregated system is enabled by default
2. **For existing installations**: Run `python migrate_cost_data.py`
3. **Start using**: `python cost_analysis.py` for enhanced reports

The aggregated cost tracking system gives you professional-grade cost monitoring and optimization capabilities for your Knowledge RAG system! 🎯 