"""
Cost Storage System

Handles persistence of API cost data to SQLite database and JSON files.
Includes aggregated cost tracking across all sessions.
"""

import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class ApiCallRecord:
    """Record of a single API call for cost tracking"""
    id: Optional[int] = None
    timestamp: str = ""
    session_id: str = ""
    model_name: str = ""
    operation_type: str = ""  # 'entity_extraction', 'relationship_extraction', etc.
    episode_id: str = ""
    document_id: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    processing_time: float = 0.0
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.total_tokens:
            self.total_tokens = self.input_tokens + self.output_tokens
        if not self.total_cost:
            self.total_cost = self.input_cost + self.output_cost


class CostStorage:
    """Handles persistence of cost tracking data"""
    
    def __init__(self, db_path: str = "cost_data/api_costs.db", json_dir: str = "cost_data"):
        self.db_path = Path(db_path)
        self.json_dir = Path(json_dir)
        self.sessions_dir = self.json_dir / "sessions"  # Subfolder for session files
        self.aggregated_file = self.json_dir / "aggregated_costs.json"  # Aggregated costs file
        self.logger = logging.getLogger(__name__)
        
        # Create directories if they don't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.json_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database and aggregated tracking
        self._init_database()
        self._init_aggregated_file()
    
    def _init_database(self):
        """Initialize the SQLite database with required tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create api_calls table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS api_calls (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        model_name TEXT NOT NULL,
                        operation_type TEXT NOT NULL,
                        episode_id TEXT,
                        document_id TEXT,
                        input_tokens INTEGER NOT NULL,
                        output_tokens INTEGER NOT NULL,
                        total_tokens INTEGER NOT NULL,
                        input_cost REAL NOT NULL,
                        output_cost REAL NOT NULL,
                        total_cost REAL NOT NULL,
                        success BOOLEAN NOT NULL,
                        error_message TEXT,
                        processing_time REAL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create sessions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        session_name TEXT,
                        start_time TEXT NOT NULL,
                        end_time TEXT,
                        total_calls INTEGER DEFAULT 0,
                        total_cost REAL DEFAULT 0.0,
                        total_tokens INTEGER DEFAULT 0,
                        success_rate REAL DEFAULT 0.0,
                        metadata TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes for better performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_id ON api_calls(session_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON api_calls(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_operation_type ON api_calls(operation_type)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_name ON api_calls(model_name)")
                
                conn.commit()
                self.logger.debug("Database initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _init_aggregated_file(self):
        """Initialize the aggregated costs JSON file if it doesn't exist"""
        if not self.aggregated_file.exists():
            initial_data = {
                "summary": {
                    "total_cost": 0.0,
                    "total_tokens": 0,
                    "total_sessions": 0,
                    "first_session_date": None,
                    "last_updated": datetime.now().isoformat(),
                    "version": "1.0"
                },
                "daily": {},
                "monthly": {},
                "yearly": {},
                "by_operation": {},
                "by_model": {}
            }
            
            with open(self.aggregated_file, 'w') as f:
                json.dump(initial_data, f, indent=2)
            
            self.logger.info(f"Initialized aggregated costs file: {self.aggregated_file}")
    
    def _load_aggregated_data(self) -> Dict[str, Any]:
        """Load the current aggregated data"""
        try:
            with open(self.aggregated_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load aggregated data: {e}")
            return self._get_empty_aggregated_data()
    
    def _get_empty_aggregated_data(self) -> Dict[str, Any]:
        """Get empty aggregated data structure"""
        return {
            "summary": {
                "total_cost": 0.0,
                "total_tokens": 0,
                "total_sessions": 0,
                "first_session_date": None,
                "last_updated": datetime.now().isoformat(),
                "version": "1.0"
            },
            "daily": {},
            "monthly": {},
            "yearly": {},
            "by_operation": {},
            "by_model": {}
        }
    
    def _save_aggregated_data(self, data: Dict[str, Any]):
        """Save aggregated data to file"""
        try:
            data["summary"]["last_updated"] = datetime.now().isoformat()
            with open(self.aggregated_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save aggregated data: {e}")
            raise
    
    def _update_aggregated_data(self, session_stats: Dict[str, Any]):
        """Update aggregated data with new session statistics"""
        try:
            data = self._load_aggregated_data()
            
            # Parse session start time to get date
            start_time = datetime.fromisoformat(session_stats["start_time"])
            session_date = start_time.date().isoformat()
            month_key = start_time.strftime("%Y-%m")
            year_key = start_time.strftime("%Y")
            
            # Update summary
            summary = data["summary"]
            summary["total_cost"] += session_stats["total_cost"]
            summary["total_tokens"] += session_stats["total_tokens"]
            summary["total_sessions"] += 1
            
            if summary["first_session_date"] is None or session_date < summary["first_session_date"]:
                summary["first_session_date"] = session_date
            
            # Update daily data
            if session_date not in data["daily"]:
                data["daily"][session_date] = {
                    "total_cost": 0.0,
                    "total_tokens": 0,
                    "total_sessions": 0,
                    "by_operation": {},
                    "by_model": {}
                }
            
            daily = data["daily"][session_date]
            daily["total_cost"] += session_stats["total_cost"]
            daily["total_tokens"] += session_stats["total_tokens"]
            daily["total_sessions"] += 1
            
            # Update daily operation breakdown
            for op, op_stats in session_stats.get("by_operation", {}).items():
                if op not in daily["by_operation"]:
                    daily["by_operation"][op] = {"calls": 0, "cost": 0.0, "tokens": 0}
                daily["by_operation"][op]["calls"] += op_stats["calls"]
                daily["by_operation"][op]["cost"] += op_stats["cost"]
                daily["by_operation"][op]["tokens"] += op_stats["tokens"]
            
            # Update daily model breakdown
            for model, model_stats in session_stats.get("by_model", {}).items():
                if model not in daily["by_model"]:
                    daily["by_model"][model] = {"calls": 0, "cost": 0.0, "tokens": 0}
                daily["by_model"][model]["calls"] += model_stats["calls"]
                daily["by_model"][model]["cost"] += model_stats["cost"]
                daily["by_model"][model]["tokens"] += model_stats["tokens"]
            
            # Update monthly data
            if month_key not in data["monthly"]:
                data["monthly"][month_key] = {
                    "total_cost": 0.0,
                    "total_tokens": 0,
                    "total_sessions": 0,
                    "days_active": 0,
                    "by_operation": {},
                    "by_model": {}
                }
            
            monthly = data["monthly"][month_key]
            monthly["total_cost"] += session_stats["total_cost"]
            monthly["total_tokens"] += session_stats["total_tokens"]
            monthly["total_sessions"] += 1
            
            # Count unique days in the month
            month_dates = {day for day in data["daily"].keys() if day.startswith(month_key)}
            monthly["days_active"] = len(month_dates)
            
            # Update monthly operation breakdown
            for op, op_stats in session_stats.get("by_operation", {}).items():
                if op not in monthly["by_operation"]:
                    monthly["by_operation"][op] = {"calls": 0, "cost": 0.0, "tokens": 0}
                monthly["by_operation"][op]["calls"] += op_stats["calls"]
                monthly["by_operation"][op]["cost"] += op_stats["cost"]
                monthly["by_operation"][op]["tokens"] += op_stats["tokens"]
            
            # Update monthly model breakdown
            for model, model_stats in session_stats.get("by_model", {}).items():
                if model not in monthly["by_model"]:
                    monthly["by_model"][model] = {"calls": 0, "cost": 0.0, "tokens": 0}
                monthly["by_model"][model]["calls"] += model_stats["calls"]
                monthly["by_model"][model]["cost"] += model_stats["cost"]
                monthly["by_model"][model]["tokens"] += model_stats["tokens"]
            
            # Update yearly data
            if year_key not in data["yearly"]:
                data["yearly"][year_key] = {
                    "total_cost": 0.0,
                    "total_tokens": 0,
                    "total_sessions": 0,
                    "months_active": 0,
                    "by_operation": {},
                    "by_model": {}
                }
            
            yearly = data["yearly"][year_key]
            yearly["total_cost"] += session_stats["total_cost"]
            yearly["total_tokens"] += session_stats["total_tokens"]
            yearly["total_sessions"] += 1
            
            # Count unique months in the year
            year_months = {month for month in data["monthly"].keys() if month.startswith(year_key)}
            yearly["months_active"] = len(year_months)
            
            # Update overall operation breakdown
            for op, op_stats in session_stats.get("by_operation", {}).items():
                if op not in data["by_operation"]:
                    data["by_operation"][op] = {"calls": 0, "cost": 0.0, "tokens": 0}
                data["by_operation"][op]["calls"] += op_stats["calls"]
                data["by_operation"][op]["cost"] += op_stats["cost"]
                data["by_operation"][op]["tokens"] += op_stats["tokens"]
            
            # Update overall model breakdown
            for model, model_stats in session_stats.get("by_model", {}).items():
                if model not in data["by_model"]:
                    data["by_model"][model] = {"calls": 0, "cost": 0.0, "tokens": 0}
                data["by_model"][model]["calls"] += model_stats["calls"]
                data["by_model"][model]["cost"] += model_stats["cost"]
                data["by_model"][model]["tokens"] += model_stats["tokens"]
            
            # Save updated data
            self._save_aggregated_data(data)
            self.logger.info(f"Updated aggregated costs with session {session_stats['session_id']}")
            
        except Exception as e:
            self.logger.error(f"Failed to update aggregated data: {e}")
            raise
    
    def store_api_call(self, record: ApiCallRecord) -> int:
        """
        Store an API call record
        
        Args:
            record: ApiCallRecord to store
            
        Returns:
            ID of the stored record
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO api_calls (
                        timestamp, session_id, model_name, operation_type,
                        episode_id, document_id, input_tokens, output_tokens,
                        total_tokens, input_cost, output_cost, total_cost,
                        success, error_message, processing_time
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.timestamp, record.session_id, record.model_name,
                    record.operation_type, record.episode_id, record.document_id,
                    record.input_tokens, record.output_tokens, record.total_tokens,
                    record.input_cost, record.output_cost, record.total_cost,
                    record.success, record.error_message, record.processing_time
                ))
                
                record_id = cursor.lastrowid
                conn.commit()
                
                self.logger.debug(f"Stored API call record with ID {record_id}")
                return record_id
                
        except Exception as e:
            self.logger.error(f"Failed to store API call record: {e}")
            raise
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Get statistics for a specific session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary with session statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get overall session stats
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_calls,
                        SUM(total_cost) as total_cost,
                        SUM(total_tokens) as total_tokens,
                        SUM(input_tokens) as input_tokens,
                        SUM(output_tokens) as output_tokens,
                        AVG(total_cost) as avg_cost_per_call,
                        AVG(processing_time) as avg_processing_time,
                        SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate,
                        MIN(timestamp) as start_time,
                        MAX(timestamp) as end_time
                    FROM api_calls 
                    WHERE session_id = ?
                """, (session_id,))
                
                row = cursor.fetchone()
                if not row:
                    return {}
                
                # Get breakdown by operation type
                cursor.execute("""
                    SELECT 
                        operation_type,
                        COUNT(*) as calls,
                        SUM(total_cost) as cost,
                        SUM(total_tokens) as tokens
                    FROM api_calls 
                    WHERE session_id = ?
                    GROUP BY operation_type
                """, (session_id,))
                
                operation_breakdown = {}
                for op_row in cursor.fetchall():
                    operation_breakdown[op_row[0]] = {
                        "calls": op_row[1],
                        "cost": op_row[2] or 0.0,
                        "tokens": op_row[3] or 0
                    }
                
                # Get breakdown by model
                cursor.execute("""
                    SELECT 
                        model_name,
                        COUNT(*) as calls,
                        SUM(total_cost) as cost,
                        SUM(total_tokens) as tokens
                    FROM api_calls 
                    WHERE session_id = ?
                    GROUP BY model_name
                """, (session_id,))
                
                model_breakdown = {}
                for model_row in cursor.fetchall():
                    model_breakdown[model_row[0]] = {
                        "calls": model_row[1],
                        "cost": model_row[2] or 0.0,
                        "tokens": model_row[3] or 0
                    }
                
                return {
                    "session_id": session_id,
                    "total_calls": row[0] or 0,
                    "total_cost": row[1] or 0.0,
                    "total_tokens": row[2] or 0,
                    "input_tokens": row[3] or 0,
                    "output_tokens": row[4] or 0,
                    "avg_cost_per_call": row[5] or 0.0,
                    "avg_processing_time": row[6] or 0.0,
                    "success_rate": row[7] or 0.0,
                    "start_time": row[8],
                    "end_time": row[9],
                    "by_operation": operation_breakdown,
                    "by_model": model_breakdown
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get session stats: {e}")
            return {}
    
    def get_daily_stats(self, date: str) -> Dict[str, Any]:
        """
        Get cost statistics for a specific date
        
        Args:
            date: Date in YYYY-MM-DD format
            
        Returns:
            Dictionary with daily statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_calls,
                        SUM(total_cost) as total_cost,
                        SUM(total_tokens) as total_tokens,
                        AVG(total_cost) as avg_cost_per_call,
                        COUNT(DISTINCT session_id) as unique_sessions
                    FROM api_calls 
                    WHERE DATE(timestamp) = ?
                """, (date,))
                
                row = cursor.fetchone()
                if not row:
                    return {}
                
                return {
                    "date": date,
                    "total_calls": row[0] or 0,
                    "total_cost": row[1] or 0.0,
                    "total_tokens": row[2] or 0,
                    "avg_cost_per_call": row[3] or 0.0,
                    "unique_sessions": row[4] or 0
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get daily stats: {e}")
            return {}
    
    def export_session_to_json(self, session_id: str, output_file: Optional[str] = None) -> str:
        """
        Export session data to JSON file and update aggregated costs
        
        Args:
            session_id: Session identifier
            output_file: Output file path (optional)
            
        Returns:
            Path to the exported JSON file
        """
        try:
            if not output_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = self.sessions_dir / f"session_{session_id}_{timestamp}.json"
            else:
                output_file = Path(output_file)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get all records for the session
                cursor.execute("""
                    SELECT * FROM api_calls 
                    WHERE session_id = ?
                    ORDER BY timestamp
                """, (session_id,))
                
                columns = [description[0] for description in cursor.description]
                records = []
                
                for row in cursor.fetchall():
                    record = dict(zip(columns, row))
                    records.append(record)
                
                # Get session stats
                stats = self.get_session_stats(session_id)
                
                export_data = {
                    "session_id": session_id,
                    "export_timestamp": datetime.now().isoformat(),
                    "statistics": stats,
                    "records": records
                }
                
                with open(output_file, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                
                # Update aggregated costs
                if stats:  # Only update if we have valid stats
                    self._update_aggregated_data(stats)
                
                self.logger.info(f"Session data exported to {output_file}")
                return str(output_file)
                
        except Exception as e:
            self.logger.error(f"Failed to export session to JSON: {e}")
            raise
    
    def get_aggregated_costs(self) -> Dict[str, Any]:
        """
        Get the current aggregated cost data
        
        Returns:
            Dictionary with aggregated cost information
        """
        return self._load_aggregated_data()
    
    def get_daily_aggregated_costs(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Get daily aggregated costs within a date range
        
        Args:
            start_date: Start date in YYYY-MM-DD format (optional)
            end_date: End date in YYYY-MM-DD format (optional)
            
        Returns:
            Dictionary with daily cost data
        """
        data = self._load_aggregated_data()
        daily_data = data.get("daily", {})
        
        if not start_date and not end_date:
            return daily_data
        
        filtered_data = {}
        for date_str, date_data in daily_data.items():
            if start_date and date_str < start_date:
                continue
            if end_date and date_str > end_date:
                continue
            filtered_data[date_str] = date_data
        
        return filtered_data
    
    def get_monthly_aggregated_costs(self, year: Optional[int] = None) -> Dict[str, Any]:
        """
        Get monthly aggregated costs for a specific year
        
        Args:
            year: Year to filter by (optional)
            
        Returns:
            Dictionary with monthly cost data
        """
        data = self._load_aggregated_data()
        monthly_data = data.get("monthly", {})
        
        if year is None:
            return monthly_data
        
        year_str = str(year)
        filtered_data = {k: v for k, v in monthly_data.items() if k.startswith(year_str)}
        return filtered_data
    
    def rebuild_aggregated_costs(self) -> Dict[str, Any]:
        """
        Rebuild aggregated costs from all session data in the database
        This is useful for fixing any inconsistencies or migrating existing data
        
        Returns:
            Dictionary with rebuild statistics
        """
        try:
            self.logger.info("Rebuilding aggregated costs from database...")
            
            # Reset aggregated data
            data = self._get_empty_aggregated_data()
            
            # Get all unique sessions from database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT session_id FROM api_calls ORDER BY timestamp")
                session_ids = [row[0] for row in cursor.fetchall()]
            
            # Process each session
            processed_sessions = 0
            for session_id in session_ids:
                try:
                    session_stats = self.get_session_stats(session_id)
                    if session_stats:
                        # Manually update data without saving after each session
                        self._update_aggregated_data_without_save(data, session_stats)
                        processed_sessions += 1
                except Exception as e:
                    self.logger.warning(f"Failed to process session {session_id}: {e}")
                    continue
            
            # Save the final aggregated data
            self._save_aggregated_data(data)
            
            rebuild_stats = {
                "total_sessions_found": len(session_ids),
                "sessions_processed": processed_sessions,
                "sessions_failed": len(session_ids) - processed_sessions,
                "total_cost": data["summary"]["total_cost"],
                "total_tokens": data["summary"]["total_tokens"],
                "rebuild_timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"Aggregated costs rebuilt: {processed_sessions}/{len(session_ids)} sessions processed")
            return rebuild_stats
            
        except Exception as e:
            self.logger.error(f"Failed to rebuild aggregated costs: {e}")
            raise
    
    def _update_aggregated_data_without_save(self, data: Dict[str, Any], session_stats: Dict[str, Any]):
        """Update aggregated data without saving (used for bulk operations)"""
        # This is the same logic as _update_aggregated_data but without the save call
        try:
            # Parse session start time to get date
            start_time = datetime.fromisoformat(session_stats["start_time"])
            session_date = start_time.date().isoformat()
            month_key = start_time.strftime("%Y-%m")
            year_key = start_time.strftime("%Y")
            
            # Update summary
            summary = data["summary"]
            summary["total_cost"] += session_stats["total_cost"]
            summary["total_tokens"] += session_stats["total_tokens"]
            summary["total_sessions"] += 1
            
            if summary["first_session_date"] is None or session_date < summary["first_session_date"]:
                summary["first_session_date"] = session_date
            
            # Update daily data
            if session_date not in data["daily"]:
                data["daily"][session_date] = {
                    "total_cost": 0.0,
                    "total_tokens": 0,
                    "total_sessions": 0,
                    "by_operation": {},
                    "by_model": {}
                }
            
            daily = data["daily"][session_date]
            daily["total_cost"] += session_stats["total_cost"]
            daily["total_tokens"] += session_stats["total_tokens"]
            daily["total_sessions"] += 1
            
            # Update daily operation breakdown
            for op, op_stats in session_stats.get("by_operation", {}).items():
                if op not in daily["by_operation"]:
                    daily["by_operation"][op] = {"calls": 0, "cost": 0.0, "tokens": 0}
                daily["by_operation"][op]["calls"] += op_stats["calls"]
                daily["by_operation"][op]["cost"] += op_stats["cost"]
                daily["by_operation"][op]["tokens"] += op_stats["tokens"]
            
            # Update daily model breakdown
            for model, model_stats in session_stats.get("by_model", {}).items():
                if model not in daily["by_model"]:
                    daily["by_model"][model] = {"calls": 0, "cost": 0.0, "tokens": 0}
                daily["by_model"][model]["calls"] += model_stats["calls"]
                daily["by_model"][model]["cost"] += model_stats["cost"]
                daily["by_model"][model]["tokens"] += model_stats["tokens"]
            
            # Update monthly data
            if month_key not in data["monthly"]:
                data["monthly"][month_key] = {
                    "total_cost": 0.0,
                    "total_tokens": 0,
                    "total_sessions": 0,
                    "days_active": 0,
                    "by_operation": {},
                    "by_model": {}
                }
            
            monthly = data["monthly"][month_key]
            monthly["total_cost"] += session_stats["total_cost"]
            monthly["total_tokens"] += session_stats["total_tokens"]
            monthly["total_sessions"] += 1
            
            # Count unique days in the month
            month_dates = {day for day in data["daily"].keys() if day.startswith(month_key)}
            monthly["days_active"] = len(month_dates)
            
            # Update monthly operation/model breakdowns (similar pattern)
            for op, op_stats in session_stats.get("by_operation", {}).items():
                if op not in monthly["by_operation"]:
                    monthly["by_operation"][op] = {"calls": 0, "cost": 0.0, "tokens": 0}
                monthly["by_operation"][op]["calls"] += op_stats["calls"]
                monthly["by_operation"][op]["cost"] += op_stats["cost"]
                monthly["by_operation"][op]["tokens"] += op_stats["tokens"]
            
            for model, model_stats in session_stats.get("by_model", {}).items():
                if model not in monthly["by_model"]:
                    monthly["by_model"][model] = {"calls": 0, "cost": 0.0, "tokens": 0}
                monthly["by_model"][model]["calls"] += model_stats["calls"]
                monthly["by_model"][model]["cost"] += model_stats["cost"]
                monthly["by_model"][model]["tokens"] += model_stats["tokens"]
            
            # Update yearly data
            if year_key not in data["yearly"]:
                data["yearly"][year_key] = {
                    "total_cost": 0.0,
                    "total_tokens": 0,
                    "total_sessions": 0,
                    "months_active": 0,
                    "by_operation": {},
                    "by_model": {}
                }
            
            yearly = data["yearly"][year_key]
            yearly["total_cost"] += session_stats["total_cost"]
            yearly["total_tokens"] += session_stats["total_tokens"]
            yearly["total_sessions"] += 1
            
            # Count unique months in the year
            year_months = {month for month in data["monthly"].keys() if month.startswith(year_key)}
            yearly["months_active"] = len(year_months)
            
            # Update overall operation breakdown
            for op, op_stats in session_stats.get("by_operation", {}).items():
                if op not in data["by_operation"]:
                    data["by_operation"][op] = {"calls": 0, "cost": 0.0, "tokens": 0}
                data["by_operation"][op]["calls"] += op_stats["calls"]
                data["by_operation"][op]["cost"] += op_stats["cost"]
                data["by_operation"][op]["tokens"] += op_stats["tokens"]
            
            # Update overall model breakdown
            for model, model_stats in session_stats.get("by_model", {}).items():
                if model not in data["by_model"]:
                    data["by_model"][model] = {"calls": 0, "cost": 0.0, "tokens": 0}
                data["by_model"][model]["calls"] += model_stats["calls"]
                data["by_model"][model]["cost"] += model_stats["cost"]
                data["by_model"][model]["tokens"] += model_stats["tokens"]
                
        except Exception as e:
            self.logger.error(f"Failed to update aggregated data (no save): {e}")
            raise
    
    def cleanup_old_records(self, days_to_keep: int = 30):
        """
        Clean up old records to manage database size
        
        Args:
            days_to_keep: Number of days of records to keep
        """
        try:
            cutoff_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            cutoff_date = cutoff_date.replace(day=cutoff_date.day - days_to_keep)
            cutoff_str = cutoff_date.isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Count records to be deleted
                cursor.execute("SELECT COUNT(*) FROM api_calls WHERE timestamp < ?", (cutoff_str,))
                count_to_delete = cursor.fetchone()[0]
                
                if count_to_delete > 0:
                    # Delete old records
                    cursor.execute("DELETE FROM api_calls WHERE timestamp < ?", (cutoff_str,))
                    conn.commit()
                    
                    self.logger.info(f"Cleaned up {count_to_delete} old records (older than {days_to_keep} days)")
                else:
                    self.logger.info("No old records to clean up")
                    
        except Exception as e:
            self.logger.error(f"Failed to cleanup old records: {e}")
            raise
    
    def get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get list of recent sessions
        
        Args:
            limit: Maximum number of sessions to return
            
        Returns:
            List of session information
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        session_id,
                        COUNT(*) as total_calls,
                        SUM(total_cost) as total_cost,
                        MIN(timestamp) as start_time,
                        MAX(timestamp) as end_time
                    FROM api_calls 
                    GROUP BY session_id
                    ORDER BY MAX(timestamp) DESC
                    LIMIT ?
                """, (limit,))
                
                sessions = []
                for row in cursor.fetchall():
                    sessions.append({
                        "session_id": row[0],
                        "total_calls": row[1],
                        "total_cost": row[2] or 0.0,
                        "start_time": row[3],
                        "end_time": row[4]
                    })
                
                return sessions
                
        except Exception as e:
            self.logger.error(f"Failed to get recent sessions: {e}")
            return [] 