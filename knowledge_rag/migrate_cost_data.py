#!/usr/bin/env python3
"""
Cost Data Migration Script

Migrates existing cost data to the new aggregated structure:
1. Moves session_*.json files to cost_data/sessions/ subfolder
2. Rebuilds aggregated_costs.json from all existing session data
3. Updates the cost tracking system to use the new structure
"""

import sys
import os
import json
import shutil
from pathlib import Path
from datetime import datetime

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

from src.cost_tracking import CostStorage


def find_session_files(cost_data_dir: Path) -> list:
    """Find all session JSON files in the cost_data directory"""
    session_files = []
    
    # Look for session_*.json files in the root cost_data directory
    pattern = "session_*.json"
    for file_path in cost_data_dir.glob(pattern):
        if file_path.is_file():
            session_files.append(file_path)
    
    return sorted(session_files, key=lambda f: f.stat().st_mtime)


def move_session_files_to_subfolder(cost_data_dir: Path, session_files: list) -> dict:
    """Move session files to the sessions/ subfolder"""
    sessions_dir = cost_data_dir / "sessions"
    sessions_dir.mkdir(exist_ok=True)
    
    migration_results = {
        "moved": 0,
        "skipped": 0,
        "errors": 0,
        "moved_files": []
    }
    
    for session_file in session_files:
        try:
            destination = sessions_dir / session_file.name
            
            # Skip if already in sessions folder
            if destination.exists():
                print(f"⏭️  Skipping {session_file.name} - already exists in sessions/")
                migration_results["skipped"] += 1
                continue
            
            # Move the file
            shutil.move(str(session_file), str(destination))
            print(f"📁 Moved {session_file.name} → sessions/{session_file.name}")
            
            migration_results["moved"] += 1
            migration_results["moved_files"].append(session_file.name)
            
        except Exception as e:
            print(f"❌ Error moving {session_file.name}: {e}")
            migration_results["errors"] += 1
    
    return migration_results


def validate_session_files(sessions_dir: Path) -> dict:
    """Validate that moved session files are valid JSON and contain expected data"""
    validation_results = {
        "valid": 0,
        "invalid": 0,
        "total_sessions": 0,
        "total_cost": 0.0,
        "total_tokens": 0,
        "date_range": {"earliest": None, "latest": None}
    }
    
    session_files = list(sessions_dir.glob("session_*.json"))
    
    for session_file in session_files:
        try:
            with open(session_file, 'r') as f:
                data = json.load(f)
            
            # Check if it has expected structure
            if "session_id" in data and "statistics" in data:
                stats = data["statistics"]
                
                # Extract basic info
                session_cost = stats.get("total_cost", 0)
                session_tokens = stats.get("total_tokens", 0)
                start_time = stats.get("start_time")
                
                validation_results["valid"] += 1
                validation_results["total_cost"] += session_cost
                validation_results["total_tokens"] += session_tokens
                
                # Track date range
                if start_time:
                    try:
                        start_date = datetime.fromisoformat(start_time).date().isoformat()
                        if validation_results["date_range"]["earliest"] is None or start_date < validation_results["date_range"]["earliest"]:
                            validation_results["date_range"]["earliest"] = start_date
                        if validation_results["date_range"]["latest"] is None or start_date > validation_results["date_range"]["latest"]:
                            validation_results["date_range"]["latest"] = start_date
                    except:
                        pass
                
                print(f"✅ Valid: {session_file.name} - ${session_cost:.4f}, {session_tokens:,} tokens")
            else:
                print(f"⚠️  Invalid structure: {session_file.name}")
                validation_results["invalid"] += 1
                
        except Exception as e:
            print(f"❌ Error validating {session_file.name}: {e}")
            validation_results["invalid"] += 1
    
    validation_results["total_sessions"] = validation_results["valid"]
    return validation_results


def backup_existing_data(cost_data_dir: Path) -> str:
    """Create a backup of existing cost data"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = cost_data_dir / f"backup_{timestamp}"
    backup_dir.mkdir(exist_ok=True)
    
    backed_up_files = []
    
    # Backup any existing aggregated_costs.json
    aggregated_file = cost_data_dir / "aggregated_costs.json"
    if aggregated_file.exists():
        backup_aggregated = backup_dir / "aggregated_costs.json"
        shutil.copy2(str(aggregated_file), str(backup_aggregated))
        backed_up_files.append("aggregated_costs.json")
    
    # Copy database file
    db_file = cost_data_dir / "api_costs.db"
    if db_file.exists():
        backup_db = backup_dir / "api_costs.db"
        shutil.copy2(str(db_file), str(backup_db))
        backed_up_files.append("api_costs.db")
    
    print(f"💾 Backup created: {backup_dir.name}")
    for file in backed_up_files:
        print(f"   • {file}")
    
    return str(backup_dir)


def main():
    """Main migration function"""
    
    print("🔄 Knowledge RAG - Cost Data Migration")
    print("=" * 45)
    print("This script will:")
    print("  1. Move session_*.json files to cost_data/sessions/")
    print("  2. Create aggregated_costs.json with all historical data")
    print("  3. Backup existing data before migration")
    print("")
    
    # Confirm with user
    try:
        confirm = input("Do you want to proceed? (y/N): ").strip().lower()
        if confirm not in ['y', 'yes']:
            print("👋 Migration cancelled by user")
            return False
    except KeyboardInterrupt:
        print("\n👋 Migration cancelled by user")
        return False
    
    cost_data_dir = Path("cost_data")
    
    # Check if cost_data directory exists
    if not cost_data_dir.exists():
        print("❌ cost_data directory not found")
        print("Nothing to migrate - this appears to be a fresh installation")
        return True
    
    print(f"\n📁 Working with: {cost_data_dir.absolute()}")
    
    try:
        # Step 1: Create backup
        print(f"\n1️⃣  Creating Backup")
        print("-" * 20)
        backup_path = backup_existing_data(cost_data_dir)
        
        # Step 2: Find existing session files
        print(f"\n2️⃣  Finding Session Files")
        print("-" * 25)
        session_files = find_session_files(cost_data_dir)
        
        if not session_files:
            print("📭 No session JSON files found in cost_data/")
            print("Nothing to migrate")
            return True
        
        print(f"📊 Found {len(session_files)} session files:")
        for session_file in session_files:
            file_size = session_file.stat().st_size / 1024  # KB
            modified = datetime.fromtimestamp(session_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            print(f"   • {session_file.name} ({file_size:.1f}KB, modified: {modified})")
        
        # Step 3: Move files to sessions subfolder
        print(f"\n3️⃣  Moving Files to sessions/")
        print("-" * 30)
        migration_results = move_session_files_to_subfolder(cost_data_dir, session_files)
        
        print(f"📈 Migration Results:")
        print(f"   • Files moved: {migration_results['moved']}")
        print(f"   • Files skipped: {migration_results['skipped']}")
        print(f"   • Errors: {migration_results['errors']}")
        
        if migration_results["errors"] > 0:
            print("⚠️  Some files had errors during migration")
            print("Check the backup directory if you need to recover any files")
        
        # Step 4: Validate moved files
        print(f"\n4️⃣  Validating Session Files")
        print("-" * 30)
        sessions_dir = cost_data_dir / "sessions"
        validation_results = validate_session_files(sessions_dir)
        
        print(f"📊 Validation Results:")
        print(f"   • Valid sessions: {validation_results['valid']}")
        print(f"   • Invalid sessions: {validation_results['invalid']}")
        print(f"   • Total cost: ${validation_results['total_cost']:.4f}")
        print(f"   • Total tokens: {validation_results['total_tokens']:,}")
        if validation_results["date_range"]["earliest"]:
            print(f"   • Date range: {validation_results['date_range']['earliest']} to {validation_results['date_range']['latest']}")
        
        # Step 5: Rebuild aggregated costs
        print(f"\n5️⃣  Rebuilding Aggregated Costs")
        print("-" * 35)
        
        # Initialize the new cost storage system
        storage = CostStorage()
        
        print("🔄 Rebuilding aggregated costs from all session data...")
        rebuild_stats = storage.rebuild_aggregated_costs()
        
        print(f"✅ Aggregated costs rebuilt:")
        print(f"   • Sessions found: {rebuild_stats['total_sessions_found']}")
        print(f"   • Sessions processed: {rebuild_stats['sessions_processed']}")
        print(f"   • Sessions failed: {rebuild_stats['sessions_failed']}")
        print(f"   • Total cost: ${rebuild_stats['total_cost']:.4f}")
        print(f"   • Total tokens: {rebuild_stats['total_tokens']:,}")
        
        # Step 6: Verify the new structure
        print(f"\n6️⃣  Verifying New Structure")
        print("-" * 30)
        
        aggregated_file = cost_data_dir / "aggregated_costs.json"
        if aggregated_file.exists():
            print(f"✅ aggregated_costs.json created ({aggregated_file.stat().st_size / 1024:.1f}KB)")
            
            # Show a quick summary
            with open(aggregated_file, 'r') as f:
                aggregated_data = json.load(f)
            
            summary = aggregated_data.get("summary", {})
            print(f"📊 Aggregated Summary:")
            print(f"   • Total sessions: {summary.get('total_sessions', 0)}")
            print(f"   • Total cost: ${summary.get('total_cost', 0):.4f}")
            print(f"   • Days active: {len(aggregated_data.get('daily', {}))}")
            print(f"   • First session: {summary.get('first_session_date', 'Unknown')}")
        else:
            print("❌ aggregated_costs.json not created")
            return False
        
        if sessions_dir.exists():
            session_count = len(list(sessions_dir.glob("session_*.json")))
            print(f"✅ sessions/ directory contains {session_count} files")
        else:
            print("❌ sessions/ directory not found")
        
        print(f"\n🎉 Migration Completed Successfully!")
        print("=" * 35)
        print("📁 New Structure:")
        print("   cost_data/")
        print("   ├── aggregated_costs.json       # 📊 All aggregated cost data")
        print("   ├── sessions/")
        print("   │   ├── session_abc123_*.json   # 📝 Individual session details")
        print("   │   └── session_def456_*.json")
        print("   ├── api_costs.db                # 🗄️  SQLite database (unchanged)")
        print(f"   └── {Path(backup_path).name}/                # 💾 Backup of original data")
        print("")
        print("✨ Benefits of the new structure:")
        print("   • Faster access to cost summaries")
        print("   • Daily/monthly/yearly aggregations")
        print("   • Cost trend analysis")
        print("   • Organized session files")
        print("   • Better performance for large datasets")
        print("")
        print("🚀 You can now use the enhanced cost analysis features!")
        
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n🔧 Recovery:")
        print(f"   Your original data is backed up in: {backup_path if 'backup_path' in locals() else 'cost_data/backup_*'}")
        print(f"   You can manually restore files if needed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 