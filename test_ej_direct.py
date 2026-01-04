"""
Direct test of EJ Service without Flask server
"""
import sys
sys.path.insert(0, 'F:/NW/NetCon/NetCon/src')

from services.ej_service import EJService
from pathlib import Path
import json
import pandas as pd

print("="*80)
print("DIRECT EJ SERVICE TEST")
print("="*80)

# Initialize service
print("\n1. Initializing EJ Service...")
try:
    ej_service = EJService()
    print("   ✓ EJ Service initialized successfully")
except Exception as e:
    print(f"   ✗ Failed to initialize: {e}")
    exit(1)

# Check health
print("\n2. Checking service health...")
try:
    health = ej_service.health_check()
    print(f"   Health Status: {health}")
except Exception as e:
    print(f"   ✗ Health check failed: {e}")

# Check trial status
print("\n3. Checking trial status...")
try:
    is_active = ej_service.is_trial_active()
    print(f"   Trial Active: {is_active}")
except Exception as e:
    print(f"   ✗ Trial check failed: {e}")

# Load EJ log files
print("\n4. Loading EJ log files...")
ej_dir = Path("F:/NW/NetCon/NetCon/ej-logs/CRM-EJBackups")
ej_files = sorted(list(ej_dir.glob("EJCRM*.0*")))[:3]  # Take first 3 files

if not ej_files:
    print("   ✗ No EJ files found")
    exit(1)

print(f"   Found {len(ej_files)} EJ files to process:")
for f in ej_files:
    print(f"     - {f.name}")

try:
    file_paths = [str(f) for f in ej_files]
    log_contents = ej_service.load_logs(file_paths)
    print(f"   ✓ Successfully loaded {len(log_contents)} files")
    
    for file_path, lines in log_contents.items():
        print(f"     - {Path(file_path).name}: {len(lines)} lines")
except Exception as e:
    print(f"   ✗ Failed to load logs: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Process transactions
print("\n5. Processing transactions...")
try:
    df_transactions = ej_service.process_transactions(log_contents)
    print(f"   ✓ Successfully processed transactions")
    print(f"   Total transactions extracted: {len(df_transactions)}")
    
    if len(df_transactions) > 0:
        print(f"\n   DataFrame Info:")
        print(f"     - Shape: {df_transactions.shape}")
        print(f"     - Columns: {len(df_transactions.columns)}")
        
        # Show sample columns
        key_columns = ['transaction_id', 'timestamp', 'card_number', 'transaction_type', 
                      'amount', 'response_code', 'status', 'scenario']
        available_columns = [col for col in key_columns if col in df_transactions.columns]
        
        print(f"\n   Sample transactions ({len(available_columns)} key columns):")
        print(df_transactions[available_columns].head(5).to_string())
        
        # Show transaction type distribution
        if 'transaction_type' in df_transactions.columns:
            print(f"\n   Transaction Type Distribution:")
            type_dist = df_transactions['transaction_type'].value_counts()
            for tx_type, count in type_dist.items():
                print(f"     - {tx_type}: {count}")
        
        # Show scenario distribution
        if 'scenario' in df_transactions.columns:
            print(f"\n   Scenario Distribution:")
            scenario_dist = df_transactions['scenario'].value_counts()
            for scenario, count in scenario_dist.items():
                print(f"     - {scenario}: {count}")
        
        # Show response code distribution
        if 'response_code' in df_transactions.columns:
            print(f"\n   Response Code Distribution:")
            code_dist = df_transactions['response_code'].value_counts().head(5)
            for code, count in code_dist.items():
                print(f"     - {code}: {count}")
        
        # Save to CSV for analysis
        output_file = "F:/NW/NetCon/NetCon/test_output.csv"
        df_transactions.to_csv(output_file, index=False)
        print(f"\n   ✓ Saved results to: {output_file}")
        
    else:
        print("   ⚠ No transactions found in the log files")
        
except Exception as e:
    print(f"   ✗ Failed to process transactions: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Get processing metrics
print("\n6. Processing Metrics:")
try:
    metrics = ej_service.get_processing_metrics()
    print(f"   - Files Processed: {metrics.files_processed}")
    print(f"   - Transactions Extracted: {metrics.transactions_extracted}")
    print(f"   - Processing Time: {metrics.processing_time:.2f}s")
    if metrics.errors:
        print(f"   - Errors: {len(metrics.errors)}")
        for error in metrics.errors[:3]:
            print(f"     • {error}")
except Exception as e:
    print(f"   ✗ Failed to get metrics: {e}")

print("\n" + "="*80)
print("TEST COMPLETED")
print("="*80)
