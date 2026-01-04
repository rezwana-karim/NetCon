"""
Simple script to test the Flask API with EJ log files
"""
import requests
import json
from pathlib import Path

# Test health endpoint
print("Testing health endpoint...")
try:
    response = requests.get("http://127.0.0.1:5000/health", timeout=5)
    print(f"Health Status: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Health check failed: {e}")

# Test EJ controller health
print("\nTesting EJ controller health...")
try:
    response = requests.get("http://127.0.0.1:5000/api/ej/health", timeout=5)
    print(f"EJ Health Status: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"EJ health check failed: {e}")

# Test load_logs endpoint with an EJ file
print("\nTesting load_logs endpoint with EJ file...")
try:
    # Get the first EJ log file
    ej_files_dir = Path("F:/NW/NetCon/NetCon/ej-logs/CRM-EJBackups")
    ej_files = list(ej_files_dir.glob("EJCRM*.0*"))
    
    if ej_files:
        test_file = ej_files[0]
        print(f"Using test file: {test_file}")
        
        with open(test_file, 'rb') as f:
            files = {'files': (test_file.name, f, 'application/octet-stream')}
            response = requests.post(
                "http://127.0.0.1:5000/api/ej/load_logs",
                files=files,
                timeout=30
            )
            
            print(f"Load logs Status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"\nSummary:")
                print(json.dumps(result.get('summary', {}), indent=2))
                print(f"\nTransactions extracted: {len(result.get('transactions', []))}")
                if result.get('transactions'):
                    print(f"\nFirst transaction sample:")
                    print(json.dumps(result['transactions'][0], indent=2))
            else:
                print(f"Response: {response.text}")
    else:
        print("No EJ files found in the directory")
        
except Exception as e:
    print(f"Load logs test failed: {e}")
    import traceback
    traceback.print_exc()
