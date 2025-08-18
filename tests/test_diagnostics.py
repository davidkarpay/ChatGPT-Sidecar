#!/usr/bin/env python3
"""
Diagnostic test suite for Sidecar search functionality.
Tests the current state and all three proposed solutions.
"""

import requests
import sqlite3
import os
import json
from pathlib import Path
import time

# Configuration
BASE_URL = "http://127.0.0.1:8088"
API_KEY = "sidecar-AfreWVOEVoCtXzMT0jejgTqsng4J-kwlICBQonyMbas"
DB_PATH = "sidecar.db"
HEADERS = {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY
}

class DiagnosticTests:
    def __init__(self):
        self.results = {}
        
    def test_server_health(self):
        """Test if server is running and healthy."""
        print("🏥 Testing server health...")
        try:
            response = requests.get(f"{BASE_URL}/healthz", timeout=5)
            success = response.status_code == 200 and response.json().get("ok") == True
            self.results["server_health"] = success
            print(f"   ✅ Server health: {'PASS' if success else 'FAIL'}")
            return success
        except Exception as e:
            print(f"   ❌ Server health: FAIL - {e}")
            self.results["server_health"] = False
            return False
    
    def test_api_auth(self):
        """Test API key authentication."""
        print("🔐 Testing API authentication...")
        try:
            # Test with correct key
            response = requests.post(f"{BASE_URL}/search", 
                                   headers=HEADERS,
                                   json={"query": "test", "k": 1},
                                   timeout=5)
            auth_works = response.status_code != 401
            
            # Test with wrong key
            bad_headers = HEADERS.copy()
            bad_headers["X-API-Key"] = "wrong-key"
            response_bad = requests.post(f"{BASE_URL}/search",
                                       headers=bad_headers, 
                                       json={"query": "test", "k": 1},
                                       timeout=5)
            auth_secure = response_bad.status_code == 401
            
            success = auth_works and auth_secure
            self.results["api_auth"] = success
            print(f"   ✅ API auth: {'PASS' if success else 'FAIL'}")
            return success
        except Exception as e:
            print(f"   ❌ API auth: FAIL - {e}")
            self.results["api_auth"] = False
            return False
    
    def test_database_state(self):
        """Test database connectivity and content."""
        print("🗄️  Testing database state...")
        try:
            if not os.path.exists(DB_PATH):
                print(f"   ❌ Database missing: {DB_PATH}")
                self.results["database"] = False
                return False
                
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Count chunks
            cursor.execute("SELECT COUNT(*) FROM chunk")
            chunk_count = cursor.fetchone()[0]
            
            # Count documents  
            cursor.execute("SELECT COUNT(*) FROM document")
            doc_count = cursor.fetchone()[0]
            
            # Check embedding refs by index
            cursor.execute("SELECT index_name, COUNT(*) FROM embedding_ref GROUP BY index_name")
            indexes = dict(cursor.fetchall())
            
            conn.close()
            
            self.results["database"] = {
                "chunks": chunk_count,
                "documents": doc_count, 
                "indexes": indexes
            }
            
            print(f"   ✅ Database: {chunk_count:,} chunks, {doc_count:,} docs")
            print(f"   📊 Indexes: {indexes}")
            return chunk_count > 0
            
        except Exception as e:
            print(f"   ❌ Database: FAIL - {e}")
            self.results["database"] = False
            return False
    
    def test_index_files(self):
        """Test FAISS index file existence."""
        print("📁 Testing index files...")
        index_dir = Path("data/indexes")
        
        files_found = {}
        if index_dir.exists():
            for file in index_dir.glob("*.faiss"):
                pkl_file = file.with_suffix(".pkl")
                files_found[file.stem] = {
                    "faiss": file.exists(),
                    "pkl": pkl_file.exists(),
                    "faiss_size": file.stat().st_size if file.exists() else 0,
                    "pkl_size": pkl_file.stat().st_size if pkl_file.exists() else 0
                }
        
        self.results["index_files"] = files_found
        
        main_exists = "main" in files_found
        print(f"   {'✅' if main_exists else '❌'} Main index: {'EXISTS' if main_exists else 'MISSING'}")
        
        for name, info in files_found.items():
            if name != "main":
                size_mb = (info["faiss_size"] + info["pkl_size"]) / 1024 / 1024
                print(f"   📦 {name}: {size_mb:.1f}MB")
        
        return files_found
    
    def test_search_endpoints(self):
        """Test all search endpoint variants."""
        print("🔍 Testing search endpoints...")
        
        test_query = "search test query"
        endpoints = {
            "basic_search": {
                "url": "/search",
                "payload": {"query": test_query, "k": 3}
            },
            "advanced_search": {
                "url": "/search/advanced", 
                "payload": {"query": test_query, "k": 3, "candidates": 10, "lambda": 0.5}
            },
            "multi_layer_search": {
                "url": "/search-multi-layer",
                "payload": {"query": test_query, "k": 3}
            }
        }
        
        # Test with different index names
        layer_tests = {}
        if self.results.get("database", {}).get("indexes"):
            for index_name in self.results["database"]["indexes"].keys():
                layer_tests[f"layer_{index_name}"] = {
                    "url": "/search",
                    "payload": {"query": test_query, "k": 3, "index_name": index_name}
                }
        
        all_tests = {**endpoints, **layer_tests}
        search_results = {}
        
        for test_name, config in all_tests.items():
            try:
                start_time = time.time()
                response = requests.post(
                    f"{BASE_URL}{config['url']}", 
                    headers=HEADERS,
                    json=config["payload"],
                    timeout=10
                )
                duration = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    hit_count = len(data.get("hits", []))
                    search_results[test_name] = {
                        "status": "SUCCESS",
                        "hits": hit_count,
                        "duration_ms": round(duration * 1000, 1)
                    }
                    print(f"   ✅ {test_name}: {hit_count} hits ({duration*1000:.1f}ms)")
                else:
                    search_results[test_name] = {
                        "status": "ERROR", 
                        "code": response.status_code,
                        "error": response.text[:100]
                    }
                    print(f"   ❌ {test_name}: {response.status_code}")
                    
            except Exception as e:
                search_results[test_name] = {"status": "EXCEPTION", "error": str(e)}
                print(f"   ❌ {test_name}: {e}")
        
        self.results["search_endpoints"] = search_results
        return search_results

def run_diagnostics():
    """Run all diagnostic tests and return results."""
    print("🚀 Starting Sidecar diagnostic tests...\n")
    
    tests = DiagnosticTests()
    
    # Run all tests
    tests.test_server_health()
    tests.test_api_auth() 
    tests.test_database_state()
    tests.test_index_files()
    tests.test_search_endpoints()
    
    print(f"\n📋 DIAGNOSTIC SUMMARY")
    print("=" * 50)
    
    # Analysis
    has_data = tests.results.get("database", {}).get("chunks", 0) > 0
    has_main_index = "main" in tests.results.get("index_files", {})
    search_working = any(
        result.get("hits", 0) > 0 
        for result in tests.results.get("search_endpoints", {}).values()
        if isinstance(result, dict)
    )
    
    print(f"Database has data: {'✅' if has_data else '❌'}")
    print(f"Main index exists: {'✅' if has_main_index else '❌'}")  
    print(f"Search returning results: {'✅' if search_working else '❌'}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    if has_data and not has_main_index:
        print("🔧 SOLUTION: Run reindex to create main index")
        print("   curl -X POST http://127.0.0.1:8088/reindex \\")
        print(f"     -H 'X-API-Key: {API_KEY}' \\")
        print("     -H 'Content-Type: application/json' \\")
        print("     -d '{\"index_name\": \"main\"}'")
    
    if not search_working and has_data:
        print("🔧 Consider using multi-layer search or specific layer search")
    
    return tests.results

if __name__ == "__main__":
    results = run_diagnostics()
    
    # Save results
    with open("diagnostic_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to diagnostic_results.json")