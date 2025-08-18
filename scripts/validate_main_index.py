#!/usr/bin/env python3
"""
Validation script for the rebuilt main index.
Performs comprehensive testing to ensure the index is working correctly.
"""

import sys
import time
import requests
import sqlite3
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.db import DB
from app.vectorstore import FaissStore

class MainIndexValidator:
    def __init__(self):
        self.db_path = "sidecar.db"
        self.index_dir = Path("data/indexes")
        self.embed_model = "sentence-transformers/all-MiniLM-L6-v2"
        
        # API testing
        self.base_url = "http://127.0.0.1:8088"
        self.api_key = "sidecar-AfreWVOEVoCtXzMT0jejgTqsng4J-kwlICBQonyMbas"
        self.headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key
        }
        
        # Main index paths
        self.main_faiss_path = self.index_dir / "main.faiss"
        self.main_pkl_path = self.index_dir / "main.pkl"
        
        print("🔍 Main Index Validator")
        print("=" * 50)
    
    def test_file_existence(self):
        """Test that main index files exist."""
        print("📁 Testing file existence...")
        
        faiss_exists = self.main_faiss_path.exists()
        pkl_exists = self.main_pkl_path.exists()
        
        print(f"   main.faiss: {'✅' if faiss_exists else '❌'} {self.main_faiss_path}")
        print(f"   main.pkl: {'✅' if pkl_exists else '❌'} {self.main_pkl_path}")
        
        if faiss_exists:
            size_mb = self.main_faiss_path.stat().st_size / 1024 / 1024
            print(f"   Size: {size_mb:.1f} MB")
        
        return faiss_exists and pkl_exists
    
    def test_index_loading(self):
        """Test that the index can be loaded."""
        print("\n🏗️  Testing index loading...")
        
        try:
            store = FaissStore(self.main_faiss_path, self.main_pkl_path, self.embed_model)
            store.load()
            
            if store.index is None:
                print("   ❌ Index is None after loading")
                return False, None
            
            vector_count = store.index.ntotal
            dimensions = store.index.d
            id_count = len(store.ids)
            
            print(f"   ✅ Index loaded successfully")
            print(f"   📊 Vectors: {vector_count:,}")
            print(f"   📐 Dimensions: {dimensions}")
            print(f"   🆔 ID mappings: {id_count:,}")
            
            # Check consistency
            if vector_count == id_count:
                print("   ✅ Vector count matches ID count")
                return True, store
            else:
                print(f"   ❌ Mismatch: {vector_count:,} vectors vs {id_count:,} IDs")
                return False, store
                
        except Exception as e:
            print(f"   ❌ Failed to load index: {e}")
            return False, None
    
    def test_database_consistency(self):
        """Test database embedding references."""
        print("\n📊 Testing database consistency...")
        
        try:
            with DB(self.db_path) as db:
                # Count main index references
                cursor = db.conn.execute(
                    "SELECT COUNT(*) as count FROM embedding_ref WHERE index_name = 'main'"
                )
                main_refs = cursor.fetchone()["count"]
                
                # Count total chunks
                cursor = db.conn.execute("SELECT COUNT(*) as count FROM chunk")
                total_chunks = cursor.fetchone()["count"]
                
                # Check for duplicates
                cursor = db.conn.execute(
                    "SELECT chunk_id, COUNT(*) as dup_count FROM embedding_ref WHERE index_name = 'main' GROUP BY chunk_id HAVING COUNT(*) > 1"
                )
                duplicates = cursor.fetchall()
                
                print(f"   📋 Main index refs: {main_refs:,}")
                print(f"   📋 Total chunks: {total_chunks:,}")
                print(f"   📈 Coverage: {100*main_refs/total_chunks:.1f}%")
                
                if duplicates:
                    print(f"   ⚠️  Found {len(duplicates)} duplicate chunk references")
                    return False
                else:
                    print("   ✅ No duplicate references found")
                
                # Ideal case: all chunks have main index references
                if main_refs == total_chunks:
                    print("   ✅ Perfect coverage: all chunks indexed")
                    return True
                elif main_refs > 0.95 * total_chunks:
                    print("   ✅ Good coverage: >95% chunks indexed")
                    return True
                else:
                    print(f"   ⚠️  Low coverage: {100*main_refs/total_chunks:.1f}%")
                    return False
                    
        except Exception as e:
            print(f"   ❌ Database test failed: {e}")
            return False
    
    def test_search_functionality(self, store):
        """Test search using the vectorstore directly."""
        print("\n🔍 Testing search functionality...")
        
        test_queries = [
            "search",
            "conversation",
            "data analysis",
            "machine learning",
            "text processing"
        ]
        
        try:
            all_passed = True
            
            for i, query in enumerate(test_queries, 1):
                start_time = time.time()
                results = store.search(query, k=5)
                duration = time.time() - start_time
                
                success = len(results) > 0
                status = "✅" if success else "❌"
                print(f"   {status} Query {i}: '{query}' → {len(results)} results ({duration*1000:.1f}ms)")
                
                if not success:
                    all_passed = False
            
            return all_passed
            
        except Exception as e:
            print(f"   ❌ Search test failed: {e}")
            return False
    
    def test_api_endpoints(self):
        """Test API endpoints that use the main index."""
        print("\n🌐 Testing API endpoints...")
        
        test_endpoints = [
            {
                "name": "Basic Search",
                "url": "/search",
                "payload": {"query": "search test", "k": 3}
            },
            {
                "name": "Advanced MMR Search", 
                "url": "/search/advanced",
                "payload": {"query": "search test", "k": 3, "candidates": 10, "lambda": 0.5}
            }
        ]
        
        all_passed = True
        
        for endpoint in test_endpoints:
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}{endpoint['url']}",
                    headers=self.headers,
                    json=endpoint["payload"],
                    timeout=10
                )
                duration = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    hits = len(data.get("hits", []))
                    
                    if hits > 0:
                        print(f"   ✅ {endpoint['name']}: {hits} hits ({duration*1000:.1f}ms)")
                    else:
                        print(f"   ⚠️  {endpoint['name']}: 0 hits ({duration*1000:.1f}ms)")
                        all_passed = False
                else:
                    print(f"   ❌ {endpoint['name']}: HTTP {response.status_code}")
                    all_passed = False
                    
            except requests.exceptions.RequestException as e:
                print(f"   ❌ {endpoint['name']}: Request failed - {e}")
                all_passed = False
            except Exception as e:
                print(f"   ❌ {endpoint['name']}: {e}")
                all_passed = False
        
        return all_passed
    
    def compare_with_multi_layer(self):
        """Compare main index results with multi-layer search."""
        print("\n⚖️  Comparing with multi-layer search...")
        
        test_query = "search functionality"
        
        try:
            # Test main index search
            main_response = requests.post(
                f"{self.base_url}/search",
                headers=self.headers,
                json={"query": test_query, "k": 5},
                timeout=10
            )
            
            # Test multi-layer search
            multi_response = requests.post(
                f"{self.base_url}/search-multi-layer",
                headers=self.headers,
                json={"query": test_query, "k": 5},
                timeout=10
            )
            
            if main_response.status_code == 200 and multi_response.status_code == 200:
                main_hits = len(main_response.json().get("hits", []))
                multi_hits = len(multi_response.json().get("hits", []))
                
                print(f"   📊 Main index: {main_hits} hits")
                print(f"   📊 Multi-layer: {multi_hits} hits")
                
                if main_hits > 0 and multi_hits > 0:
                    print("   ✅ Both search methods returning results")
                    return True
                elif main_hits > 0:
                    print("   ✅ Main index working (multi-layer may have issues)")
                    return True
                else:
                    print("   ❌ Neither search method working")
                    return False
            else:
                print("   ❌ API requests failed")
                return False
                
        except Exception as e:
            print(f"   ❌ Comparison failed: {e}")
            return False
    
    def run_validation(self):
        """Run complete validation suite."""
        start_time = time.time()
        
        print("🚀 Starting main index validation...\n")
        
        results = {}
        
        # Test 1: File existence
        results["files"] = self.test_file_existence()
        
        # Test 2: Index loading
        results["loading"], store = self.test_index_loading()
        
        # Test 3: Database consistency
        results["database"] = self.test_database_consistency()
        
        # Test 4: Search functionality (if index loaded)
        if store:
            results["search"] = self.test_search_functionality(store)
        else:
            results["search"] = False
        
        # Test 5: API endpoints
        results["api"] = self.test_api_endpoints()
        
        # Test 6: Multi-layer comparison
        results["comparison"] = self.compare_with_multi_layer()
        
        # Summary
        duration = time.time() - start_time
        passed = sum(results.values())
        total = len(results)
        
        print(f"\n📋 VALIDATION SUMMARY")
        print("=" * 50)
        print(f"Tests passed: {passed}/{total}")
        print(f"Duration: {duration:.1f} seconds")
        
        for test_name, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {status} {test_name.title()}")
        
        if passed == total:
            print(f"\n🎉 SUCCESS! Main index is fully functional.")
            print(f"   🎯 Basic search: http://127.0.0.1:8088/static/index_enhanced.html")
            return True
        else:
            print(f"\n⚠️  PARTIAL SUCCESS: {passed}/{total} tests passed")
            print(f"   💡 Check failed tests and consider re-running rebuild")
            return False

def main():
    validator = MainIndexValidator()
    success = validator.run_validation()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()