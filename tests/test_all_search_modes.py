#!/usr/bin/env python3
"""
Test all search modes to validate the proposed solutions.
"""

import requests
import time
import json

BASE_URL = "http://127.0.0.1:8088"
API_KEY = "sidecar-AfreWVOEVoCtXzMT0jejgTqsng4J-kwlICBQonyMbas"
HEADERS = {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY
}

def test_search_mode(name, endpoint, payload, expected_working=True):
    """Test a specific search mode."""
    print(f"\n🔍 Testing {name}...")
    
    try:
        start_time = time.time()
        response = requests.post(f"{BASE_URL}{endpoint}", 
                               headers=HEADERS,
                               json=payload,
                               timeout=30)
        duration = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            hits = len(data.get("hits", []))
            
            if hits > 0:
                print(f"   ✅ SUCCESS: {hits} hits in {duration*1000:.1f}ms")
                return True, hits, duration
            else:
                status = "EXPECTED" if not expected_working else "UNEXPECTED"
                print(f"   ⚠️  NO RESULTS ({status}): 0 hits in {duration*1000:.1f}ms")
                return False, 0, duration
        else:
            print(f"   ❌ ERROR: {response.status_code} - {response.text[:100]}")
            return False, 0, duration
            
    except Exception as e:
        print(f"   ❌ EXCEPTION: {e}")
        return False, 0, 0

def run_comprehensive_tests():
    """Run all search mode tests."""
    print("🚀 Testing all search modes with query: 'chunking'")
    
    test_query = "chunking"
    results = {}
    
    # Test modes
    test_cases = [
        {
            "name": "Basic Search (main index)",
            "endpoint": "/search",
            "payload": {"query": test_query, "k": 5},
            "expected": False  # Main index might not exist yet
        },
        {
            "name": "Advanced MMR Search (main index)", 
            "endpoint": "/search/advanced",
            "payload": {"query": test_query, "k": 5, "candidates": 20, "lambda": 0.5},
            "expected": False  # Main index might not exist yet
        },
        {
            "name": "Multi-Layer Search",
            "endpoint": "/search-multi-layer", 
            "payload": {"query": test_query, "k": 5},
            "expected": True  # Should work with existing indexes
        },
        {
            "name": "Precision Layer Search",
            "endpoint": "/search",
            "payload": {"query": test_query, "k": 5, "index_name": "precision"},
            "expected": False  # May need proper index loading
        },
        {
            "name": "Balanced Layer Search",
            "endpoint": "/search", 
            "payload": {"query": test_query, "k": 5, "index_name": "balanced"},
            "expected": False  # May need proper index loading
        },
        {
            "name": "Context Layer Search",
            "endpoint": "/search",
            "payload": {"query": test_query, "k": 5, "index_name": "context"}, 
            "expected": False  # May need proper index loading
        }
    ]
    
    working_modes = []
    for test_case in test_cases:
        success, hits, duration = test_search_mode(
            test_case["name"],
            test_case["endpoint"], 
            test_case["payload"],
            test_case["expected"]
        )
        
        results[test_case["name"]] = {
            "success": success,
            "hits": hits,
            "duration_ms": round(duration * 1000, 1),
            "endpoint": test_case["endpoint"],
            "payload": test_case["payload"]
        }
        
        if success:
            working_modes.append(test_case["name"])
    
    print(f"\n📊 RESULTS SUMMARY")
    print("=" * 60)
    print(f"Working modes: {len(working_modes)}")
    for mode in working_modes:
        result = results[mode]
        print(f"   ✅ {mode}: {result['hits']} hits ({result['duration_ms']}ms)")
    
    if not working_modes:
        print("   ❌ No search modes are currently working!")
        print("   💡 RECOMMENDATIONS:")
        print("   1. Wait for main index build to complete")
        print("   2. Use multi-layer search if available")
        print("   3. Check server logs for errors")
    
    # Save detailed results
    with open("search_mode_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results, working_modes

def test_ui_compatibility():
    """Test that the enhanced UI can access all endpoints."""
    print(f"\n🌐 Testing UI endpoint compatibility...")
    
    # Test that the enhanced UI file is accessible
    try:
        response = requests.get(f"{BASE_URL}/static/index_enhanced.html")
        if response.status_code == 200:
            print("   ✅ Enhanced UI accessible via /static/index_enhanced.html")
        else:
            print(f"   ❌ Enhanced UI not accessible: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Enhanced UI test failed: {e}")

if __name__ == "__main__":
    results, working_modes = run_comprehensive_tests()
    test_ui_compatibility()
    
    print(f"\n🎯 IMMEDIATE NEXT STEPS:")
    if "Multi-Layer Search" in working_modes:
        print("   1. Use Multi-Layer Search mode in the enhanced UI")
        print("   2. Access enhanced UI: http://127.0.0.1:8088/static/index_enhanced.html")
    
    if not any("main index" in mode.lower() for mode in working_modes):
        print("   3. Wait for main index build to complete, then test Basic/Advanced modes")
    
    print(f"\n💾 Detailed results saved to search_mode_test_results.json")