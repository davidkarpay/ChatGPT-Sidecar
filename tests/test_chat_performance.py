"""
Performance tests for chat functionality
"""
import pytest
import time
import psutil
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient
import requests
import json

from app.main import app


class TestChatPerformance:
    """Performance tests for chat functionality"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        return {"X-API-Key": "change-me"}
    
    @pytest.fixture
    def mock_rag_pipeline(self):
        with patch('app.main.get_rag_pipeline') as mock_get_pipeline:
            mock_pipeline = Mock()
            mock_get_pipeline.return_value = mock_pipeline
            yield mock_pipeline
    
    def test_chat_response_time(self, client, auth_headers, mock_rag_pipeline):
        """Test chat response time under normal conditions"""
        # Setup mock with realistic delay
        def mock_chat(*args, **kwargs):
            time.sleep(0.1)  # Simulate model processing time
            return {
                "session_id": "test",
                "response": "Test response",
                "context": [],
                "query": kwargs.get("query", "test")
            }
        
        mock_rag_pipeline.chat.side_effect = mock_chat
        mock_rag_pipeline.suggest_follow_up_questions.return_value = []
        
        # Measure response time
        start_time = time.time()
        response = client.post(
            "/chat",
            json={"query": "What is machine learning?"},
            headers=auth_headers
        )
        end_time = time.time()
        
        response_time = end_time - start_time
        
        assert response.status_code == 200
        assert response_time < 2.0  # Should respond within 2 seconds
        print(f"Chat response time: {response_time:.3f}s")
    
    def test_concurrent_chat_performance(self, auth_headers, mock_rag_pipeline):
        """Test performance under concurrent load"""
        # Use requests instead of TestClient for true concurrency
        base_url = "http://127.0.0.1:8088"
        
        def mock_chat(*args, **kwargs):
            time.sleep(0.05)  # Shorter delay for load testing
            return {
                "session_id": f"session-{threading.current_thread().ident}",
                "response": f"Response to: {kwargs.get('query', 'test')}",
                "context": [],
                "query": kwargs.get("query", "test")
            }
        
        mock_rag_pipeline.chat.side_effect = mock_chat
        mock_rag_pipeline.suggest_follow_up_questions.return_value = []
        
        num_requests = 10
        concurrent_limit = 5
        
        def make_request(request_id):
            start_time = time.time()
            try:
                response = requests.post(
                    f"{base_url}/chat",
                    json={
                        "query": f"Test query {request_id}",
                        "session_id": f"load-test-{request_id}"
                    },
                    headers=auth_headers,
                    timeout=10
                )
                end_time = time.time()
                return {
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "response_time": end_time - start_time,
                    "success": response.status_code == 200
                }
            except Exception as e:
                end_time = time.time()
                return {
                    "request_id": request_id,
                    "status_code": 0,
                    "response_time": end_time - start_time,
                    "success": False,
                    "error": str(e)
                }
        
        # Execute concurrent requests
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=concurrent_limit) as executor:
            futures = [executor.submit(make_request, i) for i in range(num_requests)]
            results = [future.result() for future in as_completed(futures)]
        end_time = time.time()
        
        total_time = end_time - start_time
        successful_requests = sum(1 for r in results if r["success"])
        avg_response_time = sum(r["response_time"] for r in results) / len(results)
        
        print(f"Concurrent test results:")
        print(f"  Total requests: {num_requests}")
        print(f"  Successful: {successful_requests}")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Average response time: {avg_response_time:.3f}s")
        print(f"  Requests per second: {successful_requests / total_time:.2f}")
        
        # Performance assertions
        assert successful_requests >= num_requests * 0.9  # 90% success rate
        assert avg_response_time < 1.0  # Average under 1 second
        assert total_time < 5.0  # Complete within 5 seconds
    
    def test_memory_usage_during_chat(self, client, auth_headers, mock_rag_pipeline):
        """Test memory usage during chat operations"""
        process = psutil.Process()
        
        # Measure baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        def mock_chat_with_large_context(*args, **kwargs):
            # Simulate large context processing
            large_context = [
                {
                    "source": f"Document {i}",
                    "preview": "This is a large context preview " * 50,
                    "text": "Full text content " * 100
                }
                for i in range(20)
            ]
            return {
                "session_id": "memory-test",
                "response": "Response with large context",
                "context": large_context,
                "query": kwargs.get("query", "test")
            }
        
        mock_rag_pipeline.chat.side_effect = mock_chat_with_large_context
        mock_rag_pipeline.suggest_follow_up_questions.return_value = [
            "Follow-up question " * 10 for _ in range(10)
        ]
        
        memory_measurements = []
        
        # Make multiple requests and measure memory
        for i in range(10):
            response = client.post(
                "/chat",
                json={"query": f"Memory test query {i}"},
                headers=auth_headers
            )
            
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_measurements.append(current_memory)
            
            assert response.status_code == 200
            
            # Small delay between requests
            time.sleep(0.1)
        
        peak_memory = max(memory_measurements)
        memory_increase = peak_memory - baseline_memory
        
        print(f"Memory usage:")
        print(f"  Baseline: {baseline_memory:.1f} MB")
        print(f"  Peak: {peak_memory:.1f} MB")
        print(f"  Increase: {memory_increase:.1f} MB")
        
        # Memory assertions (adjust based on your system)
        assert memory_increase < 500  # Should not increase by more than 500MB
        
        # Check for memory leaks by comparing end vs baseline
        final_memory = process.memory_info().rss / 1024 / 1024
        leak_threshold = baseline_memory * 1.2  # 20% increase threshold
        assert final_memory < leak_threshold, f"Potential memory leak: {final_memory:.1f}MB vs {baseline_memory:.1f}MB baseline"
    
    def test_streaming_performance(self, auth_headers, mock_rag_pipeline):
        """Test streaming response performance"""
        base_url = "http://127.0.0.1:8088"
        
        # Mock streaming response
        def mock_streaming_tokens():
            tokens = ["Hello", " ", "this", " ", "is", " ", "a", " ", "test", " ", "response", "!"]
            for token in tokens:
                time.sleep(0.01)  # Simulate token generation delay
                yield token
        
        mock_rag_pipeline.chat.return_value = {
            "session_id": "stream-test",
            "context": [{"source": "Test", "preview": "Test content"}],
            "response_generator": mock_streaming_tokens()
        }
        mock_rag_pipeline.suggest_follow_up_questions.return_value = ["Follow-up?"]
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{base_url}/chat/stream",
                json={"query": "Test streaming query"},
                headers=auth_headers,
                stream=True,
                timeout=10
            )
            
            first_byte_time = None
            tokens_received = 0
            
            for line in response.iter_lines():
                if first_byte_time is None:
                    first_byte_time = time.time()
                
                if line.startswith(b'data: '):
                    try:
                        data = json.loads(line[6:])
                        if data.get("type") == "token":
                            tokens_received += 1
                    except json.JSONDecodeError:
                        continue
            
            end_time = time.time()
            
            total_time = end_time - start_time
            time_to_first_byte = first_byte_time - start_time if first_byte_time else total_time
            
            print(f"Streaming performance:")
            print(f"  Total time: {total_time:.3f}s")
            print(f"  Time to first byte: {time_to_first_byte:.3f}s")
            print(f"  Tokens received: {tokens_received}")
            
            # Performance assertions
            assert response.status_code == 200
            assert time_to_first_byte < 1.0  # First byte within 1 second
            assert total_time < 5.0  # Complete within 5 seconds
            assert tokens_received > 0  # Should receive some tokens
            
        except requests.exceptions.RequestException as e:
            pytest.skip(f"Server not available for streaming test: {e}")
    
    def test_large_query_performance(self, client, auth_headers, mock_rag_pipeline):
        """Test performance with large queries"""
        # Generate large query (simulating very detailed questions)
        large_query = "Explain in detail " + "the concept of machine learning " * 100
        
        mock_rag_pipeline.chat.return_value = {
            "session_id": "large-query-test",
            "response": "Response to large query",
            "context": [],
            "query": large_query
        }
        mock_rag_pipeline.suggest_follow_up_questions.return_value = []
        
        start_time = time.time()
        response = client.post(
            "/chat",
            json={"query": large_query},
            headers=auth_headers
        )
        end_time = time.time()
        
        response_time = end_time - start_time
        
        assert response.status_code == 200
        assert response_time < 3.0  # Should handle large queries within 3 seconds
        print(f"Large query response time: {response_time:.3f}s")
    
    def test_context_retrieval_performance(self, client, auth_headers, mock_rag_pipeline):
        """Test performance with large context retrieval"""
        # Simulate retrieval of many context chunks
        large_context = [
            {
                "rank": i + 1,
                "score": 0.9 - (i * 0.01),
                "source": f"Document {i}",
                "preview": f"This is preview text for document {i} " * 20,
                "text": f"Full content for document {i} " * 50,
                "doc_id": i,
                "chunk_id": i * 10,
                "start_char": 0,
                "end_char": 1000
            }
            for i in range(20)  # Simulate 20 context chunks
        ]
        
        mock_rag_pipeline.chat.return_value = {
            "session_id": "context-test",
            "response": "Response with large context",
            "context": large_context,
            "query": "Test query"
        }
        mock_rag_pipeline.suggest_follow_up_questions.return_value = []
        
        start_time = time.time()
        response = client.post(
            "/chat",
            json={"query": "Find comprehensive information about AI", "k": 20},
            headers=auth_headers
        )
        end_time = time.time()
        
        response_time = end_time - start_time
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["context"]) == 20
        assert response_time < 2.0  # Should handle large context within 2 seconds
        print(f"Large context response time: {response_time:.3f}s")
    
    @pytest.mark.slow
    def test_sustained_load_performance(self, auth_headers, mock_rag_pipeline):
        """Test performance under sustained load (marked as slow test)"""
        base_url = "http://127.0.0.1:8088"
        
        mock_rag_pipeline.chat.side_effect = lambda *args, **kwargs: {
            "session_id": f"sustained-{threading.current_thread().ident}",
            "response": "Sustained load response",
            "context": [],
            "query": kwargs.get("query", "test")
        }
        mock_rag_pipeline.suggest_follow_up_questions.return_value = []
        
        duration = 30  # Run for 30 seconds
        request_interval = 0.5  # Request every 0.5 seconds
        
        start_time = time.time()
        results = []
        
        while time.time() - start_time < duration:
            request_start = time.time()
            
            try:
                response = requests.post(
                    f"{base_url}/chat",
                    json={"query": f"Sustained load test at {time.time()}"},
                    headers=auth_headers,
                    timeout=5
                )
                
                request_end = time.time()
                
                results.append({
                    "timestamp": request_start,
                    "response_time": request_end - request_start,
                    "status_code": response.status_code,
                    "success": response.status_code == 200
                })
                
            except requests.exceptions.RequestException as e:
                request_end = time.time()
                results.append({
                    "timestamp": request_start,
                    "response_time": request_end - request_start,
                    "status_code": 0,
                    "success": False,
                    "error": str(e)
                })
            
            # Wait for next request interval
            elapsed = time.time() - request_start
            if elapsed < request_interval:
                time.sleep(request_interval - elapsed)
        
        # Analyze results
        total_requests = len(results)
        successful_requests = sum(1 for r in results if r["success"])
        success_rate = successful_requests / total_requests if total_requests > 0 else 0
        
        successful_results = [r for r in results if r["success"]]
        if successful_results:
            avg_response_time = sum(r["response_time"] for r in successful_results) / len(successful_results)
            max_response_time = max(r["response_time"] for r in successful_results)
            min_response_time = min(r["response_time"] for r in successful_results)
        else:
            avg_response_time = max_response_time = min_response_time = 0
        
        print(f"Sustained load test results ({duration}s):")
        print(f"  Total requests: {total_requests}")
        print(f"  Successful: {successful_requests}")
        print(f"  Success rate: {success_rate:.2%}")
        print(f"  Avg response time: {avg_response_time:.3f}s")
        print(f"  Min/Max response time: {min_response_time:.3f}s / {max_response_time:.3f}s")
        
        # Performance assertions
        assert success_rate >= 0.95  # 95% success rate
        assert avg_response_time < 1.0  # Average response time under 1s
        assert max_response_time < 5.0  # No response should take more than 5s


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])