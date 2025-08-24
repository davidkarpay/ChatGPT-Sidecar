# Chat Features Testing Guide

This document describes the comprehensive testing suite for Sidecar's chat functionality, covering all aspects from unit tests to performance validation.

## 🧪 Test Suite Overview

The chat testing suite consists of 10 main test categories covering every aspect of the GPT-J integration:

### Test Categories

1. **Unit Tests** (`test_chat_unit.py`) - Core component testing
2. **Integration Tests** (`test_chat_integration.py`) - API endpoint testing  
3. **Performance Tests** (`test_chat_performance.py`) - Response time and memory
4. **Edge Case Tests** (`test_chat_edge_cases.py`) - Error handling and extremes
5. **API Contract Tests** (`test_chat_api_contracts.py`) - Request/response validation
6. **Search Tests** (`test_chat_search.py`) - RAG pipeline and search modes
7. **Conversation Tests** (`test_chat_conversations.py`) - Session management
8. **Model Tests** (`test_chat_models.py`) - Model loading and generation
9. **GPT-J Validation Tests** (`test_chat_gptj_validation.py`) - Download and environment validation
10. **Mock Infrastructure** (`fixtures/`) - Test data and mocks
11. **CI Integration** (`.github/workflows/`) - Automated testing

## 🚀 Quick Start

### Run All Tests
```bash
# Simple test run
python -m pytest tests/test_chat_*.py

# With coverage
python tests/run_tests.py all --coverage

# Verbose with report generation  
python tests/run_tests.py all --verbose --report
```

### Run Specific Test Categories
```bash
# Unit tests only
python tests/run_tests.py unit

# Integration tests
python tests/run_tests.py integration

# Performance tests
python tests/run_tests.py performance

# Fast tests only (exclude slow ones)
python tests/run_tests.py fast

# GPT-J environment validation
python tests/run_tests.py environment

# GPT-J download validation (slow)
python tests/run_tests.py download
```

### CI Mode (Complete Validation)
```bash
python tests/run_tests.py --ci
```

## 📋 Test Categories Detail

### 1. Unit Tests (`test_chat_unit.py`)

Tests core components in isolation:

- **ChatConfig**: Configuration validation, defaults, device detection
- **ChatAgent**: Model loading, prompt building, conversation history
- **RAGPipeline**: Search modes, context retrieval, result enrichment

**Coverage**: 45+ test methods  
**Focus**: Component behavior without external dependencies

```bash
pytest tests/test_chat_unit.py -v
```

### 2. Integration Tests (`test_chat_integration.py`)

Tests complete API workflows:

- **Chat Endpoints**: `/chat`, `/chat/stream`, `/analyze`
- **Session Management**: History, clearing, persistence
- **Error Handling**: Authentication, validation, internal errors
- **Concurrent Requests**: Multiple simultaneous chat sessions

**Coverage**: 25+ test methods  
**Focus**: End-to-end API functionality

```bash
pytest tests/test_chat_integration.py -v
```

### 3. Performance Tests (`test_chat_performance.py`)

Validates response times and resource usage:

- **Response Times**: Chat, streaming, concurrent load
- **Memory Usage**: Peak memory, leak detection, cleanup
- **Throughput**: Requests per second, sustained load
- **Scalability**: Multiple sessions, large context

**Coverage**: 10+ test methods  
**Focus**: Performance characteristics and limits

```bash
pytest tests/test_chat_performance.py -v -s
```

### 4. Edge Case Tests (`test_chat_edge_cases.py`)

Tests extreme inputs and failure scenarios:

- **Extreme Inputs**: Very long queries, Unicode, special characters
- **Malformed Requests**: Invalid JSON, missing fields, wrong types
- **System Failures**: Model errors, database failures, memory issues
- **Security**: Injection attempts, path traversal, XSS

**Coverage**: 20+ test methods  
**Focus**: Robustness and security

```bash
pytest tests/test_chat_edge_cases.py -v
```

### 5. API Contract Tests (`test_chat_api_contracts.py`)

Validates request/response schemas:

- **Request Validation**: Pydantic model validation, parameter ranges
- **Response Schemas**: Correct structure, types, required fields
- **Error Responses**: Proper HTTP status codes, error formats
- **Content Types**: JSON, SSE streaming, error handling

**Coverage**: 15+ test methods  
**Focus**: API contract compliance

```bash
pytest tests/test_chat_api_contracts.py -v
```

### 6. Search Functionality Tests (`test_chat_search.py`)

Tests RAG pipeline and search components:

- **Search Modes**: Adaptive, multi-layer, basic search
- **Context Retrieval**: Relevance ranking, diversity (MMR)
- **Fallback Behavior**: Missing stores, empty results
- **FAISS Integration**: Vector search, index operations

**Coverage**: 25+ test methods  
**Focus**: Search accuracy and reliability

```bash
pytest tests/test_chat_search.py -v
```

### 7. Conversation Management Tests (`test_chat_conversations.py`)

Tests session and conversation handling:

- **Session Management**: Creation, persistence, isolation
- **History Management**: Storage, retrieval, clearing
- **Context Building**: Prompt construction with history
- **Multi-turn Conversations**: Context continuity, memory

**Coverage**: 20+ test methods  
**Focus**: Conversation state management

```bash
pytest tests/test_chat_conversations.py -v
```

### 8. Model Integration Tests (`test_chat_models.py`)

Tests model loading and generation:

- **Model Configuration**: Device detection, quantization settings
- **Model Loading**: Different architectures, error handling
- **Generation**: Parameter application, streaming, memory management
- **Vector Store**: Embedding models, FAISS operations

**Coverage**: 15+ test methods  
**Focus**: Model integration and optimization

```bash
pytest tests/test_chat_models.py -v
```

### 9. GPT-J Validation Tests (`test_chat_gptj_validation.py`)

Tests GPT-J model download and environment validation:

- **Environment Validation**: Disk space, memory, connectivity, dependencies
- **Download Testing**: Tokenizer download, model download (optional)
- **Post-Download Validation**: Model loading, generation, quantization
- **Failure Scenarios**: Network issues, insufficient resources, corruption
- **Real Integration**: End-to-end testing with actual GPT-J model

**Coverage**: 25+ test methods  
**Focus**: Production readiness and download validation

```bash
# Environment validation (fast)
pytest tests/test_chat_gptj_validation.py::TestPreDownloadEnvironment -v

# Download validation (slow)
pytest tests/test_chat_gptj_validation.py -m "download" -v

# All GPT-J validation tests
pytest tests/test_chat_gptj_validation.py -v
```

## 🏗️ Test Infrastructure

### Mock Components (`tests/fixtures/`)

- **MockChatAgent**: Lightweight agent for fast testing
- **MockRAGPipeline**: Complete pipeline mock with realistic behavior
- **MockFaissStore**: Vector store simulation
- **Sample Data**: Conversations, embeddings, search results

### Global Fixtures (`conftest.py`)

- **Environment Setup**: Test configurations, cleanup
- **Authentication**: Test headers, invalid credentials
- **Performance Monitoring**: Memory tracking, timing utilities
- **Mock Auto-Loading**: Automatic mocking for isolation

### Test Configuration (`pytest.ini`)

- **Coverage Requirements**: 80% minimum coverage
- **Test Markers**: Categorization for selective running
- **Output Formatting**: Detailed reports, HTML coverage
- **Timeout Protection**: Prevents hanging tests

## 📊 Coverage Reports

### Generate Coverage Report
```bash
python -m pytest tests/test_chat_*.py --cov=app --cov-report=html
open htmlcov/index.html
```

### Coverage Targets

| Component | Target Coverage | Current Status |
|-----------|----------------|----------------|
| chat_agent.py | 90% | ✅ |
| rag_pipeline.py | 85% | ✅ |
| main.py (chat endpoints) | 80% | ✅ |
| Overall chat features | 80% | ✅ |

## 🔄 Continuous Integration

### GitHub Actions Workflow

The test suite runs automatically on:
- **Push** to main/develop branches
- **Pull requests** affecting chat components
- **Schedule** (nightly comprehensive tests)

### Test Matrix

- **Python Versions**: 3.9, 3.10, 3.11
- **Test Categories**: Parallel execution by category
- **Performance Validation**: Baseline comparisons
- **Coverage Enforcement**: Minimum thresholds

### Artifacts

CI generates and stores:
- **Test Reports**: HTML summaries with details
- **Coverage Reports**: Line-by-line coverage analysis
- **Performance Metrics**: Response time trends
- **Failure Analysis**: Detailed error information

## 🛠️ Development Workflow

### Before Committing
```bash
# Run fast tests during development
python tests/run_tests.py fast --verbose

# Run full suite before major commits
python tests/run_tests.py --ci
```

### Adding New Tests

1. **Identify Test Category**: Which file should contain your test?
2. **Follow Naming Convention**: `test_description_of_functionality`
3. **Use Appropriate Fixtures**: Leverage existing mocks and setup
4. **Add Test Markers**: Categorize appropriately
5. **Update Documentation**: Add to this guide if needed

### Test Best Practices

- **Isolation**: Each test should be independent
- **Mocking**: Mock external dependencies (models, databases)
- **Assertions**: Clear, specific assertions with good error messages
- **Performance**: Mark slow tests, provide alternatives
- **Documentation**: Clear docstrings explaining test purpose

## 🚨 Troubleshooting

### Common Issues

#### Tests Fail Due to Missing Dependencies
```bash
pip install -r requirements.txt
pip install pytest-cov pytest-html pytest-xdist pytest-timeout pytest-mock
```

#### Server Not Available for Integration Tests
```bash
# Start server in background
uvicorn app.main:app --host 127.0.0.1 --port 8088 &

# Run integration tests
python tests/run_tests.py integration

# Stop server
pkill -f uvicorn
```

#### Memory Issues in Performance Tests
```bash
# Run with limited scope
python -m pytest tests/test_chat_performance.py -k "not memory" -v
```

#### Model Loading Timeouts
```bash
# Skip slow model tests
python -m pytest tests/test_chat_models.py -m "not slow" -v
```

### Debug Mode

Run tests with maximum verbosity and no capture:
```bash
python -m pytest tests/test_chat_unit.py::TestChatConfig::test_default_config -v -s --tb=long
```

### Test Data Reset

Clear test artifacts:
```bash
rm -rf htmlcov/ .coverage test_report.html coverage.xml .pytest_cache/
```

## 📈 Performance Baselines

### Response Time Targets

- **Simple Chat**: < 2 seconds
- **Streaming Chat**: < 1 second to first token
- **Complex Search**: < 3 seconds
- **Concurrent Requests**: 90% success rate

### Memory Usage Targets

- **Baseline**: < 500MB increase during tests
- **Peak Usage**: < 2GB total
- **Leak Detection**: < 20% increase after test suite

### Throughput Targets

- **Serial Requests**: > 1 request/second
- **Concurrent Load**: > 5 requests/second with 5 workers
- **Sustained Load**: 30+ minutes without degradation

## 🎯 Test Metrics

The test suite tracks:

- **Code Coverage**: Line and branch coverage
- **Test Count**: Number of tests per category
- **Execution Time**: Test duration and trends
- **Success Rate**: Pass/fail rates over time
- **Performance**: Response times and resource usage

## 📝 Contributing

When contributing chat functionality:

1. **Write Tests First**: Test-driven development encouraged
2. **Maintain Coverage**: Don't decrease overall coverage
3. **Add Performance Tests**: For user-facing features
4. **Update Documentation**: Keep this guide current
5. **Test Edge Cases**: Consider failure scenarios

---

**Happy Testing! 🧪✨**

For questions about the test suite, see the test files themselves or contact the development team.