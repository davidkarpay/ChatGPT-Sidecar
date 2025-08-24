# Sidecar Search Quick Reference

## 🚀 Getting Started
1. Start server: `./scripts/run_dev.sh`
2. Open: `http://127.0.0.1:8088/static/index_enhanced.html`
3. Enter API key: `sidecar-AfreWVOEVoCtXzMT0jejgTqsng4J-kwlICBQonyMbas`

## 🔍 Search Modes

| Mode | Speed | Use Case | Parameters |
|------|-------|----------|------------|
| **Basic** | ⚡ Fast | Quick lookups, known terms | Query, k |
| **Advanced MMR** | 🔄 Medium | Research, diverse results | Query, k, candidates, lambda |
| **Multi-Layer** | 🔄 Slower | Comprehensive analysis | Query, k |
| **Specific Layer** | ⚡ Fast | Targeted detail level | Query, k, layer |

## ⚙️ Parameter Guide

### Lambda (λ) - Relevance vs Diversity
- **1.0**: Maximum relevance (focused)
- **0.7**: High relevance + some diversity
- **0.5**: Balanced (default)
- **0.3**: High diversity + some relevance  
- **0.0**: Maximum diversity (broad)

### Candidates
- **10-20**: Fast, focused
- **50**: Default balance
- **100+**: Comprehensive

### Layers
- **Precision**: Short chunks (facts, details)
- **Balanced**: Medium chunks (paragraphs)
- **Context**: Long chunks (full context)

## 📝 Query Patterns

### ✅ Effective Queries
```
"contract breach damages remedies"          # Legal
"API authentication JWT token"              # Technical  
"data privacy GDPR compliance"              # Policy
"machine learning model training"           # Technical
"constitutional fourth amendment"           # Legal
```

### ❌ Less Effective
```
"law stuff"                    # Too vague
"a"                           # Too short
"the contract that was..."    # Too long/narrative
```

## 🎯 Quick Decision Tree

**Need fast, specific lookup?** → Basic Search

**Researching a topic?** → Advanced MMR (λ=0.5-0.7)

**Want diverse perspectives?** → Advanced MMR (λ=0.3-0.4)

**Need comprehensive coverage?** → Multi-Layer

**Want specific detail level?** → Specific Layer

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| No results | Broader terms, check spelling, try Multi-Layer |
| Irrelevant results | More specific terms, higher lambda |
| Too similar results | Lower lambda, Multi-Layer search |
| Slow performance | Reduce candidates, use Basic search |

## 📋 Common Lambda Settings

| Task Type | Recommended λ | Reasoning |
|-----------|---------------|-----------|
| Fact-finding | 0.7-0.8 | Need precise, relevant results |
| Research | 0.5-0.6 | Balance relevance and breadth |
| Exploration | 0.3-0.4 | Discover related concepts |
| Brainstorming | 0.0-0.2 | Maximum topic diversity |

## 🎪 Example Workflows

### Legal Research
1. Start: `"employment discrimination"` (Basic)
2. Explore: `"employment discrimination harassment"` (MMR λ=0.5)
3. Deep dive: `"workplace harassment investigation"` (Multi-Layer)

### Technical Debugging  
1. Start: `"database connection error"` (Basic)
2. Troubleshoot: `"database timeout connection pool"` (MMR λ=0.6)
3. Solutions: `"database performance optimization"` (Multi-Layer)

### Policy Analysis
1. Overview: `"data protection requirements"` (Multi-Layer)
2. Specific: `"GDPR article 25 privacy design"` (Basic)
3. Related: `"privacy impact assessment GDPR"` (MMR λ=0.4)

## 📊 Performance Expectations

- **Basic**: 20-200ms
- **Advanced MMR**: 150-500ms  
- **Multi-Layer**: 500-1000ms
- **Specific Layer**: 50-200ms

## 💡 Pro Tips

1. **Start broad, narrow down** based on results
2. **Use domain vocabulary** relevant to your documents
3. **Try multiple modes** for comprehensive coverage
4. **Copy Context** for AI assistant integration
5. **Note effective terms** from good results
6. **Experiment with lambda** for optimal diversity