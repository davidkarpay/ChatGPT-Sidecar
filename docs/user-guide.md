# Sidecar Search User Guide

Welcome to Sidecar, your local-first context archive with semantic search capabilities. This guide will help you make the most of your document search experience.

## Getting Started

### Accessing the Search Interface

1. **Start the server**: `./scripts/run_dev.sh`
2. **Open your browser**: Navigate to `http://127.0.0.1:8088/static/index_enhanced.html`
3. **Enter API key**: Use the provided API key (stored automatically)

### Initial Setup

The enhanced search interface provides four powerful search modes to help you find information effectively.

## Search Interface Overview

### API Key Configuration
- **Auto-saved**: Your API key is saved automatically in browser storage
- **Required**: All search requests require authentication
- **Default key**: `sidecar-AfreWVOEVoCtXzMT0jejgTqsng4J-kwlICBQonyMbas`

### Search Controls

#### Query Input
- **Main search box**: Enter your search terms here
- **Results count (k)**: Set how many results to return (1-30, default: 8)
- **Real-time search**: Press Enter or click Search

#### Search Mode Selection

**🔍 Basic Search**
- Fast keyword-based search
- Uses main FAISS index
- Best for: Quick lookups, known terms
- Response time: ~20-200ms

**⚡ Advanced MMR**
- Maximal Marginal Relevance algorithm
- Balances relevance with diversity
- **Candidates**: Number of initial results (10-200, default: 50)
- **Lambda**: Relevance vs diversity (0.0-1.0, default: 0.5)
  - Higher lambda = more relevant, potentially similar results
  - Lower lambda = more diverse, potentially broader results

**🔄 Multi-Layer**
- Searches across precision, balanced, and context layers
- Combines different chunk sizes for comprehensive results
- Best for: Complex topics, research

**📋 Specific Layer**
- Search individual layers by chunk size
- **Precision**: Short chunks (detailed facts)
- **Balanced**: Medium chunks (standard paragraphs)  
- **Context**: Long chunks (full context)

## Search Results

### Result Display
Each search result shows:
- **Title/Source**: Document or conversation title
- **Preview**: Relevant text excerpt with highlighted search terms
- **Metadata**: Document ID, chunk ID, character positions
- **Score**: Relevance score (higher = more relevant)

### Result Actions
- **Copy Context**: Copies structured context data for use with ChatGPT
- **Term Highlighting**: Search terms are automatically highlighted
- **XSS Protection**: All content is safely rendered

### Multi-Layer Results
Multi-layer search additionally shows:
- **Layer weights**: Percentage contribution from each layer
- **Combined scoring**: Weighted relevance across layers

## Advanced Features

### MMR Parameter Tuning

**Lambda Values Guide:**
- `λ = 1.0`: Maximum relevance (similar results)
- `λ = 0.7`: High relevance with some diversity (recommended for focused research)
- `λ = 0.5`: Balanced relevance and diversity (default, good for exploration)
- `λ = 0.3`: High diversity with some relevance (broad research)
- `λ = 0.0`: Maximum diversity (discovery mode)

**Candidates Parameter:**
- **10-20**: Fast, focused results
- **50**: Default, good balance
- **100+**: Comprehensive but slower

### Layer-Specific Search

**When to use each layer:**

**Precision Layer** (Short chunks ~300-600 chars)
- Specific facts, definitions
- Technical details, specifications
- Quick reference information
- Example: "contract termination notice period"

**Balanced Layer** (Medium chunks ~600-1200 chars)
- Standard paragraphs, explanations
- Procedural information
- General research
- Example: "employment law compliance requirements"

**Context Layer** (Long chunks ~1200+ chars)
- Full context, comprehensive explanations
- Complex topics requiring background
- Multi-faceted information
- Example: "constitutional law interpretation methodology"

## Common Use Cases

### Legal Research
```
Query: "constitutional fourth amendment search seizure"
Mode: Advanced MMR (λ=0.6)
Purpose: Find diverse case law and precedents
```

### Technical Documentation
```
Query: "API authentication JWT token"
Mode: Basic Search
Purpose: Quick reference lookup
```

### Policy Analysis
```
Query: "data privacy GDPR compliance requirements"
Mode: Multi-Layer
Purpose: Comprehensive understanding across detail levels
```

### Contract Review
```
Query: "indemnification liability insurance coverage"
Mode: Advanced MMR (λ=0.5)
Purpose: Find related clauses and precedents
```

## Tips for Better Results

### Query Optimization
1. **Use specific terminology** relevant to your document collection
2. **Include 3-5 keywords** for best balance
3. **Try synonyms** if initial results are poor
4. **Start broad, then narrow** your search terms

### Mode Selection
1. **Basic**: When you know exactly what you're looking for
2. **Advanced MMR**: When researching a topic comprehensively
3. **Multi-Layer**: When you need different perspectives on a topic
4. **Specific Layer**: When you need a particular level of detail

### Result Evaluation
- **Check multiple modes** for comprehensive coverage
- **Adjust lambda** based on result diversity needs
- **Use Copy Context** to integrate with AI assistants
- **Note effective search terms** from good results for future queries

## Troubleshooting

### No Results
1. Check spelling and terminology
2. Try broader, more general terms
3. Switch to Multi-Layer search
4. Verify API key is entered correctly

### Irrelevant Results
1. Add more specific keywords
2. Increase lambda in Advanced MMR mode
3. Try a different search layer
4. Include domain-specific terminology

### Slow Performance
1. Reduce candidates parameter in Advanced MMR
2. Use Basic search for simple queries
3. Consider if index needs rebuilding
4. Check server logs for issues

### Similar Results
1. Use Advanced MMR with lower lambda (0.3-0.4)
2. Try Multi-Layer search
3. Add broader context terms to query
4. Increase candidates parameter

## Keyboard Shortcuts

- **Enter**: Execute search
- **Escape**: Clear current search (when implemented)

## Integration with AI Assistants

### Copy Context Feature
Each result includes a "Copy Context" button that copies structured data:

```json
{
  "context": [
    {
      "doc": "Document Title",
      "loc": {
        "doc_id": 123,
        "chunk_id": 456,
        "start": 1000,
        "end": 1400
      },
      "quote": "Relevant text excerpt..."
    }
  ]
}
```

This format is optimized for use with ChatGPT and other AI assistants for follow-up analysis.

## Performance Expectations

| Search Mode | Typical Response Time | Best For |
|-------------|----------------------|----------|
| Basic | 20-200ms | Quick lookups |
| Advanced MMR | 150-500ms | Research |
| Multi-Layer | 500-1000ms | Comprehensive analysis |
| Specific Layer | 50-200ms | Targeted detail level |

## Data Privacy

- **Local-first**: All searches are performed locally
- **No tracking**: Search queries are not logged or transmitted
- **Secure**: API key authentication protects your data
- **Private**: Your documents remain on your local system

---

For query writing strategies and advanced techniques, see [Writing Effective Search Queries](effective-search-queries.md).