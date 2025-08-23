# Writing Effective Search Queries in Sidecar

This guide helps you write better search queries to find relevant information in your document archive.

## Understanding Search Modes

### Basic Search
- **Use for**: Quick, straightforward searches
- **Best for**: Simple keyword matching
- **Response time**: ~20-200ms
- **Example**: "contract terms"

### Advanced MMR Search
- **Use for**: When you want diverse, non-redundant results
- **Best for**: Research and discovery
- **Features**: Balances relevance with diversity using MMR algorithm
- **Parameters**: 
  - `lambda` (0.0-1.0): 0.0 = max diversity, 1.0 = max relevance
  - `candidates`: Number of initial results to consider (default: 50)
- **Example**: "legal precedent" with lambda=0.7 for focused results

### Multi-Layer Search
- **Use for**: Comprehensive searches across different chunk sizes
- **Best for**: Complex topics requiring multiple perspectives
- **Features**: Combines precision (short), balanced (medium), and context (long) chunks
- **Response time**: ~500-1000ms
- **Example**: "constitutional law analysis"

## Query Writing Strategies

### 1. Keyword Selection

#### ✅ **DO:**
- Use specific, descriptive terms
- Include domain-specific vocabulary
- Try synonyms and related terms
- Use multiple relevant keywords

```
Good: "breach of contract damages remedies"
Good: "fourth amendment reasonable search"
Good: "machine learning model training"
```

#### ❌ **DON'T:**
- Use only common words
- Include unnecessary stopwords
- Make queries too short or too long

```
Avoid: "the law about stuff"
Avoid: "a"
Avoid: "When someone does something illegal in a contract situation what happens legally speaking"
```

### 2. Query Length Guidelines

| Query Length | Best For | Example |
|--------------|----------|---------|
| 1-2 words | Broad topic exploration | "bankruptcy", "AI ethics" |
| 3-5 words | Specific concepts | "contract breach damages", "neural network training" |
| 6-10 words | Complex topics | "constitutional fourth amendment search seizure" |
| 10+ words | Very specific scenarios | Quote matching, exact phrase finding |

### 3. Domain-Specific Strategies

#### Legal Documents
```
✅ Use legal terms: "habeas corpus", "due process", "summary judgment"
✅ Include case elements: "negligence duty breach causation damages"
✅ Specify jurisdiction: "federal court", "state law", "constitutional"
```

#### Technical Documentation
```
✅ Use technical vocabulary: "API endpoint", "database schema", "authentication"
✅ Include version info: "Python 3.8", "React hooks", "TensorFlow 2.0"
✅ Specify components: "frontend validation", "backend service", "middleware"
```

#### Business Documents
```
✅ Use business terms: "revenue recognition", "market analysis", "stakeholder"
✅ Include metrics: "ROI", "KPI", "quarterly results"
✅ Specify departments: "marketing strategy", "HR policy", "finance audit"
```

### 4. Advanced Query Techniques

#### Concept-Based Searching
Instead of exact keywords, describe the concept:
```
Instead of: "Section 501(c)(3)"
Try: "nonprofit tax exemption status"

Instead of: "Scrum methodology"
Try: "agile development sprint planning"
```

#### Multi-Faceted Queries
Combine different aspects of your topic:
```
"contract termination notice period employment"
"data privacy GDPR compliance requirements"
"merger acquisition due diligence process"
```

#### Problem-Solution Queries
Frame queries around problems or solutions:
```
"resolve payment dispute mediation"
"prevent data breach security measures"
"optimize database query performance"
```

## Search Mode Selection Guide

### Choose **Basic Search** when:
- ✅ You know specific keywords
- ✅ You want fast results
- ✅ Topic is straightforward
- ✅ You're doing initial exploration

### Choose **Advanced MMR** when:
- ✅ You want comprehensive coverage
- ✅ Topic has multiple aspects
- ✅ You need diverse perspectives
- ✅ Research requires breadth

#### MMR Lambda Tuning:
- **λ = 0.8-1.0**: Highly relevant, potentially similar results
- **λ = 0.5-0.7**: Balanced relevance and diversity (recommended)
- **λ = 0.0-0.4**: Maximum diversity, may include tangential results

### Choose **Multi-Layer** when:
- ✅ Topic spans multiple contexts
- ✅ You need both details and overview
- ✅ Working with complex documents
- ✅ Want to see different chunk sizes

## Query Examples by Use Case

### Research & Analysis
```
Basic: "constitutional interpretation methods"
Advanced MMR: "judicial review constitutional interpretation" (λ=0.6)
Multi-Layer: "supreme court constitutional law precedent"
```

### Troubleshooting
```
Basic: "database connection error"
Advanced MMR: "database connection timeout error resolution" (λ=0.7)
Multi-Layer: "database performance optimization troubleshooting"
```

### Policy & Compliance
```
Basic: "data protection requirements"
Advanced MMR: "GDPR data protection compliance requirements" (λ=0.5)
Multi-Layer: "privacy policy data protection legal compliance"
```

### Contract Review
```
Basic: "indemnification clause"
Advanced MMR: "indemnification liability insurance contract" (λ=0.6)
Multi-Layer: "contract terms indemnification liability coverage"
```

## Optimization Tips

### 1. Iterative Refinement
Start broad, then narrow:
```
1st query: "employment law"
2nd query: "employment discrimination harassment"
3rd query: "workplace harassment investigation procedure"
```

### 2. Use Search Results to Improve Queries
- Notice effective terms in good results
- Identify related concepts mentioned
- Refine based on document language

### 3. Experiment with Synonyms
```
Try variations:
- "agreement" vs "contract" vs "deal"
- "error" vs "bug" vs "issue" vs "problem"
- "analyze" vs "review" vs "examine" vs "evaluate"
```

### 4. Context Matters
Consider the document collection:
- Legal docs: Use formal legal terminology
- Technical docs: Include version numbers, specific technologies
- Business docs: Include relevant business metrics, timeframes

## Common Query Patterns

### Entity + Action + Context
```
"company acquire competitor regulatory"
"employee terminate contract notice"
"database backup restore procedure"
```

### Problem + Solution + Domain
```
"breach contract remedies commercial"
"memory leak debug application"
"tax optimization strategy corporate"
```

### Comparison Queries
```
"Python vs JavaScript performance"
"copyright vs trademark protection"
"lease vs purchase equipment financing"
```

## Troubleshooting Poor Results

### If you get no results:
1. ✅ Check spelling and terminology
2. ✅ Try broader, more general terms
3. ✅ Remove very specific details
4. ✅ Use synonyms

### If results are irrelevant:
1. ✅ Add more specific keywords
2. ✅ Use domain-specific terminology
3. ✅ Try Advanced MMR with higher lambda (0.7-0.8)
4. ✅ Include context words

### If results are too similar:
1. ✅ Use Advanced MMR with lower lambda (0.3-0.5)
2. ✅ Try Multi-Layer search
3. ✅ Add broader context terms
4. ✅ Include different aspects of the topic

## Quick Reference

| Goal | Recommended Mode | Lambda | Example Query |
|------|------------------|--------|---------------|
| Fast keyword lookup | Basic | N/A | "API documentation" |
| Comprehensive research | Advanced MMR | 0.5-0.6 | "climate change policy economics" |
| Diverse perspectives | Advanced MMR | 0.3-0.4 | "artificial intelligence ethics society" |
| Multi-context analysis | Multi-Layer | N/A | "contract negotiation strategy legal" |
| Exact concept match | Basic | N/A | "force majeure clause" |

---

**Pro Tip**: Start with Basic search for quick checks, use Advanced MMR for research, and Multi-Layer for comprehensive analysis. Adjust lambda based on whether you want focused (higher) or diverse (lower) results.