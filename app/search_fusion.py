from typing import List, Dict, Tuple
import os
from pathlib import Path
from .vectorstore import FaissStore
from .db import DB

class MultiLayerSearchFusion:
    """
    Search fusion algorithm for multi-layer chunking system.
    Combines results from precision, balanced, and context layers.
    """
    
    def __init__(self, db_path: str, embed_model: str):
        self.db_path = db_path
        self.embed_model = embed_model
        
        # Initialize stores for each layer
        self.stores = {}
        layer_configs = {
            'precision': {'index': 'data/indexes/precision.faiss', 'ids': 'data/indexes/precision.pkl'},
            'balanced': {'index': 'data/indexes/balanced.faiss', 'ids': 'data/indexes/balanced.pkl'},
            'context': {'index': 'data/indexes/context.faiss', 'ids': 'data/indexes/context.pkl'}
        }
        
        for layer_name, paths in layer_configs.items():
            store = FaissStore(Path(paths['index']), Path(paths['ids']), embed_model)
            store.load()
            self.stores[layer_name] = store
    
    def _analyze_query(self, query: str) -> Dict[str, float]:
        """
        Analyze query characteristics to determine layer weights.
        Returns weights for each layer based on query properties.
        """
        query_lower = query.lower()
        query_len = len(query.split())
        
        # Base weights
        weights = {'precision': 0.3, 'balanced': 0.5, 'context': 0.2}
        
        # Adjust based on query length
        if query_len <= 3:  # Short queries favor precision
            weights['precision'] += 0.2
            weights['context'] -= 0.1
        elif query_len >= 8:  # Long queries favor context
            weights['context'] += 0.2
            weights['precision'] -= 0.1
        
        # Technical terms favor precision layer
        technical_terms = ['error', 'function', 'class', 'method', 'variable', 'import', 'def ', 'return']
        if any(term in query_lower for term in technical_terms):
            weights['precision'] += 0.15
            weights['balanced'] -= 0.1
        
        # Conversational queries favor context
        conversational_terms = ['explain', 'how to', 'what is', 'why', 'can you', 'help me']
        if any(term in query_lower for term in conversational_terms):
            weights['context'] += 0.15
            weights['precision'] -= 0.1
        
        # Normalize weights to sum to 1.0
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}
    
    def _deduplicate_results(self, all_results: List[Dict]) -> List[Dict]:
        """
        Remove duplicate results based on chunk overlap and document similarity.
        Prioritizes higher-scored results from appropriate layers.
        """
        seen_docs = set()
        seen_text_hashes = set()
        deduplicated = []
        
        # Sort by weighted score (descending)
        all_results.sort(key=lambda x: x['weighted_score'], reverse=True)
        
        for result in all_results:
            doc_id = result['document']['id']
            text_snippet = result['snippet'][:100]  # First 100 chars for comparison
            text_hash = hash(text_snippet)
            
            # Skip if we've seen very similar text
            if text_hash in seen_text_hashes:
                continue
                
            # For same document, prefer different layers for diversity
            if doc_id in seen_docs:
                # Only include if it's from a different layer and significantly different content
                existing_layers = [r['layer'] for r in deduplicated if r['document']['id'] == doc_id]
                if result['layer'] in existing_layers:
                    continue
            
            seen_docs.add(doc_id)
            seen_text_hashes.add(text_hash)
            deduplicated.append(result)
        
        return deduplicated
    
    def search_multi_layer(self, query: str, k: int = 8) -> Dict:
        """
        Search across all layers and return fused results.
        """
        # Analyze query to determine layer weights
        layer_weights = self._analyze_query(query)
        
        # Search each layer
        all_results = []
        layer_results = {}
        
        for layer_name, store in self.stores.items():
            if store.index is None:
                layer_results[layer_name] = []
                continue
                
            # Get more results per layer to improve fusion quality
            layer_k = max(k // len(self.stores) + 2, 4)
            search_results = store.search(query, k=layer_k)
            layer_results[layer_name] = search_results
            
            if not search_results:
                continue
            
            # Get chunk data from database
            faiss_ids = [faiss_idx for faiss_idx, _ in search_results]
            with DB(self.db_path) as db:
                rows_by_fid = db.fetch_chunks_by_faiss_indices(faiss_ids, layer_name)
            
            # Process results for this layer
            for rank, (faiss_idx, score) in enumerate(search_results, start=1):
                r = rows_by_fid.get(faiss_idx)
                if not r:
                    continue
                
                # Calculate weighted score based on layer importance for this query
                layer_weight = layer_weights[layer_name]
                weighted_score = float(score) * layer_weight
                
                result = {
                    "layer": layer_name,
                    "layer_rank": rank,
                    "raw_score": float(score),
                    "layer_weight": layer_weight,
                    "weighted_score": weighted_score,
                    "chunk_id": r["chunk_id"],
                    "document": {
                        "id": r["doc_id"],
                        "title": r["title"],
                        "doc_type": r["doc_type"]
                    },
                    "snippet": r["text"],
                    "offset": {"start": r["start_char"], "end": r["end_char"]}
                }
                all_results.append(result)
        
        # Deduplicate and rank final results
        deduplicated = self._deduplicate_results(all_results)
        
        # Take top k results and add final ranking
        final_results = deduplicated[:k]
        for i, result in enumerate(final_results, start=1):
            result["rank"] = i
        
        return {
            "query": query,
            "layer_weights": layer_weights,
            "layer_results_count": {layer: len(results) for layer, results in layer_results.items()},
            "total_candidates": len(all_results),
            "hits": final_results
        }