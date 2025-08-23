#!/usr/bin/env python3
"""
Rebuild main FAISS index by merging existing layer indexes and processing missing chunks.
This approach leverages existing embeddings to minimize processing time.
"""

import os
import sys
import time
import numpy as np
import sqlite3
import pickle
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.db import DB
from app.vectorstore import FaissStore

class MainIndexRebuilder:
    def __init__(self):
        self.db_path = "sidecar.db"
        self.index_dir = Path("data/indexes")
        self.embed_model = "sentence-transformers/all-MiniLM-L6-v2"
        self.model = SentenceTransformer(self.embed_model)
        
        # Main index paths
        self.main_faiss_path = self.index_dir / "main.faiss"
        self.main_pkl_path = self.index_dir / "main.pkl"
        
        # Layer index paths
        self.layers = ["precision", "balanced", "context"]
        
        print(f"🚀 Main Index Rebuilder")
        print(f"Database: {self.db_path}")
        print(f"Index directory: {self.index_dir}")
        print(f"Model: {self.embed_model}")
        
    def backup_existing_indexes(self):
        """Backup existing main index if it exists."""
        if self.main_faiss_path.exists():
            backup_path = self.main_faiss_path.with_suffix(".faiss.backup")
            os.rename(self.main_faiss_path, backup_path)
            print(f"   📦 Backed up existing main.faiss to {backup_path}")
            
        if self.main_pkl_path.exists():
            backup_path = self.main_pkl_path.with_suffix(".pkl.backup")
            os.rename(self.main_pkl_path, backup_path)
            print(f"   📦 Backed up existing main.pkl to {backup_path}")
    
    def load_layer_index(self, layer_name):
        """Load a layer index and return vectors, ids, and chunk mappings."""
        faiss_path = self.index_dir / f"{layer_name}.faiss"
        pkl_path = self.index_dir / f"{layer_name}.pkl"
        
        if not faiss_path.exists() or not pkl_path.exists():
            print(f"   ⚠️  Layer {layer_name} index files missing")
            return None, None, None
            
        # Load FAISS index
        index = faiss.read_index(str(faiss_path))
        
        # Load ID mappings
        with open(pkl_path, 'rb') as f:
            ids = pickle.load(f)
        
        # Extract all vectors
        vectors = np.zeros((index.ntotal, index.d), dtype=np.float32)
        index.reconstruct_n(0, index.ntotal, vectors)
        
        print(f"   ✅ Loaded {layer_name}: {index.ntotal:,} vectors, {index.d} dims")
        return vectors, ids, index.d
    
    def get_all_chunks_from_db(self):
        """Get all chunks from database."""
        with DB(self.db_path) as db:
            cursor = db.conn.execute("SELECT id, text FROM chunk ORDER BY id ASC")
            chunks = {row["id"]: row["text"] for row in cursor.fetchall()}
        
        print(f"   📊 Database contains {len(chunks):,} total chunks")
        return chunks
    
    def get_existing_embedding_refs(self):
        """Get chunks that already have embeddings in layer indexes."""
        existing_chunks = set()
        layer_mappings = {}
        
        with DB(self.db_path) as db:
            for layer in self.layers:
                cursor = db.conn.execute(
                    "SELECT chunk_id, faiss_id FROM embedding_ref WHERE index_name = ?",
                    (layer,)
                )
                layer_chunks = {row["chunk_id"]: row["faiss_id"] for row in cursor.fetchall()}
                layer_mappings[layer] = layer_chunks
                existing_chunks.update(layer_chunks.keys())
                print(f"   📋 {layer}: {len(layer_chunks):,} chunks")
        
        return existing_chunks, layer_mappings
    
    def merge_layer_indexes(self):
        """Merge all layer indexes into single main index."""
        print("\n🔧 Merging layer indexes...")
        
        all_vectors = []
        all_ids = []
        vector_dim = None
        chunk_to_main_faiss_id = {}
        main_faiss_id = 0
        
        for layer in self.layers:
            print(f"   Processing {layer} layer...")
            vectors, ids, dim = self.load_layer_index(layer)
            
            if vectors is None:
                continue
                
            if vector_dim is None:
                vector_dim = dim
            elif vector_dim != dim:
                print(f"   ❌ Dimension mismatch: {vector_dim} vs {dim}")
                continue
            
            # Add vectors to combined array
            all_vectors.append(vectors)
            
            # Map chunk_ids to new main index positions
            for i, (embedding_ref_id, chunk_id) in enumerate(ids):
                all_ids.append((main_faiss_id, chunk_id))
                chunk_to_main_faiss_id[chunk_id] = main_faiss_id
                main_faiss_id += 1
        
        if not all_vectors:
            print("   ❌ No layer indexes found to merge")
            return None, None, None
        
        # Concatenate all vectors
        merged_vectors = np.vstack(all_vectors)
        print(f"   ✅ Merged {len(all_vectors)} layers: {merged_vectors.shape[0]:,} total vectors")
        
        return merged_vectors, all_ids, chunk_to_main_faiss_id
    
    def process_missing_chunks(self, all_chunks, existing_chunks):
        """Process chunks that aren't in any layer index."""
        missing_chunks = set(all_chunks.keys()) - existing_chunks
        
        if not missing_chunks:
            print("   ✅ No missing chunks to process")
            return None, None
        
        print(f"   🔍 Processing {len(missing_chunks):,} missing chunks...")
        
        # Prepare missing chunk data
        missing_texts = [all_chunks[chunk_id] for chunk_id in sorted(missing_chunks)]
        
        # Generate embeddings
        print("   🧮 Generating embeddings...")
        embeddings = self.model.encode(
            missing_texts, 
            batch_size=64, 
            show_progress_bar=True, 
            normalize_embeddings=True
        )
        embeddings = np.asarray(embeddings, dtype="float32")
        
        # Create ID mappings
        missing_ids = [(i, chunk_id) for i, chunk_id in enumerate(sorted(missing_chunks))]
        
        print(f"   ✅ Generated {embeddings.shape[0]:,} embeddings")
        return embeddings, missing_ids
    
    def build_main_index(self, merged_vectors, merged_ids, missing_vectors, missing_ids):
        """Build the main FAISS index."""
        print("\n🏗️  Building main FAISS index...")
        
        # Combine all vectors
        if missing_vectors is not None:
            all_vectors = np.vstack([merged_vectors, missing_vectors])
            all_ids = merged_ids + [(len(merged_ids) + i, chunk_id) for i, chunk_id in missing_ids]
        else:
            all_vectors = merged_vectors
            all_ids = merged_ids
        
        # Create FAISS index
        dim = all_vectors.shape[1]
        index = faiss.IndexFlatIP(dim)  # Cosine similarity via normalized dot product
        index.add(all_vectors)
        
        print(f"   ✅ Created main index: {index.ntotal:,} vectors, {dim} dimensions")
        
        # Save index
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.main_faiss_path))
        
        # Save ID mappings  
        with open(self.main_pkl_path, 'wb') as f:
            pickle.dump(all_ids, f)
        
        print(f"   💾 Saved main index to {self.main_faiss_path}")
        print(f"   💾 Saved ID mappings to {self.main_pkl_path}")
        
        return index, all_ids
    
    def update_embedding_refs(self, all_ids):
        """Update database with main index embedding references."""
        print("\n📝 Updating database embedding references...")
        
        with DB(self.db_path) as db:
            # Clear existing main index refs
            db.conn.execute("DELETE FROM embedding_ref WHERE index_name = 'main'")
            
            # Insert new refs in batches
            batch_size = 1000
            vector_dim = self.model.get_sentence_embedding_dimension()
            
            for i in range(0, len(all_ids), batch_size):
                batch = all_ids[i:i + batch_size]
                refs_data = [
                    (chunk_id, "main", vector_dim, faiss_id)
                    for faiss_id, chunk_id in batch
                ]
                
                db.conn.executemany(
                    "INSERT INTO embedding_ref (chunk_id, index_name, vector_dim, faiss_id) VALUES (?, ?, ?, ?)",
                    refs_data
                )
                
                if (i + batch_size) % 10000 == 0:
                    print(f"   📊 Processed {i + batch_size:,} references...")
        
        print(f"   ✅ Updated {len(all_ids):,} embedding references")
    
    def validate_index(self):
        """Validate the rebuilt main index."""
        print("\n🔍 Validating rebuilt index...")
        
        # Load and test index
        try:
            store = FaissStore(self.main_faiss_path, self.main_pkl_path, self.embed_model)
            store.load()
            
            if store.index is None:
                print("   ❌ Failed to load main index")
                return False
            
            # Test search
            results = store.search("test query", k=5)
            print(f"   ✅ Index loaded successfully: {store.index.ntotal:,} vectors")
            print(f"   ✅ Test search returned {len(results)} results")
            
            # Verify database consistency
            with DB(self.db_path) as db:
                cursor = db.conn.execute(
                    "SELECT COUNT(*) as count FROM embedding_ref WHERE index_name = 'main'"
                )
                ref_count = cursor.fetchone()["count"]
                
                if ref_count == store.index.ntotal:
                    print(f"   ✅ Database refs match index size: {ref_count:,}")
                    return True
                else:
                    print(f"   ❌ Mismatch: {ref_count:,} refs vs {store.index.ntotal:,} vectors")
                    return False
                    
        except Exception as e:
            print(f"   ❌ Validation failed: {e}")
            return False
    
    def run(self):
        """Execute the complete rebuild process."""
        start_time = time.time()
        
        try:
            # Step 1: Backup existing indexes
            print("\n📦 Backing up existing indexes...")
            self.backup_existing_indexes()
            
            # Step 2: Load all chunks from database
            print("\n📊 Analyzing database...")
            all_chunks = self.get_all_chunks_from_db()
            existing_chunks, layer_mappings = self.get_existing_embedding_refs()
            
            print(f"   📈 Coverage: {len(existing_chunks):,}/{len(all_chunks):,} chunks ({100*len(existing_chunks)/len(all_chunks):.1f}%)")
            
            # Step 3: Merge layer indexes
            merged_vectors, merged_ids, chunk_mapping = self.merge_layer_indexes()
            if merged_vectors is None:
                print("❌ Failed to merge layer indexes")
                return False
            
            # Step 4: Process missing chunks
            print("\n🔍 Processing missing chunks...")
            missing_vectors, missing_ids = self.process_missing_chunks(all_chunks, existing_chunks)
            
            # Step 5: Build main index
            index, all_ids = self.build_main_index(merged_vectors, merged_ids, missing_vectors, missing_ids)
            
            # Step 6: Update database
            self.update_embedding_refs(all_ids)
            
            # Step 7: Validate
            if self.validate_index():
                duration = time.time() - start_time
                print(f"\n🎉 SUCCESS! Main index rebuilt in {duration/60:.1f} minutes")
                print(f"   📊 Total vectors: {index.ntotal:,}")
                print(f"   🎯 Now test: curl -X POST http://127.0.0.1:8088/search -H 'X-API-Key: sidecar-AfreWVOEVoCtXzMT0jejgTqsng4J-kwlICBQonyMbas' -H 'Content-Type: application/json' -d '{{\"query\": \"search\", \"k\": 3}}'")
                return True
            else:
                print("\n❌ Validation failed")
                return False
                
        except Exception as e:
            print(f"\n❌ Error during rebuild: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    rebuilder = MainIndexRebuilder()
    success = rebuilder.run()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()