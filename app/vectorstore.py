
import numpy as np
from pathlib import Path
import pickle
import faiss
from sentence_transformers import SentenceTransformer

class FaissStore:
    def __init__(self, index_path: Path, ids_path: Path, model_name: str):
        self.index_path = Path(index_path)
        self.ids_path = Path(ids_path)
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.ids = []  # list of (embedding_ref_id, chunk_id)

    def load(self):
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with open(self.ids_path, 'rb') as f:
                self.ids = pickle.load(f)
        else:
            self.index = None
            self.ids = []

    def save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.model.get_sentence_embedding_dimension())
        faiss.write_index(self.index, str(self.index_path))
        with open(self.ids_path, 'wb') as f:
            pickle.dump(self.ids, f)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalized float32 embeddings for texts."""
        embs = self.model.encode(
            texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True
        )
        return np.asarray(embs, dtype="float32")

    def _encode(self, texts):
        return self.encode(texts)

    def build(self, rows):
        vecs = self._encode([r['text'] for r in rows])
        dim = vecs.shape[1]
        index = faiss.IndexFlatIP(dim)  # uncompressed, cosine via normalized dot
        index.add(vecs)
        self.index = index
        self.ids = [(r['embedding_ref_id'], r['chunk_id']) for r in rows]
        self.save()

    def add(self, rows):
        vecs = self._encode([r['text'] for r in rows])
        if self.index is None:
            self.build(rows)
            return list(range(len(rows)))
        self.index.add(vecs)
        start = len(self.ids)
        self.ids.extend([(r['embedding_ref_id'], r['chunk_id']) for r in rows])
        self.save()
        return list(range(start, start + len(rows)))

    def search(self, query: str, k: int = 8):
        if self.index is None:
            return []
        qv = self._encode([query])
        scores, idxs = self.index.search(qv, k)
        return list(zip(idxs[0].tolist(), scores[0].tolist()))
