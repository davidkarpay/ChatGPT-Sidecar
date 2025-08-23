# app/mmr.py
import numpy as np

def mmr(
    query_vec: np.ndarray,
    cand_vecs: np.ndarray,
    lamb: float = 0.5,
    k: int = 8
) -> list[int]:
    """
    Maximal Marginal Relevance selection.
    - query_vec: shape (D,)
    - cand_vecs: shape (N, D)
    Returns indices of selected candidates (into cand_vecs).
    """
    if not (0.0 <= lamb <= 1.0):
        raise ValueError("lambda must be in [0,1]")
    if cand_vecs.size == 0 or k <= 0:
        return []

    def _norm(x: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(x, axis=-1, keepdims=True)
        n[n == 0] = 1.0
        return x / n

    q = _norm(query_vec.reshape(1, -1))[0]
    C = _norm(cand_vecs)

    sims = C @ q  # similarity to query
    selected: list[int] = []
    remaining = list(range(C.shape[0]))

    while remaining and len(selected) < k:
        if not selected:
            j_rel = int(np.argmax(sims[remaining]))
            selected.append(remaining.pop(j_rel))
            continue

        # diversity: max similarity to any already selected
        sel_mat = C[selected]  # (|S|, D)
        div = np.max(C[remaining] @ sel_mat.T, axis=1)  # (len(remaining),)
        score = lamb * sims[remaining] - (1 - lamb) * div
        j = int(np.argmax(score))
        selected.append(remaining.pop(j))

    return selected