
from typing import List, Dict

def chunk_text(s: str, size: int = 1200, overlap: int = 120) -> list:
    out = []
    start = 0
    n = len(s)
    if size <= 0:
        size = 1200
    if overlap < 0:
        overlap = 0
    while start < n:
        end = min(start + size, n)
        out.append({
            'text': s[start:end],
            'start_char': start,
            'end_char': end,
            'token_estimate': (end - start) // 4
        })
        if end == n:
            break
        start = max(0, end - overlap)
    return out
