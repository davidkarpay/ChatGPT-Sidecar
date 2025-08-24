# tests/test_mmr.py
import numpy as np
from app.mmr import mmr

def test_empty():
    q = np.zeros((384,), dtype=float)
    C = np.zeros((0, 384), dtype=float)
    assert mmr(q, C, lamb=0.5, k=8) == []

def test_basic_selection():
    q = np.array([1.0, 0.0], dtype=float)
    C = np.array([[1,0], [0.9,0.1], [0,1]], dtype=float)
    idxs = mmr(q, C, lamb=0.5, k=2)
    assert len(idxs) == 2
    assert idxs[0] in (0,1)

def test_lambda_bounds():
    import pytest
    q = np.zeros((2,), dtype=float)
    C = np.zeros((1,2), dtype=float)
    with pytest.raises(ValueError):
        mmr(q, C, lamb=1.1, k=1)