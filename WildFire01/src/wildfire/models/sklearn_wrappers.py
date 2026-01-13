from __future__ import annotations
import numpy as np

def sklearn_predict_proba(model, X: np.ndarray) -> np.ndarray:
    # returns prob of class 1
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        return p[:, 1].astype(np.float32, copy=False)
    # fallback: decision_function -> sigmoid
    if hasattr(model, "decision_function"):
        z = model.decision_function(X).astype(np.float32, copy=False)
        return 1.0 / (1.0 + np.exp(-z))
    raise TypeError("Model không có predict_proba/decision_function")

def prob_to_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p)).astype(np.float32, copy=False)
