from __future__ import annotations

import numpy as np

from jgrec.rankers.hybrid.fusion_lgbm import fit_fusion_lgbm

features = np.random.default_rng(0).normal(size=(20, 100, 3)).astype(np.float32)
result = fit_fusion_lgbm(
    features[:10],
    features[10:],
    selection_metric="mrr",
    verbose=False,
)
print(f"mrr={result.best_val_mrr:.8f} model_chars={len(result.model_text)}")
