"""Isolation Forest baseline.

Treats each timestep as an iid feature vector (no temporal context).
This is the simple, well-understood baseline; foundation models will
add the temporal modeling and we'll measure the lift.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

from .base import AnomalyScorer


class IsolationForestScorer(AnomalyScorer):
    def __init__(
        self,
        n_estimators: int = 100,
        contamination: str | float = "auto",
        max_samples: str | int = "auto",
        random_state: int = 0,
        n_jobs: int = -1,
    ) -> None:
        self.params = dict(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self._model: IsolationForest | None = None

    def fit(self, train_data: np.ndarray) -> None:
        self._model = IsolationForest(**self.params)
        self._model.fit(train_data)

    def score(self, test_data: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before score().")
        # score_samples: higher = more normal. Negate for anomaly score.
        return -self._model.score_samples(test_data)
