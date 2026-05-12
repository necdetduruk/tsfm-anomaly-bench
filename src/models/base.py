"""Common interface for anomaly scorers.

All baselines and foundation models implement this so they can share
the same experiment runner and evaluation pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class AnomalyScorer(ABC):
    """Per-timestep anomaly scorer.

    Higher score = more anomalous. Models that natively produce per-window
    scores should broadcast back to per-timestep before returning.
    """

    @abstractmethod
    def fit(self, train_data: np.ndarray) -> None:
        """Fit on standardized train data of shape (T, F)."""

    @abstractmethod
    def score(self, test_data: np.ndarray) -> np.ndarray:
        """Return per-timestep anomaly scores of shape (T,)."""

    @property
    def name(self) -> str:
        return self.__class__.__name__
