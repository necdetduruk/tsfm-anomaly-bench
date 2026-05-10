"""SMD (Server Machine Dataset) loader.

Per-machine multivariate time series for anomaly detection.
28 machines x 38 metrics each. Anomaly labels exist only on the test split.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

SMD_ROOT = Path("data/smd")
N_FEATURES = 38


def list_machines(root: Path = SMD_ROOT) -> list[str]:
    """Return sorted list of machine IDs available, e.g. ['machine-1-1', ...]."""
    train_dir = root / "train"
    if not train_dir.exists():
        raise FileNotFoundError(
            f"SMD train dir not found: {train_dir}. Run scripts/download_smd.sh."
        )
    return sorted(p.stem for p in train_dir.glob("machine-*.txt"))


def _load_series(path: Path) -> np.ndarray:
    """Load an SMD .txt as float32 array (T, F)."""
    return np.loadtxt(path, delimiter=",", dtype=np.float32)


def _load_labels(path: Path) -> np.ndarray:
    """Load test_label .txt as binary array (T,)."""
    return np.loadtxt(path, delimiter=",", dtype=np.int64)


def compute_scaler(train_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean/std from train. Constant-feature std clamped to 1.0."""
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


class SMDDataset(Dataset):
    """Sliding-window dataset for one SMD machine.

    Args:
        machine: machine ID, e.g. 'machine-1-1'.
        split: 'train', 'val', or 'test'. 'val' = last 20% of train.
        window: window length in timesteps. Default 100.
        stride: window stride. Default 1 (full overlap).
        root: SMD data root.
        scaler_stats: optional (mean, std) precomputed on train. If None,
            computed from this machine's train split. For non-train splits
            you SHOULD pass the train-fitted scaler to avoid leakage.

    Returns from __getitem__:
        (window, label) where window is (window, F) float32 tensor and
        label is a (window,) int64 tensor (zeros for train/val).
    """

    def __init__(
        self,
        machine: str,
        split: str = "train",
        window: int = 100,
        stride: int = 1,
        root: Path = SMD_ROOT,
        scaler_stats: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train/val/test, got {split!r}")

        self.machine = machine
        self.split = split
        self.window = window
        self.stride = stride

        train_full = _load_series(root / "train" / f"{machine}.txt")
        test = _load_series(root / "test" / f"{machine}.txt")
        test_labels = _load_labels(root / "test_label" / f"{machine}.txt")

        # Train/val split: last 20% of train is val (no shuffle, time order)
        n_val = max(1, int(len(train_full) * 0.2))
        train, val = train_full[:-n_val], train_full[-n_val:]

        # Fit scaler on TRAIN ONLY (or use supplied stats)
        if scaler_stats is None:
            self.scaler_stats = compute_scaler(train)
        else:
            self.scaler_stats = scaler_stats

        mean, std = self.scaler_stats

        if split == "train":
            data = (train - mean) / std
            labels = np.zeros(len(data), dtype=np.int64)
        elif split == "val":
            data = (val - mean) / std
            labels = np.zeros(len(data), dtype=np.int64)
        else:  # test
            data = (test - mean) / std
            labels = test_labels

        self.data = torch.from_numpy(data.astype(np.float32))
        self.labels = torch.from_numpy(labels)

        n = len(self.data)
        if n < window:
            raise ValueError(f"{machine}/{split} has {n} steps < window {window}")
        self._starts = np.arange(0, n - window + 1, stride)

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = int(self._starts[idx])
        e = s + self.window
        return self.data[s:e], self.labels[s:e]

    @property
    def n_features(self) -> int:
        return self.data.shape[1]
