"""
Leakage guard for NeuroTraction training.

The historical bug: StandardScaler was fit on the FULL dataset before the
train/test split, leaking test statistics into training and inflating the
reported R2. This test fails if the training scripts reintroduce that pattern,
and sanity-checks that a scaler fit on train-only data differs from one fit on
the full data (which is exactly why the leak mattered).
"""
import re
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

LEARNING = Path(__file__).resolve().parents[1] / "learning"


def _src(name):
    return (LEARNING / name).read_text()


def test_no_fit_transform_on_full_X():
    """Neither train script may fit the scaler on the whole X before splitting."""
    for script in ("train_traction_ai.py", "train_traction_ai_v2.py"):
        src = _src(script)
        # The leak signature: `X = scaler.fit_transform(X)` before a split.
        assert not re.search(r"^\s*X\s*=\s*scaler\.fit_transform\(\s*X\s*\)", src, re.M), (
            f"{script} fits the scaler on the full dataset -> leakage"
        )
        # Must fit on a train partition and only transform the test partition.
        assert "scaler.fit_transform(X_train_raw)" in src, f"{script} must fit on train only"
        assert "scaler.transform(X_test_raw)" in src, f"{script} must transform test with train stats"


def test_no_hardcoded_home_path():
    """No source file may contain a hardcoded /home/<user> path."""
    import glob
    root = LEARNING.parent
    py_files = glob.glob(str(root / "learning" / "*.py")) + glob.glob(str(root / "scripts" / "*.py"))
    for path in py_files:
        assert "/home/rhutvik" not in Path(path).read_text(), f"{path} has a hardcoded absolute path"


def test_leak_actually_changes_scaling():
    """Demonstrate why the fix matters: train-only stats != full-data stats."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    split = 160
    full = StandardScaler().fit(X)
    train_only = StandardScaler().fit(X[:split])
    # Means/scales should differ -> fitting on full data leaks test info.
    assert not np.allclose(full.mean_, train_only.mean_, atol=1e-6)
