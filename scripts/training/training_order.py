"""Deterministic epoch-wise ordering for training records."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import TypeVar


T = TypeVar("T")


def epoch_training_rows(
    rows: Sequence[T], *, seed: int, epoch: int, shuffle: bool = True
) -> list[T]:
    """Return a fresh, reproducibly shuffled row list for one epoch.

    Validation callers should not use this helper: their ordering stays fixed.
    A separate RNG derived from ``seed + epoch`` avoids dependence on any model
    or sampling RNG state while ensuring that a resumed run recreates the same
    epoch order.
    """
    ordered = list(rows)
    if shuffle:
        random.Random(int(seed) + int(epoch)).shuffle(ordered)
    return ordered
