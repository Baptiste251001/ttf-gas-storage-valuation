"""Interface shared by the price models used in the valuation engine."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike


class PriceModel(Protocol):
    """A spot model fitted to the forward curve, with a simulated (possibly vector) state."""

    def variance(self, t: ArrayLike, T: ArrayLike | None = None) -> np.ndarray: ...

    def simulate_state(
        self, times: ArrayLike, n_paths: int, rng: np.random.Generator
    ) -> np.ndarray: ...

    def spot(self, state: np.ndarray, times: ArrayLike, fwd: ArrayLike) -> np.ndarray: ...

    def forward_curve(
        self, state_t: np.ndarray, t: float, T: np.ndarray, fwd0: np.ndarray
    ) -> np.ndarray: ...

    def basis(self, state_k: np.ndarray, t: float, name: str, degree: int) -> np.ndarray: ...
