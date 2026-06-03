"""Paper-faithful E2LSH hash family for the LSH-DHT baseline.

Implements the Datar, Indyk, Immorlica, Mirrokni (2004) ``p``-stable LSH scheme
for ``L_2`` distance:

    h_{a,b}(v) = floor((a . v + b) / w)

with ``a`` drawn from a standard Gaussian (the 2-stable distribution for L_2)
and ``b`` uniform in ``[0, w)``. The full ANN scheme is then assembled with
two amplification steps:

* AND amplification: ``g_j(v) = (h_{j,1}(v), ..., h_{j,k}(v))`` is a tuple of
  ``k`` independent base hashes; two vectors collide in table ``j`` iff they
  agree on all ``k`` base hashes.
* OR amplification: ``L`` independent tables ``g_1, ..., g_L``; the candidate
  set for a query ``q`` is the union of buckets ``T_j[g_j(q)]`` across the
  ``L`` tables.

Mirrors the parameter conventions of the reference implementation
``CharlesLiu7/p-stable-lsh-python``: the bucket width is named ``w`` here
(matching the paper text) but ``r`` is accepted as an alias.

This module is intentionally self-contained. Distributed bucket placement,
publication, query routing and metric accounting all live in
:mod:`src.lsh_network`; this file only exposes deterministic numpy math.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


BucketId = tuple[int, ...]


@dataclass(slots=True)
class E2LSHFamily:
    """Stateless E2LSH hash family with stacked projection storage.

    All ``L * k`` base hashes are kept in a single ``(L*k, d)`` projection
    matrix ``A`` and a ``(L*k,)`` offset vector ``B`` so that hashing a
    ``(d,)`` vector is one matrix multiplication.
    """

    d: int
    k: int
    L: int
    w: float
    seed: int
    A: np.ndarray
    B: np.ndarray

    @classmethod
    def build(
        cls,
        *,
        d: int,
        k: int,
        L: int,
        w: float | None = None,
        seed: int = 0,
        r: float | None = None,
    ) -> "E2LSHFamily":
        """Construct an ``E2LSHFamily`` with deterministic seeded draws.

        ``w`` is the paper symbol for the bucket width; ``r`` is accepted as
        an alias for compatibility with ``CharlesLiu7/p-stable-lsh-python``.
        """

        bucket_width = _resolve_bucket_width(w=w, r=r)
        if d <= 0:
            raise ValueError(f"d must be positive, got {d}")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if L <= 0:
            raise ValueError(f"L must be positive, got {L}")

        rng = np.random.default_rng(seed)
        # Gaussian for L_2 (2-stable); standard normal entries.
        projection = rng.standard_normal(size=(L * k, d)).astype(np.float32)
        offsets = rng.uniform(low=0.0, high=bucket_width, size=(L * k,)).astype(np.float32)
        return cls(d=d, k=k, L=L, w=float(bucket_width), seed=int(seed), A=projection, B=offsets)

    def hash_tables(self, vec: np.ndarray) -> tuple[BucketId, ...]:
        """Return the ``L`` bucket ids for ``vec``.

        Each bucket id is a length-``k`` tuple of Python ``int``\\ s suitable
        for use as a dictionary key.
        """

        if vec.shape != (self.d,):
            raise ValueError(
                f"Expected vector of shape ({self.d},), got {vec.shape}"
            )
        # Single matmul produces all L*k base hashes at once.
        projection = self.A @ vec.astype(np.float32, copy=False) + self.B
        bucket_ints = np.floor(projection / self.w).astype(np.int64)
        rows = bucket_ints.reshape(self.L, self.k)
        return tuple(tuple(int(value) for value in row) for row in rows)

    def hash_tables_batch(self, vectors: np.ndarray) -> np.ndarray:
        """Vectorized variant returning a ``(N, L, k)`` int64 array.

        Useful for the calibration sweep; not used on the simulator hot path.
        """

        if vectors.ndim != 2 or vectors.shape[1] != self.d:
            raise ValueError(
                f"Expected vectors of shape (N, {self.d}), got {vectors.shape}"
            )
        projection = vectors.astype(np.float32, copy=False) @ self.A.T + self.B
        bucket_ints = np.floor(projection / self.w).astype(np.int64)
        return bucket_ints.reshape(vectors.shape[0], self.L, self.k)

    def base_hash_count(self) -> int:
        return self.L * self.k


def _resolve_bucket_width(*, w: float | None, r: float | None) -> float:
    """Accept either ``w`` (paper) or ``r`` (reference impl) but not both."""

    if w is not None and r is not None and float(w) != float(r):
        raise ValueError(f"Conflicting bucket-width arguments: w={w}, r={r}")
    chosen = w if w is not None else r
    if chosen is None:
        raise ValueError("Bucket width must be provided as 'w' (or alias 'r')")
    chosen_value = float(chosen)
    if not chosen_value > 0:
        raise ValueError(f"Bucket width must be > 0, got {chosen_value}")
    return chosen_value


def analytic_collision_probability(*, c: float, w: float | None = None, r: float | None = None) -> float:
    """Theoretical per-base-hash collision probability for L_2 distance ``c``.

    Implements the integral from Datar et al. (2004), Section 3.2:

        p(c) = integral_{0}^{w} (1 / c) * f_N(t / c) * (1 - t / w) dt

    where ``f_N`` is the standard Gaussian density. Mirrors
    ``pstable.p`` in the reference ``CharlesLiu7/p-stable-lsh-python``.
    Used by :mod:`scripts.calibrate_lsh_params` to compare empirical and
    theoretical collision rates.
    """

    bucket_width = _resolve_bucket_width(w=w, r=r)
    if c <= 0:
        return 1.0

    try:
        from scipy import integrate
    except ImportError:  # pragma: no cover - scipy ships with sentence-transformers
        return _trapezoidal_collision_probability(c=c, w=bucket_width)

    def integrand(t: float) -> float:
        return _gaussian_pdf(t / c) * (1.0 - t / bucket_width) / c

    value, _err = integrate.quad(integrand, 0.0, bucket_width)
    return float(value)


def _gaussian_pdf(x: float) -> float:
    return float(np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi))


def _trapezoidal_collision_probability(*, c: float, w: float, steps: int = 4096) -> float:
    """Fallback numeric integration used only if scipy is unavailable."""

    grid = np.linspace(0.0, w, num=steps + 1)
    values = (1.0 / c) * np.exp(-0.5 * (grid / c) ** 2) / np.sqrt(2.0 * np.pi) * (1.0 - grid / w)
    # NumPy 2 renamed np.trapz to np.trapezoid; support both for portability.
    trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(trapezoid(values, grid))


def collision_probabilities(distances: Iterable[float], *, w: float) -> list[float]:
    """Convenience helper for calibration plots."""

    return [analytic_collision_probability(c=float(c), w=w) for c in distances]
