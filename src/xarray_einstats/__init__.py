"""Stats, linear algebra and einops for xarray."""

from __future__ import annotations

from .linalg import einsum, raw_einsum, einsum_path, matmul

__all__ = ["einsum", "raw_einsum", "einsum_path", "matmul"]

__version__ = "0.4.0.dev0"
