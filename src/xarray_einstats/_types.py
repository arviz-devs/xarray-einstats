"""Shared type aliases for xarray-einstats."""

from collections.abc import Hashable, Sequence

type Dims = Sequence[Hashable] | None
type DimOrDims = Hashable | Sequence[Hashable] | None
