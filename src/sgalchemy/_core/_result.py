"""Expose projection rows and result collections."""

from collections.abc import Iterator, Mapping
from typing import Any


class Row(Mapping[str, Any]):
    """One projection row with mapping and attribute access."""

    def __init__(self, values: Mapping[str, Any]) -> None:
        """Copy projection values into deterministic row storage.

        Args:
            values: Projection values to copy into the row.
        """
        self._values = dict(values)

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 - Projection keys select heterogeneous values dynamically.
        """Return a projected value by rendered field-path key."""
        if name.startswith('_'):
            raise AttributeError(name)
        try:
            return self._values[name]
        except KeyError:
            raise AttributeError(name) from None

    def __getitem__(self, key: str) -> Any:  # noqa: ANN401 - Projection keys select heterogeneous values dynamically.
        """Return a projected value by key."""
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate projected keys in stored order."""
        return iter(self._values)

    def __len__(self) -> int:
        """Return the number of projected values."""
        return len(self._values)


class Result(list[Row]):
    """Concrete collection of projection rows."""
