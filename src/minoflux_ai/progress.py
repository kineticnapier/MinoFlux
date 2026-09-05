from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class _NullProgress(Generic[T]):
    def __init__(self, iterable: Iterable[T] | None = None, *, total: int | None = None, **_: Any) -> None:
        self._iterable = iterable
        self.total = total
        self.n = 0

    def __iter__(self) -> Iterator[T]:
        if self._iterable is None:
            return iter(())
        for item in self._iterable:
            yield item
            self.n += 1

    def update(self, amount: int = 1) -> None:
        self.n += int(amount)

    def set_postfix(self, ordered_dict: dict[str, object] | None = None, **kwargs: object) -> None:
        return None

    def set_description(self, desc: str | None = None, refresh: bool = True) -> None:
        return None

    def close(self) -> None:
        return None



def progress_bar(
    iterable: Iterable[T] | None = None,
    *,
    total: int | None = None,
    desc: str | None = None,
    unit: str = "it",
    leave: bool = True,
    **kwargs: Any,
):
    """Return tqdm when available, otherwise a no-op compatible progress object."""
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return _NullProgress(iterable, total=total, desc=desc, unit=unit, leave=leave, **kwargs)
    return tqdm(iterable, total=total, desc=desc, unit=unit, leave=leave, dynamic_ncols=True, **kwargs)
