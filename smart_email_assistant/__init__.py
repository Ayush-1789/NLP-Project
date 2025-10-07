"""Top-level package for Smart Email Assistant with re-exported API."""
from __future__ import annotations

from importlib import import_module
from typing import Any

_internal = import_module(".smart_email_assistant", __name__)

__all__ = list(getattr(_internal, "__all__", []))

if hasattr(_internal, "__path__"):
    __path__ = _internal.__path__

for _name in __all__:
    globals()[_name] = getattr(_internal, _name)


def __getattr__(name: str) -> Any:
    return getattr(_internal, name)
