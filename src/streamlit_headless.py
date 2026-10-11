"""
A do-nothing ``streamlit`` for running the app's functions without a server.

``14_app.py`` calls ``st.*`` at import time and inside ``analyze_resume``.
``verify_deployment.py`` needs to call that real code (the same
``analyze_resume`` the browser button calls) to prove the app and the CLI
agree, without launching Streamlit. ``install()`` puts this stand-in in
``sys.modules['streamlit']``; every widget call becomes a no-op and the
caching decorators return the function unchanged.

Only for verification and tests. ``streamlit run`` never imports it.
"""

from __future__ import annotations

import sys
import types
from typing import Any


class _Anything:
    """Absorbs any attribute access, call, ``with`` block or iteration."""

    def __getattr__(self, name: str) -> Any:

        if name.startswith("__"):
            raise AttributeError(name)

        return _Anything()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return _Anything()

    def __enter__(self) -> "_Anything":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def __iter__(self):
        return iter([_Anything(), _Anything(), _Anything(), _Anything()])

    def __bool__(self) -> bool:
        return False


def _cache(*args: Any, **kwargs: Any) -> Any:
    """``@st.cache_x`` and ``@st.cache_x(...)`` both return the function."""

    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]

    return lambda function: function


def build() -> types.ModuleType:

    module = types.ModuleType("streamlit")

    module.__dict__["__getattr__"] = lambda name: _Anything()

    module.cache_data = _cache
    module.cache_resource = _cache

    module.session_state = {}

    module.__headless__ = True

    return module


def install(force: bool = False) -> types.ModuleType:
    """Install the stand-in unless a real streamlit is already imported."""

    existing = sys.modules.get("streamlit")

    if existing is not None and not force and not getattr(
        existing, "__headless__", False
    ):
        return existing

    module = build()

    sys.modules["streamlit"] = module

    return module
