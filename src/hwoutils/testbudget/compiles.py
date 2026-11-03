"""Process-wide JAX compile counter built on ``jax.monitoring``."""

import jax.monitoring

_BACKEND_COMPILE = "/jax/core/compile/backend_compile_duration"
_CACHE_HIT = "/jax/compilation_cache/cache_hits"

_count = 0
_installed = False


def _on_duration(event: str, duration: float, **kwargs) -> None:
    global _count
    if event == _BACKEND_COMPILE:
        _count += 1


def _on_event(event: str, **kwargs) -> None:
    global _count
    if event == _CACHE_HIT:
        _count += 1


def install() -> None:
    """Register the listeners once per process."""
    global _installed
    if _installed:
        return
    jax.monitoring.register_event_duration_secs_listener(_on_duration)
    jax.monitoring.register_event_listener(_on_event)
    _installed = True


def compile_count() -> int:
    """Return the number of distinct XLA programs compiled or loaded from the cache."""
    return _count
