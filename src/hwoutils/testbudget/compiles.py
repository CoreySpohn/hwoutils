"""Process-wide JAX compile counter and timer built on ``jax.monitoring``.

Events used (names read from jax/_src/dispatch.py and jax/_src/compiler.py):

* ``/jax/core/compile/backend_compile_duration`` is a duration event wrapped
  around ``compile_or_get_cached``, so it fires for a real XLA compile and
  also for a persistent-cache load.
* ``/jax/compilation_cache/cache_hits`` is a plain event recorded inside that
  wrapped block, on the same thread and before the duration event, only when
  the program was loaded from the persistent cache.

A duration event that follows a cache-hit event on the same thread is a cache
load and is tallied separately; every other duration event is a real compile
and adds one to the count and its duration to the seconds.
"""

import threading

import jax.monitoring

_BACKEND_COMPILE = "/jax/core/compile/backend_compile_duration"
_CACHE_HIT = "/jax/compilation_cache/cache_hits"

_count = 0
_seconds = 0.0
_cache_hits = 0
_installed = False
_state = threading.local()


def _on_duration(event: str, duration: float, **kwargs) -> None:
    global _count, _seconds
    if event != _BACKEND_COMPILE:
        return
    if getattr(_state, "pending_hit", False):
        _state.pending_hit = False
        return
    _count += 1
    _seconds += float(duration)


def _on_event(event: str, **kwargs) -> None:
    global _cache_hits
    if event == _CACHE_HIT:
        _cache_hits += 1
        _state.pending_hit = True


def install() -> None:
    """Register the listeners once per process."""
    global _installed
    if _installed:
        return
    jax.monitoring.register_event_duration_secs_listener(_on_duration)
    jax.monitoring.register_event_listener(_on_event)
    _installed = True


def compile_count() -> int:
    """Return the number of XLA programs compiled, excluding cache loads."""
    return _count


def compile_seconds() -> float:
    """Return the backend compile time in seconds, excluding cache loads."""
    return _seconds


def cache_hit_count() -> int:
    """Return the number of programs loaded from the persistent cache."""
    return _cache_hits
