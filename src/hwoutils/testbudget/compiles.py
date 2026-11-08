"""Process-wide JAX compile counter and timer built on ``jax.monitoring``.

Events used (names read from jax/_src/dispatch.py and jax/_src/compiler.py):

* ``/jax/core/compile/backend_compile_duration`` is a duration event wrapped
  around ``compile_or_get_cached``, so it fires for a real XLA compile and
  also for a persistent-cache load.
* ``/jax/compilation_cache/cache_hits`` is a plain event recorded inside that
  wrapped block, on the same thread and before the duration event, only when
  the program was loaded from the persistent cache.

JAX records no event that only a real backend compile (or only a cache load)
emits: with the persistent cache off nothing is recorded inside the compile
path, and ``/jax/compilation_cache/cache_misses`` exists only when the cache is
on (compilation_cache.py, ``put_executable_and_time``). So the two cases are
told apart by order: each cache-hit event adds one to a per-thread pending
counter (compiler.py, ``compile_or_get_cached``, after ``_cache_read``
succeeds) and the duration event that closes the same block consumes one. A
duration event with no pending hit is a real compile and adds one to the count
and its duration to the seconds.

``reset_pending`` clears only the calling thread's counter, at each test start
on the pytest thread, where no compile can be in flight. Other threads are left
alone because a background cache load may span a test boundary and must still
pair with its duration.

One assumption remains: other monitoring listeners neither raise nor compile
inside a compile. JAX calls listeners synchronously, so a raising listener
makes the compiling call itself raise and the test errors visibly; it can also
strand a pending hit on a thread other than the pytest thread, which is not
cleaned up.
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


def _pending() -> int:
    return getattr(_state, "pending_hits", 0)


def _on_duration(event: str, duration: float, **kwargs) -> None:
    global _count, _seconds
    if event != _BACKEND_COMPILE:
        return
    global _cache_hits
    pending = _pending()
    if pending > 0:
        _state.pending_hits = pending - 1
        _cache_hits += 1
        return
    _count += 1
    _seconds += float(duration)


def _on_event(event: str, **kwargs) -> None:
    if event == _CACHE_HIT:
        _state.pending_hits = _pending() + 1


def reset_pending() -> None:
    """Forget unpaired cache-hit events on the calling thread."""
    _state.pending_hits = 0


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
    """Return the number of programs loaded from the persistent cache.

    A load is counted when its duration event closes it, the same moment a real
    compile is counted, so both fall in the same test and module.
    """
    return _cache_hits
