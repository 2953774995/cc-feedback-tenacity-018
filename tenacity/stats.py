# Copyright 2026 Julien Danjou
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Call statistics for retried functions.

The :func:`tracked` decorator records how often a function is called, how many
attempts those calls consumed and which exceptions each attempt raised.  It is
meant to be stacked *on top of* the retry decorator, so that each retried call
is seen once and each attempt it makes is counted::

    from tenacity import retry, stop_after_attempt
    from tenacity.stats import tracked

    @tracked
    @retry(stop=stop_after_attempt(3))
    def fetch():
        ...

    fetch()
    fetch.calls      # 1
    fetch.attempts   # number of attempts performed so far (1 if it succeeded)
    fetch.failures   # [Failure(attempt=1, type="ValueError", ...)]

The counters accumulate across calls; they are never reset by the decorator.
"""

from __future__ import annotations

import contextvars
import dataclasses
import functools
import threading
import typing as t

from tenacity import BaseRetrying, _has_tornado, _utils

__all__ = ["Failure", "tracked"]

# 1-based index of the attempt currently running within a call.  Retried
# functions get it from the retrying object (see `_attempt_hook`), which is
# authoritative and works for tornado too; plain functions simply increment it.
# Being a context variable, it also keeps concurrent calls in threads and
# asyncio tasks apart.
_attempt_number: contextvars.ContextVar[int] = contextvars.ContextVar(
    "tenacity_stats_attempt_number", default=0
)


@dataclasses.dataclass(frozen=True, slots=True)
class Failure:
    """A single failed attempt.

    :ivar attempt: 1-based index of the attempt within its call.
    :ivar type: name of the exception class raised by the attempt.
    :ivar exception: the exception instance that was raised.
    """

    attempt: int
    type: str
    exception: BaseException


class _CallStats:
    __slots__ = ("_lock", "_wrappers", "attempts", "calls", "failures")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.attempts = 0
        self.calls = 0
        self.failures: list[Failure] = []
        # Every wrapper sharing this state (the original and any built by
        # ``retry_with``) is refreshed, so they all observe the same totals.
        self._wrappers: list[t.Any] = []

    def add_wrapper(self, wrapper: t.Any) -> None:
        with self._lock:
            self._wrappers.append(wrapper)
        self._touch()

    def begin_call(self) -> None:
        """Count one call."""
        with self._lock:
            self.calls += 1
        self._touch()

    def record_attempt(self) -> None:
        """Count one attempt."""
        with self._lock:
            self.attempts += 1
        self._touch()

    def record_failure(self, attempt: int, exception: BaseException) -> None:
        """Attach the exception raised by attempt number ``attempt``."""
        with self._lock:
            self.failures.append(Failure(attempt, type(exception).__name__, exception))
        self._touch()

    def _touch(self) -> None:
        with self._lock:
            wrappers = list(self._wrappers)
            calls, attempts = self.calls, self.attempts
            failures = self.failures
        for wrapper in wrappers:
            wrapper.calls = calls
            wrapper.attempts = attempts
            wrapper.failures = failures


def _record_attempt(fn: t.Callable[..., t.Any], stats: _CallStats) -> t.Any:
    """Wrap ``fn`` so each invocation is counted as one retry attempt."""
    if _utils.is_coroutine_callable(fn):

        @functools.wraps(fn)
        async def attempt(*args: t.Any, **kwargs: t.Any) -> t.Any:
            index = _enter_attempt(stats)
            try:
                return await fn(*args, **kwargs)
            except BaseException as exc:
                stats.record_failure(index, exc)
                raise

    elif _is_tornado_coroutine(fn):
        import tornado.gen

        @tornado.gen.coroutine
        @functools.wraps(fn)
        def attempt(*args: t.Any, **kwargs: t.Any) -> t.Any:
            index = _enter_attempt(stats)
            try:
                result = yield fn(*args, **kwargs)
            except BaseException as exc:
                stats.record_failure(index, exc)
                raise
            raise tornado.gen.Return(result)

    else:

        @functools.wraps(fn)
        def attempt(*args: t.Any, **kwargs: t.Any) -> t.Any:
            index = _enter_attempt(stats)
            try:
                return fn(*args, **kwargs)
            except BaseException as exc:
                stats.record_failure(index, exc)
                raise

    return attempt


def _enter_attempt(stats: _CallStats) -> int:
    index = _attempt_number.get() + 1
    _attempt_number.set(index)
    stats.record_attempt()
    return index


def _attempt_hook(
    original: t.Callable[..., t.Any] | None,
) -> t.Callable[..., t.Any]:
    """Build a ``before`` callback that publishes the current attempt number.

    Using the retrying object's own number (rather than counting inside the
    wrapped function) means the index is correct even for tornado, which does
    not carry context-variable writes across a coroutine's suspension points.
    """
    if original is not None and _utils.is_coroutine_callable(original):

        async def before(retry_state: t.Any) -> None:
            _attempt_number.set(retry_state.attempt_number - 1)
            await original(retry_state)

    elif original is not None:

        def before(retry_state: t.Any) -> None:
            _attempt_number.set(retry_state.attempt_number - 1)
            original(retry_state)

    else:

        def before(retry_state: t.Any) -> None:
            _attempt_number.set(retry_state.attempt_number - 1)

    return before


def _is_tornado_coroutine(fn: object) -> bool:
    if not _has_tornado():
        return False
    import tornado.gen

    return tornado.gen.is_coroutine_function(fn)


def _wrap(fn: t.Callable[..., t.Any], driver: t.Any, stats: _CallStats) -> t.Any:
    """Build the public wrapper around ``driver`` and publish the counters."""
    if _utils.is_coroutine_callable(driver):

        @functools.wraps(fn)
        async def tracked_fn(*args: t.Any, **kwargs: t.Any) -> t.Any:
            stats.begin_call()
            token = _attempt_number.set(0)
            try:
                return await driver(*args, **kwargs)
            finally:
                _attempt_number.reset(token)

    elif _is_tornado_coroutine(driver):
        import tornado.gen

        # Yield the driver rather than returning its Future, so the counters
        # stay attached to the call until the retry loop has finished.
        @tornado.gen.coroutine
        @functools.wraps(fn)
        def tracked_fn(*args: t.Any, **kwargs: t.Any) -> t.Any:
            stats.begin_call()
            token = _attempt_number.set(0)
            try:
                result = yield driver(*args, **kwargs)
            finally:
                _attempt_number.reset(token)
            raise tornado.gen.Return(result)

    else:

        @functools.wraps(fn)
        def tracked_fn(*args: t.Any, **kwargs: t.Any) -> t.Any:
            stats.begin_call()
            token = _attempt_number.set(0)
            try:
                return driver(*args, **kwargs)
            finally:
                _attempt_number.reset(token)

    # functools.wraps copied the attributes of the *original* retry wrapper.
    # Point them at the retrying object this wrapper actually drives, so
    # `statistics` and `retry` describe the instrumented calls.
    if hasattr(driver, "retry"):
        tracked_fn.retry = driver.retry
        tracked_fn.statistics = driver.statistics
    stats.add_wrapper(tracked_fn)
    return tracked_fn


def tracked(fn: t.Callable[..., t.Any]) -> t.Any:
    """Track calls, attempts and failures of ``fn``.

    ``fn`` may be a plain function or a function already wrapped with
    ``@retry``.  In the latter case the retry policy is reused unchanged: a
    copy of the retrying object is taken and the attempts it runs are
    recorded.
    """
    stats = _CallStats()
    controller = getattr(fn, "retry", None)

    if isinstance(controller, BaseRetrying) and hasattr(fn, "__wrapped__"):
        inner = fn.__wrapped__

        def build(ctrl: BaseRetrying) -> t.Any:
            # Chain onto whatever ``before`` the caller asked for, so the
            # attempt number is recorded without dropping their callback.
            hook = _attempt_hook(ctrl.before)
            configured = ctrl.copy(before=hook)
            return configured.wraps(_record_attempt(inner, stats))

        wrapper = _wrap(fn, build(controller), stats)

        def retry_with(*args: t.Any, **kwargs: t.Any) -> t.Any:
            return _wrap(fn, build(controller.copy(*args, **kwargs)), stats)

    else:
        wrapper = _wrap(fn, _record_attempt(fn, stats), stats)

        def retry_with(*args: t.Any, **kwargs: t.Any) -> t.Any:
            raise TypeError(f"{fn!r} is not decorated with @retry")

    if isinstance(controller, BaseRetrying):
        wrapper.retry_with = retry_with
    return wrapper
