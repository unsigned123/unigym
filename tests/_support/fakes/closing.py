"""Close-boundary doubles for backend lifecycle fault injection."""

from __future__ import annotations


class RecordingClose:
    """A closeable boundary object that records successful close calls."""

    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        """Record one successful close."""
        self.close_calls += 1


class FailingClose:
    """A closeable boundary object whose first close fails."""

    def __init__(self) -> None:
        self.close_attempts = 0

    def close(self) -> None:
        """Fail the first teardown attempt."""
        self.close_attempts += 1
        raise RuntimeError("prop backend teardown failed")
