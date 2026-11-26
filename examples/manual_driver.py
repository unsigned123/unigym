"""Thin wrapper around the packaged ``unigym.examples.manual_driver`` quickstart.

The canonical example now ships inside the distribution.  This file only
delegates so ``PYTHONPATH=. python examples/manual_driver.py`` keeps working.
"""

from __future__ import annotations

from unigym.examples.manual_driver import main

if __name__ == "__main__":
    main()
