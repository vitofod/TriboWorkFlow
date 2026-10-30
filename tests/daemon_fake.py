# Author: Vito Foderà
# Created: 2026-10-29

"""Fictitious orchestrators started in the background by the tests of the daemon.

Each pass writes ``passes`` in the directory of the run; at the end of the
last pass, stopped by a signal or not, it writes ``saved``, as the real
orchestrator saves its state. Nothing is submitted. ``broken`` fails while
building the orchestrator, ``slow`` builds it after the seconds written in
``delay``.
"""

import time
from pathlib import Path


class FakeOrchestrator:
    """Passes every 0.05 s until ``stop`` is true, or ``passes`` passes if written in ``limit``."""

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)

    def run(self, sleep, stop):
        limit = self.run_dir / 'limit'
        passes = 0
        while True:
            passes += 1
            (self.run_dir / 'passes').write_text(str(passes))
            if stop() or (limit.exists() and passes >= int(limit.read_text())):
                break
            sleep(0.05)
            if stop():
                break
        (self.run_dir / 'saved').write_text(str(passes))
        return passes


def build(run_dir):
    return FakeOrchestrator(run_dir)


def broken(run_dir):
    raise RuntimeError('the factory failed')


def slow(run_dir):
    time.sleep(float((Path(run_dir) / 'delay').read_text()))
    return FakeOrchestrator(run_dir)
