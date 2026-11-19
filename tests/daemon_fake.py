# Author: Vito Foderà
# Created: 2026-10-29

"""Fictitious orchestrators started in the background by the tests of the daemon.

Each pass runs in a process of its own, as the daemon runs them, and writes
``passes`` (its number) and appends its pid to ``pids`` in the directory of
the run; at the end of the last pass, stopped by a signal or not, the
background process writes ``saved``, as the real orchestrator saves its
state. Nothing is submitted. ``build`` writes the arguments it receives after
the directory in ``factory_args.json``; ``broken`` fails while building the
orchestrator, ``slow`` builds it after the seconds written in ``delay``. A
pass whose number is listed in ``kill`` stops its own process with SIGKILL,
as the limit of CPU time of a login node does.
"""

import json
import os
import signal
import time
from pathlib import Path


class FakeOrchestrator:
    """Passes every 0.05 s until ``stop`` is true, or ``passes`` passes if written in ``limit``."""

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)

    def logged_pass(self, n):
        kill = self.run_dir / 'kill'
        if kill.exists() and n in json.loads(kill.read_text()):
            os.kill(os.getpid(), signal.SIGKILL)
        (self.run_dir / 'passes').write_text(str(n))
        with open(self.run_dir / 'pids', 'a') as pids:
            pids.write(f'{os.getpid()}\n')
        limit = self.run_dir / 'limit'
        return None, limit.exists() and n >= int(limit.read_text())

    def run(self, sleep, stop, one_pass):
        passes = 0
        while True:
            passes += 1
            if one_pass(passes) or stop():
                break
            sleep(0.05)
            if stop():
                break
        (self.run_dir / 'saved').write_text(str(passes))
        return passes


def build(run_dir, *args):
    (Path(run_dir) / 'factory_args.json').write_text(json.dumps(list(args)))
    return FakeOrchestrator(run_dir)


def broken(run_dir):
    raise RuntimeError('the factory failed')


def slow(run_dir):
    time.sleep(float((Path(run_dir) / 'delay').read_text()))
    return FakeOrchestrator(run_dir)
