# Author: Vito Foderà
# Created: 2026-10-27

"""Checks of the execution on an HPC machine, with a fake engine: no calculation runs.

Run by hand on the HPC machine, from the login node; see ``docs/execution.md``.
Commands:

``prepare RUN``
    Writes a run of fake calculations and their group of jobs, and submits
    the jobs with ``--submit``. Each calculation runs a small Python program
    in place of pw.x, started through the launcher of the machine (e.g.
    ``srun``); on rank 0 it prints a synthetic pw.x output and logs where and
    when it ran; every rank exits with the code given to the calculation, in
    turn 0, 2, 3 and 255 (the codes of pw.x 7.2 for a run that ended, an SCF
    not converged, a relaxation not converged, a stop at ``max_seconds``).
``report RUN``
    Once the jobs are over: whether each calculation ran exactly once, the
    exit codes seen by the worker against those of the fake engine, the
    claims, and how the calculations were spread over jobs and hosts.
``locks DIR``
    One contender of the lock check: tries to take every one of ``--names``
    locks with ``O_CREAT | O_EXCL`` or ``mkdir`` and writes those it took.
    Started many times at once, e.g. ``srun -N 2 -n 64 python hpc_check.py
    locks DIR --start-at ...``.
``locks-report DIR``
    Whether each lock was taken by exactly one contender.

The exit status of ``report`` and ``locks-report`` is 1 when a calculation
ran twice or a lock had more than one owner.
"""

import argparse
import json
import os
import random
import shlex
import socket
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from triboworkflow.core import store
from triboworkflow.execution import layout
from triboworkflow.execution.jobs import (MANIFEST_FILE, SUBMITTED_FILE, Manifest, read_manifest,
                                          submit_jobs, write_job)
from triboworkflow.execution.scheduler import JobScheduler

SYSTEM = 'check'
STEP = 'fake_scf'
EXIT_CODES = (0, 2, 3, 255)
FAKE_FILE = 'fake_pw.py'
SPEC_FILE = 'fake.json'
LOG_DIR = 'fake_log'

FAKE_ENGINE = '''\
# Fake pw.x of hpc_check.py: no calculation.
import json, os, socket, sys, time
args = sys.argv[1:]
run_input = args[args.index('-in') + 1]
with open('fake.json') as f:
    spec = json.load(f)
rank = next((os.environ[v] for v in ('SLURM_PROCID', 'PMI_RANK', 'OMPI_COMM_WORLD_RANK',
                                     'PMIX_RANK') if v in os.environ), '0')
started = time.time()
time.sleep(spec['sleep'])
if rank == '0':
    max_seconds = next((line.split('=')[1].strip() for line in open(run_input)
                        if line.strip().lower().startswith('max_seconds')), None)
    sys.stdout.write(spec['output'])
    sys.stdout.flush()
    entry = {'label': os.path.basename(os.getcwd()), 'host': socket.gethostname(),
             'job_id': os.environ.get('SLURM_JOB_ID') or os.environ.get('PBS_JOBID'),
             'started': started, 'ended': time.time(), 'exit': spec['exit'],
             'max_seconds': max_seconds}
    path = os.path.join(spec['log'], entry['label'] + '.' + str(os.getpid()) + '.json')
    with open(path, 'w') as f:
        json.dump(entry, f)
sys.exit(spec['exit'])
'''

# Complete scf output of pw.x 7.2, reduced to the lines the states depend on.
OUTPUT = '\n'.join([
    '', '     Program PWSCF v.7.2 starts on  1Jan2026 at 10: 0: 0 ', '',
    '     bravais-lattice index     =            0', '',
    '     crystal axes: (cart. coord. in units of alat)', '',
    '     Self-consistent Calculation', '',
    '     End of self-consistent calculation', '',
    '!    total energy              =     -31.10000000 Ry', '',
    '     convergence has been achieved in   7 iterations', '',
    '=' + '-' * 78 + '=', '   JOB DONE.', '=' + '-' * 78 + '=', ''])

INPUT = "&CONTROL\n  calculation = 'scf'\n  prefix = '{prefix}'\n  outdir = '{outdir}'\n/\n"


def prepare(args):
    run_dir = Path(args.run).resolve()
    if run_dir.exists():
        sys.exit(f'{run_dir} exists: give a new directory')
    log_dir = run_dir / LOG_DIR
    log_dir.mkdir(parents=True)
    fake = run_dir / FAKE_FILE
    fake.write_text(FAKE_ENGINE)
    sdir = layout.system_dir(run_dir, SYSTEM)
    calculations = []
    for i in range(args.calcs):
        calc_id = f'{STEP}/c{i:03d}'
        path = layout.calc_dir(run_dir, SYSTEM, calc_id)
        path.mkdir(parents=True)
        (path / layout.INPUT_FILE).write_text(
            INPUT.format(prefix=layout.calc_prefix(calc_id), outdir=layout.OUTDIR))
        (path / SPEC_FILE).write_text(json.dumps({
            'log': str(log_dir), 'sleep': args.sleep, 'output': OUTPUT,
            'exit': EXIT_CODES[i % len(EXIT_CODES)]}))
        store.write_record(sdir, store.Record(
            system=SYSTEM, step=STEP, label=f'c{i:03d}', calc_type='fake',
            engine=store.Engine('fake', FAKE_FILE), energy_model={'fake': True},
            fingerprint=store.fingerprint({'fake': i}),
            files={'input': layout.INPUT_FILE, 'output': layout.OUTPUT_FILE}))
        calculations.append((SYSTEM, calc_id))
    command = ' '.join(filter(None, [args.launcher, shlex.quote(sys.executable),
                                      shlex.quote(str(fake))]))
    manifest = Manifest(name='check', scheduler=args.scheduler, run_command=command,
                        walltime=args.walltime, max_seconds_margin=args.margin,
                        calculations=tuple(calculations))
    group = write_job(run_dir, manifest, args.jobscript)
    print(f'Run written in {run_dir}; group of jobs in {group}')
    print(f'Command of each calculation: {command} -in run.pwi >> output.pwo')
    scheduler = JobScheduler(args.scheduler, dry_run=not args.submit)
    job_ids = submit_jobs(group, scheduler, args.groups)
    if args.submit:
        print(f'Submitted {len(job_ids)} jobs: {", ".join(job_ids)}')
    else:
        print('Not submitted (no --submit). Commands:')
        for cmd, _, cwd in scheduler.commands:
            print(f'  cd {cwd} && {" ".join(cmd)}')


def report(args):
    run_dir = Path(args.run).resolve()
    manifest, _ = read_manifest(layout.job_dir(run_dir, 'check') / MANIFEST_FILE)
    submitted = layout.job_dir(run_dir, 'check') / SUBMITTED_FILE
    job_ids = json.loads(submitted.read_text())['job_ids'] if submitted.exists() else []
    runs = defaultdict(list)
    for path in (run_dir / LOG_DIR).glob('*.json'):
        entry = json.loads(path.read_text())
        runs[entry['label']].append(entry)
    sdir = layout.system_dir(run_dir, SYSTEM)
    failed = False
    codes = Counter()
    per_job, per_host = Counter(), Counter()
    not_run = []
    print(f'{len(manifest.calculations)} calculations, {len(job_ids)} jobs submitted')
    for _, calc_id in manifest.calculations:
        label = calc_id.split('/')[1]
        record = store.read_record(sdir, calc_id)
        path = layout.calc_dir(run_dir, SYSTEM, calc_id)
        entries = runs.get(label, [])
        if len(entries) > 1 or len(record.attempts) > 1:
            failed = True
            print(f'  {label}: RUN {len(entries)} TIMES, {len(record.attempts)} attempts')
        if not entries:
            not_run.append(label)
            continue
        entry = entries[0]
        per_job[entry['job_id']] += 1
        per_host[entry['host']] += 1
        claim = json.loads((path / layout.CLAIM_FILE).read_text())
        attempt = record.attempts[-1] if record.attempts else None
        recorded = None if attempt is None else attempt.exit_code
        codes[(entry['exit'], recorded)] += 1
        if attempt is not None and (claim['job_id'] != attempt.job_id or claim['n'] != attempt.n):
            failed = True
            print(f'  {label}: claim {claim} does not match attempt {attempt.n} '
                  f'of job {attempt.job_id}')
    print('Exit codes, fake engine -> seen by the worker:')
    for (given, seen), count in sorted(codes.items(), key=lambda item: (item[0][0], str(item[0][1]))):
        note = '' if given == seen else '   <- changed by the launcher'
        print(f'  {given} -> {seen}: {count} calculations{note}')
    print('Calculations per job: ' + ', '.join(f'{j}: {c}' for j, c in sorted(per_job.items(),
                                                                             key=str)))
    print('Calculations per host: ' + ', '.join(f'{h}: {c}' for h, c in sorted(per_host.items())))
    by_job = defaultdict(list)
    for entries in runs.values():
        for e in entries:
            by_job[e['job_id']].append((e['started'], e['max_seconds']))
    print('max_seconds written for each job, in the order of the runs:')
    for job_id, values in sorted(by_job.items(), key=str):
        print(f'  {job_id}: {", ".join(str(m) for _, m in sorted(values))}')
    intervals = sorted((e['started'], e['ended'], e['job_id']) for es in runs.values() for e in es)
    overlaps = sum(1 for (s1, e1, j1), (s2, e2, j2) in zip(intervals, intervals[1:])
                   if s2 < e1 and j1 != j2)
    print(f'Runs of different jobs that overlap in time (consecutive pairs): {overlaps}')
    if not_run:
        print(f'Not run ({len(not_run)}): {", ".join(not_run)} (time over, or jobs not finished)')
    print('RESULT: ' + ('a calculation ran more than once' if failed
                        else 'each calculation ran at most once'))
    return 1 if failed else 0


def locks(args):
    lock_dir = Path(args.dir)
    (lock_dir / 'locks').mkdir(parents=True, exist_ok=True)
    (lock_dir / 'results').mkdir(parents=True, exist_ok=True)
    if args.start_at:
        time.sleep(max(0.0, args.start_at - time.time()))
    names = list(range(args.names))
    random.shuffle(names)
    taken = []
    for i in names:
        path = lock_dir / 'locks' / f'{args.method}-{i:05d}'
        try:
            if args.method == 'excl':
                os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666))
            else:
                os.mkdir(path)
        except FileExistsError:
            continue
        taken.append(i)
    me = f'{socket.gethostname()}-{os.getpid()}'
    result = lock_dir / 'results' / f'{args.method}-{me}.json'
    result.write_text(json.dumps({'method': args.method, 'host': socket.gethostname(),
                                  'pid': os.getpid(), 'taken': taken}))


def locks_report(args):
    results = [json.loads(p.read_text()) for p in (Path(args.dir) / 'results').glob('*.json')]
    failed = False
    for method in sorted({r['method'] for r in results}):
        mine = [r for r in results if r['method'] == method]
        owners = Counter(i for r in mine for i in r['taken'])
        names = len(list((Path(args.dir) / 'locks').glob(f'{method}-*')))
        twice = sorted(i for i, c in owners.items() if c > 1)
        hosts = sorted({r['host'] for r in mine})
        print(f'{method}: {len(mine)} contenders on {len(hosts)} hosts ({", ".join(hosts)}), '
              f'{names} locks, {len(owners)} taken, {len(twice)} with more than one owner')
        if twice:
            failed = True
            print(f'  more than one owner: {twice[:20]}')
    print('RESULT: ' + ('a lock had more than one owner' if failed
                        else 'every lock had exactly one owner'))
    return 1 if failed else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare', help='write the fake run and submit its jobs')
    p.add_argument('run', help='new directory of the run, on the shared filesystem')
    p.add_argument('--jobscript', required=True, help='jobscript of the machine, without a '
                   'launch line; the worker line is added to a copy')
    p.add_argument('--scheduler', default='slurm')
    p.add_argument('--launcher', default='', help='prefix of the command, e.g. "srun" or '
                   '"mpirun -np 4"; empty: the fake engine runs directly')
    p.add_argument('--calcs', type=int, default=40)
    p.add_argument('--groups', type=int, default=8, help='number of jobs')
    p.add_argument('--walltime', type=int, default=1800, help='seconds, as in the jobscript')
    p.add_argument('--margin', type=int, default=600, help='max_seconds_margin, seconds')
    p.add_argument('--sleep', type=float, default=5.0, help='seconds of each fake run')
    p.add_argument('--submit', action='store_true', help='submit; otherwise a dry run')
    p = sub.add_parser('report', help='check a run whose jobs are over')
    p.add_argument('run')
    p = sub.add_parser('locks', help='one contender of the lock check')
    p.add_argument('dir', help='directory of the locks, on the shared filesystem')
    p.add_argument('--method', choices=('excl', 'mkdir'), required=True)
    p.add_argument('--names', type=int, default=500)
    p.add_argument('--start-at', type=float, default=None,
                   help='epoch time to wait for, so that all contenders start together')
    p = sub.add_parser('locks-report', help='check the locks taken')
    p.add_argument('dir')
    args = parser.parse_args(argv)
    return {'prepare': prepare, 'report': report, 'locks': locks,
            'locks-report': locks_report}[args.command](args) or 0


if __name__ == '__main__':
    sys.exit(main())
