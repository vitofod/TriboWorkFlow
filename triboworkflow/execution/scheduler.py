# Author: Vito Foderà
# Created: 2026-10-27

"""Job submission, listing, state and cancellation for several batch schedulers.

``JobScheduler`` runs the commands of one scheduler (``slurm``, ``pbs``,
``torque``, ``lsf``, ``sge``, ``htcondor`` and its alias ``condor``) from the
table ``SCHEDULER_CONFIG``, or from a table given as a dict. It submits a
jobscript and returns the job id, cancels a job, lists the active jobs of the
current user, also split into running and queued, and reports the state of
one job.

Job ids are strings, exactly as the submission prints them (e.g.
``"12345"``, or ``"12345.server"`` for PBS and Torque). The listings may print
them in another form (PBS and Torque ids without the server part, HTCondor
ids as ``cluster.process``), so ids are compared with ``JobScheduler.job_key``.

The listings of Slurm, LSF and SGE are read by column position. Those of PBS
and Torque read the named attributes of ``qstat -f`` (JSON for PBS, one
``name = value`` line per attribute for Torque) and keep the jobs whose owner
is the current user; the one of HTCondor reads ``condor_q -af:j JobStatus``.
A listing that cannot be read raises an error instead of returning lists that
look empty; so does the split into running and queued jobs of a custom table.

``get_job_state`` gives the state of one job as the scheduler prints it, and
whether the job is active (queued or running), ended, or in a state not
known (``SCHEDULER_STATES``). Where the scheduler keeps the finished jobs the
state also comes from them: Slurm ``sacct``, PBS ``qstat -x``, HTCondor
``condor_history`` when ``condor_q`` no longer lists the job. Torque shows a
completed job only as long as the server keeps it. For LSF and SGE the state
is read from the listing of active jobs, so a job missing from it has no
state: its absence does not tell whether or how it ended.

With ``dry_run=True`` no command is run: each command is appended to
``JobScheduler.commands``, ``submit_job`` returns None, the listings are
empty and the states of the jobs are not known.

Notes
-----
Ported from Xsorb (``xsorb/io/scheduler.py``, CC BY 4.0), which gives the
table of the schedulers, the commands, the parsing of the column listings and
the error messages. The column listings depend on the output format
configured on the machine. The listings of PBS, Torque and HTCondor, the
listing command of SGE and the states of single jobs are not those of Xsorb:
their commands follow the manuals of Slurm, OpenPBS, Torque, SGE and HTCondor
and have not been tried on a cluster.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass

# Commands of each scheduler. ``list_active_jobs`` lists the running and queued
# jobs of the user ($USER is replaced by the user name), read by columns unless
# ``listing_format`` names a format of named attributes; ``job_state`` and
# ``job_history`` give the state of one job ($JOB_ID is replaced by its id);
# ``submit_method = "stdin"`` passes the jobscript on the standard input.
# ``test_cmd`` is part of the original table and is not used.
SCHEDULER_CONFIG = {
    "slurm": {
        "submit": ["sbatch"],
        "cancel": ["scancel"],
        "id_regex": r"Submitted batch job (\d+)",
        "id_format": r"^\d+$",
        "test_cmd": "sbatch",
        "list_active_jobs": ["squeue", "-h", "-u", "$USER", "-t", "RUNNING,PENDING,CONFIGURING"],
        "job_state": ["sacct", "-X", "-n", "-o", "State%30", "-j", "$JOB_ID"]
    },
    "pbs": {
        "submit": ["qsub"],
        "cancel": ["qdel"],
        "id_regex": r"(\d+(?:\.\S+)?)",
        "id_format": r"^\d+(?:\.\S+)?$",
        "test_cmd": "qsub",
        "list_active_jobs": ["qstat", "-f", "-F", "json"],
        "listing_format": "pbs_json",
        "job_state": ["qstat", "-f", "-F", "json", "-x", "$JOB_ID"]
    },
    "torque": {
        "submit": ["qsub"],
        "cancel": ["qdel"],
        "id_regex": r"(\d+(?:\.\S+)?)",
        "id_format": r"^\d+(?:\.\S+)?$",
        "test_cmd": "qsub",
        "list_active_jobs": ["qstat", "-f", "-1"],
        "listing_format": "qstat_full",
        "job_state": ["qstat", "-f", "-1", "$JOB_ID"]
    },
    "lsf": {
        "submit": ["bsub"],
        "cancel": ["bkill"],
        "id_regex": r"Job <(\d+)> is submitted",
        "id_format": r"^\d+$",
        "test_cmd": "bsub",
        "submit_method": "stdin",
        "list_active_jobs": ["bjobs", "-u", "$USER"]
    },
    "sge": {
        "submit": ["qsub"],
        "cancel": ["qdel"],
        "id_regex": r"Your job (\d+)",
        "id_format": r"^\d+$",
        "test_cmd": "qsub",
        # Default states of qstat, i.e. -s psr; the -s r,qw of Xsorb is not in the manual.
        "list_active_jobs": ["qstat", "-u", "$USER"],
        "hold_listing": ["qstat", "-u", "$USER", "-s", "$HOLD"]
    },
    "htcondor": {
        "submit": ["condor_submit"],
        "cancel": ["condor_rm"],
        "id_regex": r"submitted to cluster (\d+)",
        "id_format": r"^\d+$",
        "test_cmd": "condor_submit",
        "list_active_jobs": ["condor_q", "$USER", "-af:j", "JobStatus"],
        "listing_format": "condor_af",
        "job_state": ["condor_q", "$JOB_ID", "-af:j", "JobStatus"],
        "job_history": ["condor_history", "$JOB_ID", "-af:j", "JobStatus"]
    },
    "condor": {  # alias of htcondor
        "submit": ["condor_submit"],
        "cancel": ["condor_rm"],
        "id_regex": r"submitted to cluster (\d+)",
        "id_format": r"^\d+$",
        "test_cmd": "condor_submit",
        "list_active_jobs": ["condor_q", "$USER", "-af:j", "JobStatus"],
        "listing_format": "condor_af",
        "job_state": ["condor_q", "$JOB_ID", "-af:j", "JobStatus"],
        "job_history": ["condor_history", "$JOB_ID", "-af:j", "JobStatus"]
    }
}

# Schedulers whose ids have a part after a dot: the server for PBS and Torque,
# the process of the cluster for HTCondor.
_DOTTED_IDS = ("pbs", "torque", "htcondor", "condor")

# Running, queued and active states of the listings of named attributes: for
# PBS and Torque the states of the column listing of Xsorb; every job listed by
# ``condor_q`` is active (None).
_LISTED_STATES = {
    "pbs_json": ({"R"}, {"Q"}, {"R", "Q", "H", "T", "W", "S"}),
    "qstat_full": ({"R"}, {"Q"}, {"R", "Q", "H", "T", "W", "S"}),
    "condor_af": ({"2"}, {"1", "5"}, None),
}

# Active (queued or running) and ended states of a single job, from the manuals
# of the schedulers; a state in neither set is not known. Slurm prints a state
# flag in place of the base state when one is set; the active ones are those
# kept active by SCS. PREEMPTED is not active, but Slurm may requeue the job,
# which is then active again. LSF and SGE are read from the listing of active
# jobs only, with the states of Xsorb.
SCHEDULER_STATES = {
    "slurm": (frozenset({"PENDING", "RUNNING", "REQUEUED", "RESIZING", "SUSPENDED",
                         "CONFIGURING", "COMPLETING", "EXPEDITING", "UPDATE_DB"}),
              frozenset({"BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED",
                         "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "TIMEOUT"})),
    "pbs": (frozenset({"B", "E", "H", "Q", "R", "S", "T", "U", "W"}), frozenset({"F"})),
    "torque": (frozenset({"E", "H", "Q", "R", "S", "T", "W"}), frozenset({"C"})),
    "lsf": (frozenset({"RUN", "PEND", "PSUSP", "USUSP", "SSUSP"}), frozenset()),
    "sge": (frozenset({"r", "t", "s", "qw", "hqw", "hRwq"}), frozenset()),
    "htcondor": (frozenset({"1", "2", "5", "6", "7"}), frozenset({"3", "4"})),
    "condor": (frozenset({"1", "2", "5", "6", "7"}), frozenset({"3", "4"})),
}

# Ended states after which a job may become active again: Slurm may requeue a
# preempted job, depending on the configured PreemptMode.
REVERSIBLE_STATES = {"slurm": frozenset({"PREEMPTED"})}

# States of held jobs, which stay in the queue and do not run until the hold is
# released: HTCondor ``JobStatus`` 5, PBS and Torque ``H``. For SGE every state
# with ``h`` (``_SGE_HOLD_LETTER``).
HELD_STATES = {"htcondor": frozenset({"5"}), "condor": frozenset({"5"}),
               "pbs": frozenset({"H"}), "torque": frozenset({"H"})}
_SGE_HOLD_LETTER = "h"

# Hold types of SGE, selected one at a time with ``qstat -s``, and their
# reason: user, operator and system holds need an intervention; array and job
# dependencies and the start time asked with ``qsub -a`` are released by SGE
# itself. The first reason of ``HOLD_REASONS`` found wins.
SGE_HOLD_TYPES = {"hu": "intervention", "ho": "intervention", "hs": "intervention",
                  "hd": "dependency", "hj": "dependency", "ha": "start_time"}
HOLD_REASONS = ("intervention", "dependency", "start_time")

# Column of the state in the column listings of LSF and SGE.
_STATE_COLUMN = {"lsf": 2, "sge": 4}

_DIGITS = re.compile(r"[0-9]+")
_CONDOR_AF_LINE = re.compile(r"(\d+\.\d+)\s+(\S+)")


class JobSchedulerError(Exception):
    """A scheduler command that is missing, fails or prints an unexpected output."""


class JobSubmissionError(JobSchedulerError):
    """A submission command that ran and failed, with whether the job may exist.

    Attributes
    ----------
    accepted : False or None
        False if the job was surely not created: the command exited with an
        error and printed no job id. None if it may have been: the command
        timed out, failed in an unexpected way, exited with an error after
        printing a job id, or succeeded without printing one.
    """

    def __init__(self, message, accepted):
        super().__init__(message)
        self.accepted = accepted


@dataclass(frozen=True)
class JobState:
    """State of one job, as reported by the scheduler.

    Attributes
    ----------
    job_id : str
        Id of the job, as returned by ``JobScheduler.submit_job``.
    state : str or None
        State as printed by the scheduler, e.g. ``"TIMEOUT"`` or
        ``"CANCELLED+"`` (Slurm), ``"F"`` (PBS), ``"4"`` (HTCondor
        ``JobStatus``; one value per process, joined by commas, if the
        cluster has several). None if the scheduler reports nothing about the
        job.
    active : bool or None
        True if the job is queued or running, False if it ended, None if its
        state is not known.
    held : bool
        True if the job is held in the queue (``JobScheduler.is_held``).
    hold_reason : str or None
        For a held job, ``"intervention"``, ``"dependency"`` or
        ``"start_time"`` when the scheduler tells it (SGE), None when only
        the state is known.
    hold_detail : str or None
        For a held job, what the scheduler says about the hold, as printed:
        the SGE hold types that list the job (e.g. ``"hj"``), the
        ``Hold_Types`` of PBS and Torque; None if nothing is said.
    """

    job_id: str
    state: str | None
    active: bool | None
    held: bool = False
    hold_reason: str | None = None
    hold_detail: str | None = None


class JobScheduler:
    """Commands of one batch scheduler.

    Parameters
    ----------
    scheduler : str or dict
        Name of a scheduler of ``SCHEDULER_CONFIG`` (any case), or a table with
        the same keys; a dict is stored as ``SCHEDULER_CONFIG["custom"]`` and
        the scheduler is then named ``"custom"``.
    logger : logging.Logger or None, optional
        Logger of the messages; the logger of this module by default.
    timeout : int, optional
        Seconds after which a scheduler command is stopped.
    dry_run : bool, optional
        If true, commands are recorded in ``commands`` and not run.

    Attributes
    ----------
    scheduler_name : str
        Lower-case name of the scheduler.
    config : dict
        Table of the scheduler.
    commands : list of tuple
        In a dry run, ``(command, stdin, cwd)`` of each command not run.

    Raises
    ------
    JobSchedulerError
        If the name is not in ``SCHEDULER_CONFIG``.
    """

    def __init__(self, scheduler, logger=None, timeout=30, dry_run=False):
        if isinstance(scheduler, dict):
            SCHEDULER_CONFIG["custom"] = scheduler
            scheduler = "custom"

        self.scheduler_name = scheduler.lower()
        self.logger = logger or logging.getLogger(__name__)
        self.timeout = timeout
        self.dry_run = dry_run
        self.commands = []

        if self.scheduler_name not in SCHEDULER_CONFIG:
            raise JobSchedulerError(
                f"Scheduler '{scheduler}' is not supported. "
                f"Supported schedulers: {list(SCHEDULER_CONFIG.keys())}. "
                "Please provide a valid scheduler name or a custom configuration."
            )

        self.config = SCHEDULER_CONFIG[self.scheduler_name]

    def _run_command(self, cmd, input_data=None, cwd=None):
        """Run a command and return its ``subprocess.CompletedProcess``.

        Parameters
        ----------
        cmd : list of str
            Command and arguments.
        input_data : str or None, optional
            Text written on the standard input.
        cwd : str or os.PathLike or None, optional
            Directory the command runs in; the current one by default.

        Raises
        ------
        JobSchedulerError
            If the command is not found, exits with an error or times out. A
            missing command keeps the ``FileNotFoundError`` as ``__cause__``.
        """
        if self.dry_run:
            self.commands.append((cmd, input_data, cwd))
            self.logger.debug(f"Dry run, not executing: {' '.join(cmd)}")
            return subprocess.CompletedProcess(cmd, 0, stdout='', stderr='')
        try:
            self.logger.debug(f"Executing command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                input=input_data,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=True,
                cwd=cwd
            )

            return result

        except subprocess.CalledProcessError as e:
            raise JobSchedulerError(
                f"Command failed with return code {e.returncode}: {e.stderr.strip()}"
            ) from e
        except subprocess.TimeoutExpired as e:
            raise JobSchedulerError(f"Command timed out after {self.timeout} seconds") from e
        except FileNotFoundError as e:
            raise JobSchedulerError(f"Command '{cmd[0]}' not found") from e
        except Exception as e:
            raise JobSchedulerError(f"Unexpected error: {str(e)}") from e

    def _extract_job_id(self, output):
        """Job id printed by the submission, as a string, or None."""
        regex = self.config["id_regex"]
        match = re.search(regex, output)
        return match.group(1) if match else None

    def _validate_job_id(self, job_id):
        """Whether ``job_id`` has the id format of the scheduler."""
        pattern = self.config["id_format"]
        return bool(re.match(pattern, str(job_id)))

    def _build_command(self, operation, argument=None, extra_args=None):
        """Command of ``operation`` (a key of the table) with its arguments."""
        cmd = self.config[operation].copy()

        if argument:
            cmd.append(argument)

        if extra_args:
            cmd.extend(extra_args)

        return cmd

    def _user_name(self):
        """Name of the current user, from $USER or $USERNAME."""
        username = os.getenv("USER") or os.getenv("USERNAME")
        if not username:
            raise JobSchedulerError("Cannot determine current username")
        return username

    def _list_command(self):
        """Command listing the active jobs, with $USER replaced by the user name."""
        if "list_active_jobs" not in self.config:
            raise JobSchedulerError(f"Active job listing not supported for {self.scheduler_name}")

        cmd = self.config["list_active_jobs"].copy()

        if "$USER" in cmd:
            username = self._user_name()
            cmd = [arg.replace("$USER", username) if arg == "$USER" else arg for arg in cmd]
        return cmd

    def _job_command(self, operation, job_id):
        """Command of ``operation`` (``job_state`` or ``job_history``) for one job."""
        return [job_id if arg == "$JOB_ID" else arg for arg in self.config[operation]]

    def _listed_job_id(self, field):
        """Job id in the first column of a job listing, or None for other lines."""
        if self.scheduler_name in _DOTTED_IDS and "." in field:
            field = field.split(".")[0]
        return field if _DIGITS.fullmatch(field) else None

    def job_key(self, job_id):
        """Form of a job id that matches the job listings.

        Parameters
        ----------
        job_id : str
            Id returned by ``submit_job`` or found in a listing.

        Returns
        -------
        str
            For PBS and Torque the part before the first dot (the id without
            the server), for HTCondor the cluster without the process,
            otherwise the id itself.
        """
        job_id = str(job_id)
        if self.scheduler_name in _DOTTED_IDS:
            return job_id.split(".")[0]
        return job_id

    def submit_job(self, script_path, script_args=None, cwd=None):
        """Submit a jobscript and return the job id.

        Parameters
        ----------
        script_path : str
            Jobscript, made absolute before the submission.
        script_args : list of str or None, optional
            Arguments of the jobscript. With ``submit_method = "stdin"`` (LSF)
            they are exported as ``ARG1``, ``ARG2``, ... at the top of the
            jobscript passed on the standard input.
        cwd : str or os.PathLike or None, optional
            Directory the submission runs in; the current one by default.

        Returns
        -------
        str or None
            Job id as printed by the scheduler; None in a dry run.

        Raises
        ------
        JobSchedulerError
            If the jobscript is missing or unreadable, the arguments are not a
            list of strings (nothing is run), or the command is not found.
        JobSubmissionError
            If the command fails or no id is found in its output; its
            ``accepted`` tells whether the job may have been created.
        """
        if not script_path or not isinstance(script_path, str):
            raise JobSchedulerError("Script path must be a non-empty string")

        script_path = os.path.abspath(script_path)
        if not os.path.exists(script_path):
            raise JobSchedulerError(f"Job script '{script_path}' not found")

        if not os.access(script_path, os.R_OK):
            raise JobSchedulerError(f"Job script '{script_path}' is not readable")

        if script_args is not None:
            if not isinstance(script_args, list):
                raise JobSchedulerError("script_args must be a list of strings")
            if not all(isinstance(arg, str) for arg in script_args):
                raise JobSchedulerError("All script_args must be strings")

        if self.config.get("submit_method") == "stdin":
            cmd = self._build_command("submit")

            with open(script_path, 'r') as f:
                script_content = f.read()

            if script_args:
                env_vars = "\n".join([f"export ARG{i+1}={arg}" for i, arg in enumerate(script_args)])
                script_content = f"{env_vars}\n{script_content}"
                self.logger.debug(f"LSF: Added environment variables for script arguments: {script_args}")

            result = self._submit(cmd, input_data=script_content, cwd=cwd)
        else:
            cmd = self._build_command("submit", script_path, script_args)
            result = self._submit(cmd, cwd=cwd)

        if self.dry_run:
            return None

        job_id = self._extract_job_id(result.stdout)
        if not job_id:
            raise JobSubmissionError(
                f"Could not extract job ID from output: {result.stdout.strip()}", accepted=None)

        log_msg = f"Submitted job {job_id}"
        if script_args and logging.getLogger().isEnabledFor(logging.DEBUG):
            log_msg += f" with arguments: {' '.join(script_args)}"
        self.logger.info(log_msg)
        return job_id

    def _submit(self, cmd, input_data=None, cwd=None):
        """Run a submission command; a failure tells whether the job may exist.

        A missing command passes unchanged (nothing ran, see ``hpc_only``).
        An exit with an error and no job id in the output surely created no
        job; any other failure (a timeout, an unexpected error, an exit with
        an error after printing a job id) may have created one.
        """
        try:
            return self._run_command(cmd, input_data=input_data, cwd=cwd)
        except JobSchedulerError as e:
            cause = e.__cause__
            if isinstance(cause, FileNotFoundError):
                raise
            refused = (isinstance(cause, subprocess.CalledProcessError)
                       and self._extract_job_id(cause.stdout or "") is None)
            raise JobSubmissionError(str(e), accepted=False if refused else None) from cause

    def cancel_job(self, job_id):
        """Cancel a job.

        Parameters
        ----------
        job_id : str
            Id of the job.

        Returns
        -------
        bool
            Whether the cancel command succeeded; a failure is logged.

        Raises
        ------
        JobSchedulerError
            If the id is empty or does not have the id format of the scheduler.
        """
        if not job_id:
            raise JobSchedulerError("Job ID cannot be empty")

        if not self._validate_job_id(job_id):
            raise JobSchedulerError(f"Invalid job ID format: {job_id}")

        cmd = self._build_command("cancel", str(job_id))

        try:
            self._run_command(cmd)
            self.logger.info(f"Cancelled job {job_id}")
            return True
        except JobSchedulerError as e:
            self.logger.error(f"Failed to cancel job {job_id}: {str(e)}")
            return False

    def get_active_job_ids(self):
        """Ids of the running and queued jobs of the current user.

        Returns
        -------
        list of str
            Ids in the form of the listing (see ``job_key``).

        Raises
        ------
        JobSchedulerError
            If the user name is unknown, the listing command fails or its
            output is not in the expected format.
        """
        cmd = self._list_command()

        try:
            result = self._run_command(cmd)
            job_ids = self._extract_active_job_ids(result.stdout)

            self.logger.debug(f"Found {len(job_ids)} active jobs for user")
            return job_ids

        except JobSchedulerError as e:
            self.logger.error(f"Failed to get active jobs: {str(e)}")
            raise

    def get_lists_of_running_and_queued_job_ids(self):
        """Ids of the running jobs and of the queued jobs of the current user.

        The state is read from a fixed column of the listing (the third for
        LSF, the fifth for Slurm and SGE), whose lines not starting with a
        job id are skipped; from ``job_state`` for PBS and Torque, from
        ``JobStatus`` for HTCondor.

        Returns
        -------
        running, queued : list of str
            Ids in the form of the listing (see ``job_key``).

        Raises
        ------
        JobSchedulerError
            If the scheduler is a custom table (whose states are not known),
            the user name is unknown, the listing command fails or its output
            is not in the expected format.
        """
        if self.scheduler_name == "custom":
            raise JobSchedulerError("Running and queued jobs cannot be told apart for a custom "
                                    "scheduler: the column of the state is not known")
        cmd = self._list_command()

        try:
            result = self._run_command(cmd)
            running_jobs, queued_jobs = self._split_running_and_queued(result.stdout)

            self.logger.debug(f"Found {len(running_jobs)} running and {len(queued_jobs)} queued jobs for user")
            return running_jobs, queued_jobs

        except JobSchedulerError as e:
            self.logger.error(f"Failed to get active jobs: {str(e)}")
            raise

    def _split_running_and_queued(self, output):
        """Running and queued job ids in the output of the listing command.

        The state is read from the third column for LSF, the fifth for Slurm
        and SGE, from the named attributes for PBS, Torque and HTCondor.
        """
        if "listing_format" in self.config:
            running_states, queued_states, _ = _LISTED_STATES[self.config["listing_format"]]
            jobs = self._listed_jobs(output)
            return (_unique(job_id for job_id, state in jobs if state in running_states),
                    _unique(job_id for job_id, state in jobs if state in queued_states))

        running_jobs = []
        queued_jobs = []

        lines = output.strip().split('\n')
        for line in lines:
            if not line.strip():
                continue

            fields = line.split()
            if not fields:
                continue

            job_id = self._listed_job_id(fields[0])
            if job_id is None:
                continue

            if self.scheduler_name == "lsf":
                if len(fields) >= 3:
                    status = fields[2]
                    if status == "RUN":
                        running_jobs.append(job_id)
                    elif status == "PEND":
                        queued_jobs.append(job_id)
            elif self.scheduler_name == "slurm":
                if len(fields) >= 5:
                    status = fields[4]  # ST column of squeue
                    if status in ["R", "CG"]:  # running, completing
                        running_jobs.append(job_id)
                    elif status in ["PD", "CF", "S"]:  # pending, configuring, suspended
                        queued_jobs.append(job_id)
            elif self.scheduler_name == "sge":
                if len(fields) >= 5:
                    status = fields[4]  # state column of qstat
                    if status in ["r", "t", "s"]:  # running, transferring, suspended
                        running_jobs.append(job_id)
                    elif status in ["qw", "hqw", "hRwq"]:  # queued waiting, on hold
                        queued_jobs.append(job_id)

        return running_jobs, queued_jobs

    def _extract_active_job_ids(self, output):
        """Active job ids in the output of the listing command.

        PBS, Torque and LSF also list finished jobs, which are filtered by the
        state; the listings of the other schedulers hold active jobs only.
        """
        if "listing_format" in self.config:
            _, _, active_states = _LISTED_STATES[self.config["listing_format"]]
            return _unique(job_id for job_id, state in self._listed_jobs(output)
                           if active_states is None or state in active_states)

        job_ids = []
        lines = output.strip().split('\n')

        for line in lines:
            if not line.strip():
                continue

            fields = line.split()
            if not fields:
                continue

            job_id = self._listed_job_id(fields[0])
            if job_id is None:
                continue

            if self._validate_job_id(job_id):
                if self.scheduler_name == "lsf":
                    if len(fields) >= 3 and fields[2] in ["RUN", "PEND", "PSUSP", "USUSP", "SSUSP"]:
                        job_ids.append(job_id)
                else:
                    job_ids.append(job_id)

        return job_ids

    def _listed_jobs(self, output):
        """Ids (see ``job_key``) and states of the jobs in a listing of named attributes.

        ``qstat -f`` lists the jobs of every user: for PBS and Torque only
        those whose ``Job_Owner`` (``user@host``) is the current user are kept.

        Raises
        ------
        JobSchedulerError
            If the output is not in the expected format, or a PBS or Torque
            job has no ``Job_Owner`` or ``job_state``.
        """
        if self.dry_run:
            return []
        listing_format = self.config["listing_format"]
        if listing_format == "condor_af":
            return [(self.job_key(job_id), state) for job_id, state in _read_condor_af(output)]

        user = self._user_name()
        listed = []
        for job_id, attributes in _read_qstat(listing_format, output):
            owner, state = attributes.get("Job_Owner"), attributes.get("job_state")
            if owner is None or state is None:
                raise JobSchedulerError(
                    f"Job {job_id} without Job_Owner or job_state in the output of "
                    f"'{' '.join(self.config['list_active_jobs'])}'")
            if owner.split("@")[0] == user:
                listed.append((self.job_key(job_id), state))
        return listed

    def get_job_state(self, job_id):
        """State of one job, as the scheduler reports it.

        Slurm: ``sacct``, which also keeps the finished jobs (the most recent
        record of the id, so the current one of a requeued job), with a state
        column of 30 characters, where ``CANCELLED by <uid>`` fits. A state
        that does not fit is cut and followed by ``+``: only the part before
        ``+`` that is a whole name of ``SCHEDULER_STATES`` is classified (e.g.
        ``CANCELLED+``), so a name cut short (e.g. ``OUT_OF_ME+``) is not
        known. PBS: ``qstat -x``, which also shows the finished
        jobs. Torque: ``qstat``, which shows a completed job as long as the
        server keeps it. HTCondor: ``condor_q``, then ``condor_history`` if
        the queue no longer has the job. LSF and SGE: the listing of active
        jobs. For a held job (``is_held``) SGE gives the reason, from the
        listing of each hold type (``SGE_HOLD_TYPES``); PBS and Torque give
        ``Hold_Types`` as detail, not interpreted.

        Parameters
        ----------
        job_id : str
            Id returned by ``submit_job``.

        Returns
        -------
        JobState
            ``active`` from ``SCHEDULER_STATES``. A job that the scheduler
            does not report (missing from the history, or from the listing
            for LSF and SGE) has state and ``active`` None: its absence does
            not tell whether or how it ended.

        Raises
        ------
        JobSchedulerError
            If the scheduler is a custom table, a command fails (for PBS and
            Torque also when the server no longer knows the job) or its
            output is not in the expected format.
        """
        if self.scheduler_name == "custom":
            raise JobSchedulerError("The state of a job is not known for a custom scheduler: "
                                    "its commands and states are not known")
        job_id = str(job_id)
        attributes = {}
        if "job_state" not in self.config:
            state = self._state_from_listing(job_id)
        else:
            state, attributes = self._query_state("job_state", job_id)
            if state is None and "job_history" in self.config:
                state, attributes = self._query_state("job_history", job_id)

        held = self.is_held(state)
        reason = detail = None
        if held and self.scheduler_name == "sge":
            types = self._sge_hold_types(job_id)
            reasons = {SGE_HOLD_TYPES[t] for t in types}
            reason = next((r for r in HOLD_REASONS if r in reasons), None)
            detail = ",".join(types) or None
        elif held:
            detail = attributes.get("Hold_Types")

        job_state = JobState(job_id, state, self._is_active(state), held, reason, detail)
        self.logger.debug(f"State of job {job_id}: {job_state.state} (active: {job_state.active})")
        return job_state

    def _query_state(self, operation, job_id):
        """State printed by the command ``operation`` for one job, or None, and the
        attributes of the job (PBS and Torque; otherwise empty)."""
        cmd = self._job_command(operation, job_id)
        output = self._run_command(cmd).stdout
        if self.dry_run:
            return None, {}
        key = self.job_key(job_id)

        if self.scheduler_name == "slurm":
            lines = [line.strip() for line in output.splitlines() if line.strip()]
            if len(lines) > 1:
                raise JobSchedulerError(
                    f"More than one record for job {job_id} in the output of '{' '.join(cmd)}'")
            return (lines[0] if lines else None), {}

        if self.scheduler_name in ("htcondor", "condor"):
            states = [state for listed, state in _read_condor_af(output)
                      if self.job_key(listed) == key]
            return (",".join(states) if states else None), {}

        for listed, attributes in _read_qstat(self.config["listing_format"], output):
            if self.job_key(listed) == key:
                if "job_state" not in attributes:
                    raise JobSchedulerError(
                        f"Job {job_id} without job_state in the output of '{' '.join(cmd)}'")
                return attributes["job_state"], attributes
        return None, {}

    def _sge_hold_types(self, job_id):
        """SGE hold types (``SGE_HOLD_TYPES``) whose listing has the job, in order."""
        key, types = self.job_key(job_id), []
        user = self._user_name()
        for hold in SGE_HOLD_TYPES:
            cmd = [user if arg == "$USER" else hold if arg == "$HOLD" else arg
                   for arg in self.config["hold_listing"]]
            output = self._run_command(cmd).stdout
            if any(fields and self._listed_job_id(fields[0]) == key
                   for fields in (line.split() for line in output.splitlines())):
                types.append(hold)
        return types

    def _state_from_listing(self, job_id):
        """State of one job in the column listing of active jobs (LSF, SGE), or None."""
        output = self._run_command(self._list_command()).stdout
        key, column = self.job_key(job_id), _STATE_COLUMN[self.scheduler_name]
        for line in output.splitlines():
            fields = line.split()
            if fields and self._listed_job_id(fields[0]) == key and len(fields) > column:
                return fields[column]
        return None

    def is_held(self, state):
        """Whether a job in ``state`` is held in the queue.

        True for a state of ``HELD_STATES`` (for an HTCondor cluster, if any
        of its processes is held) and, for SGE, a state with ``h``; False
        otherwise, and for None. A held job is not released by this module.
        """
        if state is None:
            return False
        if self.scheduler_name == "sge":
            return _SGE_HOLD_LETTER in state
        held = HELD_STATES.get(self.scheduler_name, ())
        return any(value in held for value in state.split(","))

    def is_final(self, state):
        """Whether a job in ``state`` (as in ``JobState.state``) can no longer change.

        Parameters
        ----------
        state : str or None
            State printed by the scheduler.

        Returns
        -------
        bool
            True for an ended state, except those after which the scheduler
            may requeue the job (``REVERSIBLE_STATES``); False for an active
            state, a state not known, and None.
        """
        if self._is_active(state) is not False:
            return False
        name = state.split()[0].rstrip("+")
        return name not in REVERSIBLE_STATES.get(self.scheduler_name, ())

    def _is_active(self, state):
        """True, False or None for a state printed by the scheduler (``SCHEDULER_STATES``)."""
        if state is None:
            return None
        active_states, ended_states = SCHEDULER_STATES[self.scheduler_name]

        def classify(value):
            if value in active_states:
                return True
            return False if value in ended_states else None

        if self.scheduler_name == "slurm":
            return classify(state.split()[0].rstrip("+"))
        # One value per process of an HTCondor cluster: active if any is.
        values = [classify(value) for value in state.split(",")]
        if True in values:
            return True
        return False if all(value is False for value in values) else None

    def get_supported_schedulers(self):
        """Names of the schedulers in ``SCHEDULER_CONFIG``."""
        return list(SCHEDULER_CONFIG.keys())

    def __repr__(self):
        return f"JobScheduler(scheduler='{self.scheduler_name}')"


def _unique(job_ids):
    """Job ids in order, without repetitions (the processes of an HTCondor cluster)."""
    return list(dict.fromkeys(job_ids))


def _read_qstat(listing_format, output):
    """(job id, attributes) of each job in the output of ``qstat -f``.

    ``pbs_json``: the table ``Jobs`` of ``qstat -f -F json`` (OpenPBS), missing
    when no job is listed. ``qstat_full``: a ``Job Id: <id>`` line followed by
    one ``name = value`` line per attribute (Torque ``qstat -f -1``, which
    does not fold long values).

    Raises
    ------
    JobSchedulerError
        If the output is not in that format.
    """
    if listing_format == "pbs_json":
        try:
            data = json.loads(output)
        except json.JSONDecodeError as e:
            raise JobSchedulerError(f"Output of qstat is not JSON: {e}") from e
        jobs = data.get("Jobs", {}) if isinstance(data, dict) else None
        if not isinstance(jobs, dict) or not all(isinstance(a, dict) for a in jobs.values()):
            raise JobSchedulerError("Unexpected JSON output of qstat: no table of jobs")
        return list(jobs.items())

    jobs = []
    for line in output.splitlines():
        if not line.strip():
            continue
        if line.startswith("Job Id:"):
            jobs.append((line.split(":", 1)[1].strip(), {}))
        elif jobs and " = " in line:
            name, value = line.split(" = ", 1)
            jobs[-1][1][name.strip()] = value.strip()
        else:
            raise JobSchedulerError(f"Unexpected line in the output of qstat -f: '{line.strip()}'")
    return jobs


def _read_condor_af(output):
    """(job id, JobStatus) of each line printed with ``-af:j JobStatus``.

    Raises
    ------
    JobSchedulerError
        If a line is not ``<cluster>.<process> <JobStatus>``.
    """
    jobs = []
    for line in output.splitlines():
        if not line.strip():
            continue
        match = _CONDOR_AF_LINE.fullmatch(line.strip())
        if match is None:
            raise JobSchedulerError(f"Unexpected line in the output of HTCondor: '{line.strip()}'")
        jobs.append((match[1], match[2]))
    return jobs
