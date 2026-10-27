# Author: Vito Foderà
# Created: 2026-10-27

"""Job submission, listing and cancellation for several batch schedulers.

``JobScheduler`` runs the commands of one scheduler (``slurm``, ``pbs``,
``torque``, ``lsf``, ``sge``, ``htcondor`` and its alias ``condor``) from the
table ``SCHEDULER_CONFIG``, or from a table given as a dict. It submits a
jobscript and returns the job id, cancels a job and lists the active jobs of
the current user, also split into running and queued.

Job ids are strings, exactly as the submission prints them (e.g.
``"12345"``, or ``"12345.server"`` for PBS and Torque). The job listings of
PBS and Torque print the id without the server part, so ids are compared
with ``JobScheduler.job_key``.

With ``dry_run=True`` no command is run: each command is appended to
``JobScheduler.commands``, ``submit_job`` returns None and the listings are
empty.

The job listings are read for ``slurm``, ``lsf`` and ``sge``. For ``pbs``,
``torque``, ``htcondor`` and ``condor`` the columns read do not match the
output of their listing commands (``qstat -u`` prints the number of nodes
where the state is looked for; ``condor_q`` may group the jobs by owner, and
prints ids such as ``591.0``), so a running job would be missing from the
lists. Their listings raise an error instead of returning lists that look
empty; so does the split into running and queued jobs of a custom table.

Notes
-----
Ported from Xsorb (``xsorb/io/scheduler.py``, CC BY 4.0), which gives the
table of the schedulers, the commands, the parsing of the outputs and the
error messages. The job listings are read by column position, so they depend
on the output format configured on the machine.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess

# Commands of each scheduler. ``list_active_jobs`` lists the running and queued
# jobs of the user ($USER is replaced by the user name); ``submit_method =
# "stdin"`` passes the jobscript on the standard input. ``test_cmd`` is part of
# the original table and is not used.
SCHEDULER_CONFIG = {
    "slurm": {
        "submit": ["sbatch"],
        "cancel": ["scancel"],
        "id_regex": r"Submitted batch job (\d+)",
        "id_format": r"^\d+$",
        "test_cmd": "sbatch",
        "list_active_jobs": ["squeue", "-h", "-u", "$USER", "-t", "RUNNING,PENDING,CONFIGURING"]
    },
    "pbs": {
        "submit": ["qsub"],
        "cancel": ["qdel"],
        "id_regex": r"(\d+(?:\.\S+)?)",
        "id_format": r"^\d+(?:\.\S+)?$",
        "test_cmd": "qsub",
        "list_active_jobs": ["qstat", "-u", "$USER"]
    },
    "torque": {
        "submit": ["qsub"],
        "cancel": ["qdel"],
        "id_regex": r"(\d+(?:\.\S+)?)",
        "id_format": r"^\d+(?:\.\S+)?$",
        "test_cmd": "qsub",
        "list_active_jobs": ["qstat", "-u", "$USER"]
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
        "list_active_jobs": ["qstat", "-u", "$USER", "-s", "r,qw"]
    },
    "htcondor": {
        "submit": ["condor_submit"],
        "cancel": ["condor_rm"],
        "id_regex": r"submitted to cluster (\d+)",
        "id_format": r"^\d+$",
        "test_cmd": "condor_submit",
        "list_active_jobs": ["condor_q", "$USER"]
    },
    "condor": {  # alias of htcondor
        "submit": ["condor_submit"],
        "cancel": ["condor_rm"],
        "id_regex": r"submitted to cluster (\d+)",
        "id_format": r"^\d+$",
        "test_cmd": "condor_submit",
        "list_active_jobs": ["condor_q", "$USER"]
    }
}

# Schedulers whose submission prints the id with the server part.
_SERVER_SUFFIX = ("pbs", "torque")

# Schedulers whose job listings cannot be read with the columns of the parser.
UNREADABLE_LISTINGS = ("pbs", "torque", "htcondor", "condor")

_DIGITS = re.compile(r"[0-9]+")


class JobSchedulerError(Exception):
    """A scheduler command that is missing, fails or prints an unexpected output."""


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

    def _list_command(self):
        """Command listing the active jobs, with $USER replaced by the user name."""
        if "list_active_jobs" not in self.config:
            raise JobSchedulerError(f"Active job listing not supported for {self.scheduler_name}")
        if self.scheduler_name in UNREADABLE_LISTINGS:
            raise JobSchedulerError(
                f"The job listing of {self.scheduler_name} cannot be read: the columns read do "
                f"not match the output of '{' '.join(self.config['list_active_jobs'])}', so "
                "the state of the jobs is not known")

        cmd = self.config["list_active_jobs"].copy()

        if "$USER" in cmd:
            username = os.getenv("USER") or os.getenv("USERNAME")
            if not username:
                raise JobSchedulerError("Cannot determine current username")
            cmd = [arg.replace("$USER", username) if arg == "$USER" else arg for arg in cmd]
        return cmd

    def _listed_job_id(self, field):
        """Job id in the first column of a job listing, or None for other lines."""
        if self.scheduler_name in _SERVER_SUFFIX and "." in field:
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
            For PBS and Torque the part before the first dot, otherwise the
            id itself.
        """
        job_id = str(job_id)
        if self.scheduler_name in _SERVER_SUFFIX:
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
            list of strings, the command fails, or no id is found in its output.
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

            result = self._run_command(cmd, input_data=script_content, cwd=cwd)
        else:
            cmd = self._build_command("submit", script_path, script_args)
            result = self._run_command(cmd, cwd=cwd)

        if self.dry_run:
            return None

        job_id = self._extract_job_id(result.stdout)
        if not job_id:
            raise JobSchedulerError(
                f"Could not extract job ID from output: {result.stdout.strip()}"
            )

        log_msg = f"Submitted job {job_id}"
        if script_args and logging.getLogger().isEnabledFor(logging.DEBUG):
            log_msg += f" with arguments: {' '.join(script_args)}"
        self.logger.info(log_msg)
        return job_id

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
            If the listing of the scheduler cannot be read
            (``UNREADABLE_LISTINGS``), the user name is unknown or the listing
            command fails.
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

        The state is read from a fixed column of the listing: the third for
        LSF, the fifth for Slurm and SGE. Lines whose first column is not a
        job id are skipped.

        Returns
        -------
        running, queued : list of str
            Ids in the form of the listing (see ``job_key``).

        Raises
        ------
        JobSchedulerError
            If the listing of the scheduler cannot be read
            (``UNREADABLE_LISTINGS``, and a custom table, whose states are not
            known), the user name is unknown or the listing command fails.
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

        The state is read from the sixth column for PBS, Torque and
        HTCondor, the third for LSF, the fifth for Slurm and SGE.
        """
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

            if self.scheduler_name in ["pbs", "torque"]:
                if len(fields) >= 6:
                    status = fields[5]
                    if status == "R":
                        running_jobs.append(job_id)
                    elif status == "Q":
                        queued_jobs.append(job_id)
            elif self.scheduler_name == "lsf":
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
            elif self.scheduler_name in ["htcondor", "condor"]:
                if len(fields) >= 6:
                    status = fields[5]  # ST column of condor_q
                    if status in ["R"]:  # running
                        running_jobs.append(job_id)
                    elif status in ["I", "H"]:  # idle, on hold
                        queued_jobs.append(job_id)

        return running_jobs, queued_jobs

    def _extract_active_job_ids(self, output):
        """Active job ids in the output of the listing command.

        PBS, Torque and LSF also list finished jobs, which are filtered by the
        state column; the listings of the other schedulers hold active jobs
        only.
        """
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
                if self.scheduler_name in ["pbs", "torque"]:
                    if len(fields) >= 6 and fields[5] in ["R", "Q", "H", "T", "W", "S"]:
                        job_ids.append(job_id)
                elif self.scheduler_name == "lsf":
                    if len(fields) >= 3 and fields[2] in ["RUN", "PEND", "PSUSP", "USUSP", "SSUSP"]:
                        job_ids.append(job_id)
                else:
                    job_ids.append(job_id)

        return job_ids

    def get_supported_schedulers(self):
        """Names of the schedulers in ``SCHEDULER_CONFIG``."""
        return list(SCHEDULER_CONFIG.keys())

    def __repr__(self):
        return f"JobScheduler(scheduler='{self.scheduler_name}')"
