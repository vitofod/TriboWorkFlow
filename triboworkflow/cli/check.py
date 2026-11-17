# Author: Vito Foderà
# Created: 2026-10-30

"""Command ``check``: read and validate the settings of a run.

Notes
-----
As ``settings`` of Xsorb (``xsorb/cli/settings.py``, CC BY 4.0): a summary of
the settings read.
"""

from .command import CLICommandBase


class CLICommand(CLICommandBase):
    """read and validate config.toml, and print a summary of it

The settings are read as every command reads them: an error names the TOML
path of the field. The advice on the values changed by the relaunches is
printed as a warning. The walltime and the jobs per group of each step are
printed; the steps without a walltime in [execution.<step>] (they get
24:00:00) and, with Slurm, the jobscripts with a --time, which the walltimes
replace, are printed as warnings. The placeholders left in the settings and
in the jobscripts (e.g. <venv>) are listed. Nothing is written.

Example:
 $ triboworkflow check
 $ triboworkflow check -r ~/runs/CuAl
    """

    loads_run = False

    def add_arguments(self, parser):
        pass

    def run(self, args, run_dir, settings):
        from triboworkflow.core.settings import SCHEMA_VERSION, load_settings

        from .run import config_file, print_placeholders, print_step_jobs, system_name

        path = config_file(run_dir)
        settings = load_settings(path)
        print(f'Settings: {path} (schema version {SCHEMA_VERSION})')
        if settings.interface is not None:
            print(f'Interface workflow, system {system_name(settings)}')
            print(f'Steps: {", ".join(settings.interface.steps)}')
        else:
            print(f'One calculation: {settings.structure.file}')
        execution = settings.execution
        if execution is None:
            print('Execution: no [execution]')
        else:
            groups = 'not set (give them in [execution.<step>], --groups, or answer in a ' \
                'terminal)' if execution.groups is None else execution.groups
            print(f'Execution: scheduler {execution.scheduler}, groups: {groups}')
            if settings.interface is not None:
                print_step_jobs(settings)
        print_placeholders(settings)
