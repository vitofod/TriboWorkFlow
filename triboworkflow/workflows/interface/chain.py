# Author: Vito Foderà
# Created: 2026-11-08

"""The steps that a step of the interface workflow reads from.

A step writes its entries with a provenance made of its parameters, of its
calculations and of the content of the entries that it read. That content says
which values the step started from, not whether they are still current: the
entries of a surface that the settings or a calculation changed after its
analysis keep the content that the step downstream read, and the step
downstream would stay done on top of them. A step is therefore done only if the
steps it reads from are done too, up to ``match``: ``upstream_done`` is the
condition that the ``done`` of a step adds to the comparison of its own
provenance, and ``check_upstream`` gives the same condition as an error with the
steps to run again, for the ``expected`` of a step that reads entries written by
a step that has others upstream (``ppes`` reads the adhesion, which reads the
surface, which reads ``match``).

The steps upstream of a step are those that the graph of the orchestrator gives
it (``interface_graph``, with the surface that ``[interface.adhesion]`` names),
so that the data a step reads and the order in which the orchestrator runs it
stay one thing.
"""

from triboworkflow.core.store import PrerequisiteError
from triboworkflow.execution.orchestrator import interface_graph


def _load_run(run_dir):
    # Imported here: the command line imports the registry of the steps.
    from triboworkflow.cli.run import load_run

    return load_run(run_dir)


def upstream_steps(step, run_dir):
    """The steps whose entries ``step`` reads.

    Parameters
    ----------
    step : Step
        A step of the interface workflow, with its ``name``.
    run_dir : path-like
        Directory of the run, whose settings name the surface of the adhesion.

    Returns
    -------
    list of Step
        The direct dependencies of ``step`` in ``interface_graph``, as the
        registry builds them.
    """
    # Imported here: the registry imports the steps, which import this module.
    from .registry import get_step

    graph = interface_graph(_load_run(run_dir).interface.adhesion.pes)
    return [get_step(name) for name in graph[step.name]]


def upstream_done(step, run_dir, system):
    """Whether every step that ``step`` reads from is done.

    Parameters
    ----------
    step : Step
    run_dir : path-like
    system : str

    Returns
    -------
    bool
        True if ``done`` holds for each step in ``upstream_steps``, which in
        turn asks the steps upstream of it.
    """
    return all(other.done(run_dir, system) for other in upstream_steps(step, run_dir))


def not_done_upstream(step, run_dir, system):
    """Names of the steps upstream of ``step`` that are not done, the farthest first.

    Parameters
    ----------
    step : Step
    run_dir : path-like
    system : str

    Returns
    -------
    list of str
        Every step reached from ``step`` through the graph whose ``done`` is
        false, once each, ordered from ``match`` downwards.
    """
    seen, ordered = set(), []

    def visit(current):
        for other in upstream_steps(current, run_dir):
            if other.name not in seen:
                seen.add(other.name)
                visit(other)
                ordered.append(other)

    visit(step)
    return [other.name for other in ordered if not other.done(run_dir, system)]


def check_upstream(step, run_dir, system):
    """Stop a step whose steps upstream are not done.

    Parameters
    ----------
    step : Step
        The step whose upstream steps are checked (not ``step`` itself).
    run_dir : path-like
    system : str

    Raises
    ------
    PrerequisiteError
        Naming the steps upstream of ``step`` that are not done, or whose
        entries are not current. Nothing is recomputed: they have to run first.
    """
    names = not_done_upstream(step, run_dir, system)
    if names:
        raise PrerequisiteError(f'{system}: steps upstream of {step.name} not done or not '
                                f'current: {", ".join(names)}; run them first')
