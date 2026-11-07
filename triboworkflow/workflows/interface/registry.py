# Author: Vito Foderà
# Created: 2026-10-30

"""Steps of the interface workflow available to the command line.

``STEPS`` maps the name of a step of the graph of the orchestrator
(``INTERFACE_STEPS``) to the class that does its work, a subclass of
``execution.orchestrator.Step`` built without arguments. Each step adds
itself here once it is implemented; until then the command line answers that
the step is not available, and nothing is prepared or submitted.

In this version ``match``, ``pes`` and ``pes_scf`` are available: a run lists
only these in ``[interface].steps``; with any other step the command line
stops with ``step '<name>' not available in this version`` before doing
anything.
"""

from triboworkflow.execution.orchestrator import Step

from .match import MatchStep
from .pes import RelaxedPesStep, StaticPesStep

STEPS: dict[str, type[Step]] = {'match': MatchStep, 'pes': RelaxedPesStep,
                                'pes_scf': StaticPesStep}


class StepNotAvailable(LookupError):
    """A step of the graph that this version does not implement yet."""


def get_step(name):
    """The step named ``name``.

    Parameters
    ----------
    name : str
        One of ``INTERFACE_STEPS``.

    Returns
    -------
    Step

    Raises
    ------
    StepNotAvailable
        If no class of ``STEPS`` does the step.
    """
    try:
        cls = STEPS[name]
    except KeyError:
        raise StepNotAvailable(f'step {name!r} not available in this version') from None
    return cls()
