# Author: Vito Foderà
# Created: 2026-11-14

"""Registry of the types of MLIP model and of the ASE calculator that loads each one.

A calculation with a machine-learned interatomic potential (MLIP) is described by three
settings of ``[mlip]``: ``type``, the family of the model; ``path``, the file of the model on
the machine that runs the calculation; ``kwargs``, the keyword arguments of the function that
loads it. The registry maps each type to its loader and turns the three settings into the
description of the call (``ase_calculator_call``).

The registry holds data and imports no MLIP library (``torch``, ``mace``, ``deepmd``): the
calculator is created by the executor, which runs in the environment of the MLIP and imports
the factory from its name. The call it returns contains only JSON types, so it can be written
in a job file and in the record of a calculation.

Three types are registered. Others are added with ``register_model_type``.

======== ================ ======================================== =================
Type     Package          ASE factory                              Backends
======== ================ ======================================== =================
DP       deepmd-kit       ``deepmd.calculator:DP``                 ase, lammps
MACE     mace-torch       ``mace.calculators:MACECalculator``      ase, lammps
MACE_MP  mace-torch       ``mace.calculators:mace_mp``             ase
======== ================ ======================================== =================

Nothing here looks at the model file: it lives on the cluster and the settings are also read
where it does not exist. The caller chooses and carries the model; the loader fails if the
file is wrong. ``kwargs`` are passed as written and never completed with defaults, so that
what the calculation used is what the settings say.

Notes
-----
The registry, the factories and the way the path reaches each of them follow SCS
(``exploration/workflow/md/ase.py:36-76`` and ``:322-335``, CC BY 4.0): ``DP(model_path,
**kwargs)``, ``MACECalculator(model_path=model_path, device='cuda', **kwargs)`` and
``mace_mp(model=model_path, device='cuda', **kwargs)``.

To remove: differences from SCS. ``head`` of MACE comes from ``kwargs`` only, where SCS fixed
it at ``'Default'`` (A103, DEV-51). ``default_dtype`` is required for MACE with ASE, where SCS
left the default of the library (DEV-57). The type is compared whole and as written, where SCS
labelling strips the characters of ``'ML/'`` from it (A15); the path is required, where SCS
fell back on models in ``../../Training/NN0/``. GRACE and Eqv2 of SCS are not registered.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Backends an MLIP calculation can run with: the calculators of ASE, and LAMMPS.
BACKENDS = ('ase', 'lammps')


class ModelError(ValueError):
    """Invalid description of a model.

    Parameters
    ----------
    key : str
        Key of ``[mlip]`` at fault: ``'type'``, ``'path'`` or ``'kwargs'``.
    message : str
        What is wrong.
    """

    def __init__(self, key, message):
        self.key, self.message = key, message
        super().__init__(f'{key}: {message}')


@dataclass(frozen=True)
class ModelType:
    """A type of model: how ASE loads it and what it needs.

    Attributes
    ----------
    name : str
        Value of ``[mlip] type``.
    package : str
        Package that contains the loader, as pip names it: the one whose installed version
        ``[mlip] version`` is compared with.
    ase_factory : str
        Importable name of the function that returns the ASE calculator, ``'module:function'``.
    path_argument : str or None
        Keyword with which the path of the model reaches the factory; None if it is the first
        positional argument.
    fixed_kwargs : dict
        Keyword arguments the workflow sets, which ``kwargs`` cannot repeat. ``device`` is
        ``'cuda'`` for the types that take it: MLIP calculations run on GPU only.
    ase_required_kwargs : tuple of str
        Keys that ``kwargs`` must contain with the ASE backend, because they change the
        energies and are recorded with the model.
    backends : tuple of str
        Backends the type is declared for (``BACKENDS``).
    """

    name: str
    package: str
    ase_factory: str
    path_argument: str | None = None
    fixed_kwargs: dict = field(default_factory=dict)
    ase_required_kwargs: tuple = ()
    backends: tuple = BACKENDS

    @property
    def reserved_kwargs(self):
        """tuple of str: keys that the workflow sets and ``kwargs`` cannot contain."""
        reserved = tuple(self.fixed_kwargs)
        return reserved + (self.path_argument,) if self.path_argument else reserved


MODEL_TYPES: dict[str, ModelType] = {}


def register_model_type(model_type):
    """Add a type of model to the registry.

    Parameters
    ----------
    model_type : ModelType

    Raises
    ------
    ValueError
        If a type with the same name is registered.
    """
    if model_type.name in MODEL_TYPES:
        raise ValueError(f'the model type {model_type.name!r} is already registered')
    MODEL_TYPES[model_type.name] = model_type


register_model_type(ModelType(
    'DP', 'deepmd-kit', 'deepmd.calculator:DP'))
register_model_type(ModelType(
    'MACE', 'mace-torch', 'mace.calculators:MACECalculator', path_argument='model_path',
    fixed_kwargs={'device': 'cuda'}, ase_required_kwargs=('default_dtype',)))
register_model_type(ModelType(
    'MACE_MP', 'mace-torch', 'mace.calculators:mace_mp', path_argument='model',
    fixed_kwargs={'device': 'cuda'}, ase_required_kwargs=('default_dtype',), backends=('ase',)))


def get_model_type(type_name):
    """Type of model registered with a name.

    Parameters
    ----------
    type_name : str
        Name as written in ``[mlip] type``: the whole name, with its case.

    Returns
    -------
    ModelType

    Raises
    ------
    ModelError
        If no type has that name.
    """
    if type_name not in MODEL_TYPES:
        raise ModelError('type', f'{type_name!r} is not a model type; allowed values: '
                                 f'{", ".join(map(repr, MODEL_TYPES))}')
    return MODEL_TYPES[type_name]


def check_model(type_name, path, kwargs, backend=None):
    """Check the description of a model.

    The file of the model is not looked at.

    Parameters
    ----------
    type_name : str
        Name of a registered type.
    path : str
        Absolute path of the model on the machine that runs the calculation.
    kwargs : dict
        Keyword arguments of the loader, as written in the settings.
    backend : {'ase', 'lammps'} or None, optional
        Backend of the calculation. Without it only what does not depend on the backend is
        checked, as when the settings are read, before a step has chosen its engine.

    Returns
    -------
    ModelType

    Raises
    ------
    ModelError
        If the type is unknown; the path is not absolute, in the style of POSIX (the worker
        runs in the directory of the calculation, so a relative path would not lead to the
        model); ``kwargs`` contains a key that the workflow sets; the type is not declared for
        the backend; or ``kwargs`` lacks a key that the backend requires.
    """
    model_type = get_model_type(type_name)
    if not isinstance(path, str) or not path.startswith('/'):
        raise ModelError('path', 'give the absolute path of the model on the machine that runs '
                                 f'the calculations, starting with "/"; got {path!r}')
    for key in model_type.reserved_kwargs:
        if key in kwargs:
            what = ('the path of the model, which is set by "path"' if key == model_type.path_argument
                    else f'set by the workflow to {model_type.fixed_kwargs[key]!r}')
            raise ModelError('kwargs', f'{key!r} is {what}: remove it from kwargs')
    if backend is None:
        return model_type
    if backend not in model_type.backends:
        raise ModelError('type', f'{type_name} is not available with the backend {backend!r}; '
                                 f'available: {", ".join(model_type.backends)}')
    if backend == 'ase':
        missing = [key for key in model_type.ase_required_kwargs if key not in kwargs]
        if missing:
            raise ModelError('kwargs', f'{type_name} with ASE requires {", ".join(missing)}: it '
                                       'changes the energies, so it is never filled in and must '
                                       'be written in the settings, e.g. default_dtype = "float64"')
    return model_type


def ase_calculator_call(type_name, path, kwargs):
    """Description of the call that creates the ASE calculator of a model.

    Parameters
    ----------
    type_name : str
        Name of a registered type.
    path : str
        Absolute path of the model on the machine that runs the calculation.
    kwargs : dict
        Keyword arguments of the loader, as written in the settings. They are passed after
        those the workflow sets (``device`` and the path) and without any change or default.

    Returns
    -------
    dict
        ``{'factory': 'module:function', 'args': [...], 'kwargs': {...}}``, JSON types only:
        the executor imports ``module``, takes ``function`` and calls it with ``args`` and
        ``kwargs``.

    Raises
    ------
    ModelError
        As ``check_model`` with the ASE backend.
    """
    model_type = check_model(type_name, path, kwargs, backend='ase')
    call_kwargs = {model_type.path_argument: path} if model_type.path_argument else {}
    call_kwargs.update(model_type.fixed_kwargs)
    call_kwargs.update(kwargs)
    return {'factory': model_type.ase_factory,
            'args': [] if model_type.path_argument else [path],
            'kwargs': call_kwargs}
