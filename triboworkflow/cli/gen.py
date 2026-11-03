# Author: Vito Foderà
# Created: 2026-11-02

"""Command ``gen``: the pw.x input of one calculation, generated and not run.

Notes
-----
As ``gen`` of Xsorb (``xsorb/cli/gen.py``, CC BY 4.0): the inputs are
generated from the settings without launching any calculation. Here one
input of ``[structure]``, with the extxyz of its structure, written next to
the settings file; no run, lock or database is involved.
"""

import logging
from collections import Counter
from pathlib import Path

from .command import CLICommandBase

logger = logging.getLogger(__name__)


def output_paths(settings_path, generation):
    """Paths of the input and of the extxyz written by ``gen``.

    Parameters
    ----------
    settings_path : path-like
        The settings file; relative names are taken from its folder.
    generation : triboworkflow.core.settings.schema.GenerationParams

    Returns
    -------
    input_path, extxyz_path : pathlib.Path
        ``file_prefix``, and a file with the same name and the suffix
        ``.extxyz``.
    """
    path = Path(generation.file_prefix)
    if not path.is_absolute():
        path = Path(settings_path).parent / path
    return path, path.with_name(f'{path.stem}.extxyz')


class CLICommand(CLICommandBase):
    """generate the pw.x input of one calculation and the extxyz of its structure

Reads a settings file with [structure] (see examples/single_input.toml): the
structure file and its constraints, the namelists as written, the species
and [k_points]. Writes, in the folder of the settings file, the input named
by [generation].file_prefix and an extxyz with the same name (e.g.
input.pwi and input.extxyz): cell, positions, labels and constraints of the
input, to look at in OVITO. With overwrite = false an existing file is an
error and nothing is written. The changes made to the settings (an scf makes
one step and has no &IONS) are printed as warnings. Nothing is run; a run
directory is not needed.

Example:
 $ triboworkflow gen single_input.toml
    """

    loads_run = False

    def add_arguments(self, parser):
        parser.add_argument('settings', help='settings file (TOML) with [structure]')

    def run(self, args, run_dir, settings):
        from triboworkflow.core.settings import load_settings
        from triboworkflow.core.store import write_text
        from triboworkflow.engines.qe.prepare import single_input

        from .run import RunError

        settings_path = Path(args.settings)
        settings = load_settings(settings_path)
        if settings.structure is None:
            raise RunError(f'{settings_path}: [interface] describes the workflow, whose '
                           'calculations are prepared in a run by "triboworkflow prepare <step>"; '
                           'gen needs [structure]')
        input_path, extxyz_path = output_paths(settings_path, settings.generation)
        if input_path == extxyz_path:
            raise RunError(f'[generation].file_prefix = {settings.generation.file_prefix!r}: the '
                           'input would have the name of its extxyz')
        found = [str(p) for p in (input_path, extxyz_path) if p.exists()]
        if found and not settings.generation.overwrite:
            raise RunError(f'{", ".join(found)} already exist and [generation].overwrite is '
                           'false: nothing written')
        generated = single_input(settings)
        write_text(input_path, generated.pw.text)
        write_text(extxyz_path, generated.structure_text)
        for change in generated.pw.changes:
            logger.warning('Warning: %s', change)
        structure = generated.structure
        sources = Counter(structure.constraints.source)
        kpoints = [lines for name, lines in generated.pw.cards if name == 'K_POINTS'][0]
        print(f'Input: {input_path.resolve()}')
        print(f'Structure: {extxyz_path.resolve()}')
        print(f'calculation = {generated.pw.calculation}; {len(structure.structure)} atoms, '
              f'{len(settings.atomic_species)} species; {" ".join(kpoints)}')
        print(f'Structure read from {structure.file} ({structure.format}); constraints: '
              + ', '.join(f'{n} {source}' for source, n in sorted(sources.items())))
