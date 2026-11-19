# Author: Vito Foderà
# Created: 2026-11-18

"""Tests of the species written ``"auto"`` in ``[[atomic_species]]``.

Their mass and pseudopotential are read from the ``ATOMIC_SPECIES`` of the
structure files of the settings when the settings are loaded; the order and the
labels of the species, and ``starting_magnetization``, stay as written.
"""

import logging
from pathlib import Path

import pytest
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

from triboworkflow.core.settings import SettingsError, load_settings
from triboworkflow.core.settings.schema import AUTO, Species
from triboworkflow.core.structures import StructureError, resolve_species, write_structure

DATA = Path(__file__).resolve().parent / 'data' / 'qe'
EXAMPLES = Path(__file__).resolve().parent.parent / 'examples'

PWI = """&CONTROL
  calculation = 'scf'
/
&SYSTEM
  ibrav = 0, nat = 1, ntyp = {ntyp}
/
&ELECTRONS
/
CELL_PARAMETERS angstrom
  3.0 0.0 0.0
  0.0 3.0 0.0
  0.0 0.0 20.0
ATOMIC_SPECIES
{species}
ATOMIC_POSITIONS angstrom
{first} 0.0 0.0 2.0
K_POINTS gamma
"""


def write_pwi(path, species):
    """A pw.x input with the ATOMIC_SPECIES ``[(label, mass, UPF file), ...]``."""
    lines = '\n'.join(f'{label} {mass} {upf}' for label, mass, upf in species)
    path.write_text(PWI.format(ntyp=len(species), species=lines, first=species[0][0]))
    return path


@pytest.fixture
def settings(tmp_path):
    """The settings of the interface example, with the slab files in ``tmp_path``."""
    loaded = load_settings(EXAMPLES / 'interface.toml')
    loaded.source = tmp_path / 'config.toml'
    loaded.interface.slabs[0].file, loaded.interface.slabs[1].file = 'a.pwi', 'b.pwi'
    return loaded


def values(settings):
    return [(s.label, s.mass, s.pseudopotential, s.starting_magnetization)
            for s in settings.atomic_species]


def test_species_read_from_the_slab_files_keep_their_order_and_the_magnetization(tmp_path, settings):
    write_pwi(tmp_path / 'a.pwi', [('Fe', 55.845, 'Fe.upf'), ('Cu', 63.546, 'Cu.upf')])
    write_pwi(tmp_path / 'b.pwi', [('Cu', 63.546, 'Cu.upf')])
    settings.atomic_species = [Species('Cu', AUTO, AUTO), Species('Fe', AUTO, 'Fe_given.upf', 0.4)]
    sentences = resolve_species(settings)
    # The order is that of the settings, not that of the files; what was written stays.
    assert values(settings) == [('Cu', 63.546, 'Cu.upf', None), ('Fe', 55.845, 'Fe_given.upf', 0.4)]
    assert len(sentences) == 3
    assert sentences[0] == f'atomic_species[0].mass = 63.546 for Cu, read from {tmp_path / "a.pwi"}'
    assert not settings.missing_values()


def test_nothing_is_read_when_no_species_asks(settings):
    before = values(settings)
    assert resolve_species(settings) == []     # a.pwi and b.pwi do not exist
    assert values(settings) == before


def test_species_in_no_file(tmp_path, settings):
    write_pwi(tmp_path / 'a.pwi', [('Fe', 55.845, 'Fe.upf')])
    write_pwi(tmp_path / 'b.pwi', [('Fe', 55.845, 'Fe.upf')])
    settings.atomic_species = [Species('Zn', AUTO, AUTO)]
    with pytest.raises(StructureError, match=r"atomic_species\[0\]: no species 'Zn' in "):
        resolve_species(settings)


def test_files_that_disagree_on_a_species(tmp_path, settings):
    write_pwi(tmp_path / 'a.pwi', [('Cu', 63.546, 'Cu.upf')])
    write_pwi(tmp_path / 'b.pwi', [('Cu', 63.546, 'Cu_other.upf')])
    settings.atomic_species = [Species('Cu', AUTO, AUTO)]
    with pytest.raises(StructureError) as error:
        resolve_species(settings)
    message = str(error.value)
    assert 'atomic_species[0].pseudopotential: the files give different values' in message
    assert f'Cu.upf in {tmp_path / "a.pwi"}' in message
    assert f'Cu_other.upf in {tmp_path / "b.pwi"}' in message
    # The mass is the same in both files: asking only for it works.
    settings.atomic_species = [Species('Cu', AUTO, 'Cu.upf')]
    resolve_species(settings)
    assert values(settings) == [('Cu', 63.546, 'Cu.upf', None)]


def test_file_that_is_not_an_input_or_an_output_of_pw_x(tmp_path, settings):
    write_pwi(tmp_path / 'a.pwi', [('Cu', 63.546, 'Cu.upf')])
    write_structure(Structure(Lattice.cubic(3.0), ['Cu'], [[0, 0, 0]]), tmp_path / 'b.extxyz')
    settings.interface.slabs[1].file = 'b.extxyz'
    settings.atomic_species = [Species('Cu', AUTO, AUTO)]
    with pytest.raises(StructureError, match=r'b.extxyz: "auto" in atomic_species needs .* pw.x'):
        resolve_species(settings)


def test_unresolved_species_are_placeholders(settings):
    settings.atomic_species = [Species('Cu', AUTO, 'Cu.upf'), Species('Fe', 55.845, AUTO)]
    assert settings.missing_values() == ['atomic_species[0].mass', 'atomic_species[1].pseudopotential']
    with pytest.raises(SettingsError, match='fill in before generating an input'):
        settings.require_complete()


def test_loaded_settings_have_the_species_of_the_structure_file(tmp_path, caplog):
    for name in ('example_relax.toml', 'example_relax.pwi'):
        (tmp_path / name).write_text((DATA / name).read_text())
    text = (tmp_path / 'example_relax.toml').read_text()
    text = text.replace('mass = 12.011', 'mass = "auto"').replace(
        'pseudopotential = "C.pbe-n-kjpaw_psl.1.0.0.UPF"', 'pseudopotential = "auto"')
    assert text.count('"auto"') == 2
    (tmp_path / 'example_relax.toml').write_text(text)
    with caplog.at_level(logging.INFO, logger='triboworkflow'):
        loaded = load_settings(tmp_path / 'example_relax.toml')
    reference = load_settings(DATA / 'example_relax.toml')
    assert loaded.atomic_species == reference.atomic_species
    assert loaded.species_read and loaded.species_read[0].startswith('atomic_species[0].mass = ')
    assert 'atomic_species[0].mass = 12.011 for C' in caplog.text


def test_a_file_is_read_once_while_it_does_not_change(tmp_path, settings, monkeypatch):
    """The orchestrator loads the settings again for every step at every pass."""
    from triboworkflow.core.structures import inputs

    write_pwi(tmp_path / 'a.pwi', [('Cu', 63.546, 'Cu.upf')])
    write_pwi(tmp_path / 'b.pwi', [('Cu', 63.546, 'Cu.upf')])
    reads, real = [], inputs.read_structure

    def counted(path, *args):
        reads.append(path)
        return real(path, *args)

    monkeypatch.setattr(inputs, 'read_structure', counted)
    for _ in range(3):
        settings.atomic_species = [Species('Cu', AUTO, AUTO)]
        resolve_species(settings)
    assert reads == [tmp_path / 'a.pwi', tmp_path / 'b.pwi']
    write_pwi(tmp_path / 'b.pwi', [('Cu', 63.546, 'Cu.upf'), ('Fe', 55.845, 'Fe.upf')])
    settings.atomic_species = [Species('Fe', AUTO, AUTO)]
    resolve_species(settings)
    assert reads[2:] == [tmp_path / 'b.pwi']  # changed: read again
    assert values(settings) == [('Fe', 55.845, 'Fe.upf', None)]


def test_the_species_read_are_logged_once(tmp_path, caplog):
    for name in ('example_relax.toml', 'example_relax.pwi'):
        (tmp_path / name).write_text((DATA / name).read_text())
    path = tmp_path / 'example_relax.toml'
    path.write_text(path.read_text().replace('mass = 12.011', 'mass = "auto"'))
    with caplog.at_level(logging.INFO, logger='triboworkflow'):
        for _ in range(3):
            assert load_settings(path).species_read
    assert caplog.text.count('atomic_species[0].mass = 12.011 for C') == 1


def test_loaded_settings_with_a_species_in_no_file(tmp_path):
    for name in ('example_relax.toml', 'example_relax.pwi'):
        (tmp_path / name).write_text((DATA / name).read_text())
    text = (tmp_path / 'example_relax.toml').read_text()
    text = text.replace('label = "C"', 'label = "Xx"').replace('mass = 12.011', 'mass = "auto"')
    (tmp_path / 'example_relax.toml').write_text(text)
    with pytest.raises(SettingsError, match=r"example_relax.toml: atomic_species: .* no species 'Xx'"):
        load_settings(tmp_path / 'example_relax.toml')
