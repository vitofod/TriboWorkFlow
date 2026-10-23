# Author: Vito Foderà
# Created: 2026-10-22

"""TOML settings loaded into dataclasses and validated.

See ``schema.py`` for the tables and ``loader.py`` for the reading.
"""

from .loader import load_settings, settings_from_dict, validate
from .schema import SCHEMA_VERSION, Settings, SettingsError

__all__ = ['SCHEMA_VERSION', 'Settings', 'SettingsError', 'load_settings',
           'settings_from_dict', 'validate']
