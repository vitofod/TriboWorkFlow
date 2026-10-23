"""TOML settings loaded into dataclasses and validated."""

from .loader import load_settings, settings_from_dict, validate
from .schema import SCHEMA_VERSION, Settings, SettingsError

__all__ = ['SCHEMA_VERSION', 'Settings', 'SettingsError', 'load_settings',
           'settings_from_dict', 'validate']
