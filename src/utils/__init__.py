"""Utility modules for Shimmer3 application"""

__version__ = "1.1.0"
__author__ = "Luis Campoy"
__description__ = "Shimmer3 IMU Data Streaming, Logging, and Export Tool"

# Only import what actually exists in the utils package
from .config import Config
from .config_helpers import safe_config_get
from .path_utils import normalize_path, ensure_dir

__all__ = ['Config', 'safe_config_get', 'normalize_path', 'ensure_dir']
