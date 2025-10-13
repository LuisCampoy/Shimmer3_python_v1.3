"""Utility modules for Shimmer3 application"""

__version__ = "1.1.0"
__author__ = "Luis Campoy"
__description__ = "Shimmer3 IMU Data Streaming, Logging, and Export Tool"

# Only import what actually exists in the utils package
from .config import Config

__all__ = ['Config']