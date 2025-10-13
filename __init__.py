"""
Shimmer3 IMU Data Streaming Application

A Python application for streaming, logging, and exporting data from Shimmer3 IMU devices.
"""

__version__ = "1.1.0" 
__author__ = "Luis Campoy"
__description__ = "Shimmer3 IMU Data Streaming, Logging, and Export Tool"

# Import main classes from src level modules  
from .src.shimmer_client import ShimmerClient
from .src.data_logger import DataLogger  
from .src.data_exporter import DataExporter
from .src.utils.config import Config

__all__ = ['ShimmerClient', 'DataLogger', 'DataExporter', 'Config']