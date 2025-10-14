#!/usr/bin/env python3
"""
Entry point for Shimmer3 IMU Data Streamer
"""
import sys
import os
import asyncio
import logging
import traceback  # Add this import

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.main import main
except ImportError:
    from main import main

if __name__ == "__main__":
    asyncio.run(main())