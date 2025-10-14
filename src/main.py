# Shimmer3 IMU Data Streaming, Logging, and Export Tool
# Main entry point for the application
# Script created on 10/08/2025
# Last revision on 10/14/2025

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import os
import json
from utils.config_helpers import safe_config_get, validate_config_section, get_config_with_defaults

# Fixed imports - use absolute imports for entry point compatibility
try:
    # Try relative imports first (when imported as module)
    from .shimmer_client import ShimmerClient
    from .data_logger import DataLogger
    from .data_exporter import DataExporter
    from .utils.config import Config
    from .utils.config_helpers import safe_config_get, validate_config_section, get_config_with_defaults
except ImportError:
    # Fall back to absolute imports (when run as script)
    from shimmer_client import ShimmerClient
    from data_logger import DataLogger
    from data_exporter import DataExporter
    from utils.config import Config
    from utils.config_helpers import safe_config_get, validate_config_section, get_config_with_defaults

# Create Class for the application
class ShimmerStreamer:
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the Shimmer3 Streamer application"""
        self.logger = logging.getLogger(__name__)
        self.config = None
        self.shimmer_client = None
        self.data_logger = None
        self.data_exporter = None
        
        # Load configuration
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'shimmer_config.json')
        
        self.load_config(config_path)

    def load_config(self, config_path: str):
        """Load configuration from JSON file"""
        try:
            # Ensure config_path is a string
            if isinstance(config_path, Path):
                config_path = str(config_path)
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            self.logger.info(f"Configuration loaded from {config_path}")
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise
            
    def initialize_components(self):
        """Initialize all system components"""
        try:
            if self.config is None:
                raise RuntimeError("Configuration not loaded. Cannot initialize components.")

            # Initialize Shimmer client with its sub-config dict
            shimmer_config = self.config.get('shimmer', {})
            device_id = shimmer_config.get('device_id', None)
            self.shimmer_client = ShimmerClient(shimmer_config, device_id=device_id)
            
            # Initialize data logger
            data_config = self.config.get('data', {})
            raw_dir = data_config.get('raw_directory', 'data/raw')
            file_format = data_config.get('format', 'csv')
            max_file_size_mb = data_config.get('max_file_size_mb', data_config.get('max_file_size', 100))
            buffer_size = data_config.get('buffer_size', 1000)
            compression = data_config.get('compression', False)
            auto_flush_interval = data_config.get('auto_flush_interval', 5.0)

            self.data_logger = DataLogger(
                output_dir=raw_dir,
                file_format=file_format,
                buffer_size=buffer_size,
                max_file_size_mb=max_file_size_mb,
                config=self.config
            )
            
            # Initialize data exporter
            export_config = self.config.get('export', {})
            processed_dir = data_config.get('processed_directory', 'data/processed')
            if export_config.get('enabled', True):
                # DataExporter expects input/output directories (strings/PathLikes), not dicts
                self.data_exporter = DataExporter(
                    input_dir=raw_dir,
                    output_dir=processed_dir,
                )
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise

    async def connect_shimmer(self) -> bool:
        '''
        Connect to the Shimmer3 device.
        Args:
            self: Instance of the class.
        Returns:
            None
        '''
        try:
            if not self.shimmer_client:
                self.logger.error("Shimmer client not initialized")
                return False

            connected = await self.shimmer_client.connect()
            if not connected:
                self.logger.error("Could not connect to Shimmer3.")
                return False

            # Configure IMU sensors
            if not isinstance(self.config, dict):
                raise RuntimeError("Configuration not loaded or invalid format")
            shimmer_cfg = self.config.get('shimmer', {}) if isinstance(self.config, dict) else {}
            sensors = shimmer_cfg.get('sensors', ['accelerometer', 'gyroscope', 'magnetometer'])
            sampling_rate = shimmer_cfg.get('sampling_rate', 51.2)

            configured = await self.shimmer_client.configure_sensors(sensors, sampling_rate)
            if not configured:
                self.logger.error("Could not configure Shimmer3 sensors.")
                return False

            self.logger.info(f'Shimmer3 connected and configured with sensors: {sensors}')
            return True

        except Exception as e:
            self.logger.error(f'Failed to connect/configure Shimmer3: {e}')
            return False
    
    async def start_streaming(self) -> None:
        '''
        Start streaming data from Shimmer3 and logging it.
        Args:
            self: Instance of the class.
        Returns:
            None
        '''
        try:
            if not self.shimmer_client or not self.data_logger:
                raise RuntimeError("Components not properly initialized")
                
            self.running = True
            self.logger.info('Starting data streaming...')

            # start streaming
            await self.shimmer_client.start_streaming()

            # Main streaming loop
            while self.running:
                try:
                    # Get data from Shimmer
                    data_packet = await self.shimmer_client.get_data()

                    if data_packet:
                        # log raw data
                        await self.data_logger.log_data(data_packet)

                        # Optional: Real-time processing/display
                        if isinstance(self.config, dict) and self.config.get('display.real_time', False):
                            self._display_data(data_packet)

                    # Check for export trigger
                    try:
                        if self.data_logger.should_export():
                            await self.export_data()
                    except Exception as e:
                        self.logger.error(f"should_export failed: {e}")
                        await asyncio.sleep(0.5)  # brief backoff to reduce spam

                    # Small delay to prevent excessive CPU usage
                    await asyncio.sleep(0.01)

                except asyncio.CancelledError:
                    self.logger.info("Streaming cancelled")
                    break
                except Exception as e:
                    self.logger.error(f'Error during streaming: {e}')
                    continue_on_error = True
                    if isinstance(self.config, dict):
                        continue_on_error = self.config.get('error_handling.continue_on_error', True)
                    if not continue_on_error:
                        break
                    # Brief pause before retry
                    await asyncio.sleep(0.1)

        except Exception as e:
            self.logger.error(f'Streaming failed: {e}')
            raise
        finally:
            await self.cleanup()

    async def export_data(self) -> None:
        '''
        Export logged data to processed format.
        Args:
            self: Instance of the class.
        Returns:
            None
        '''
        try:
            if not self.data_exporter:
                raise RuntimeError("Data exporter not initialized")
                
            self.logger.info('Starting data export...')

            export_formats = []
            if isinstance(self.config, dict):
                export_formats = self.config.get('export.formats', ['csv', 'hdf5'])
            else:
                export_formats = ['csv', 'hdf5']

            for format_type in export_formats:
                await self.data_exporter.export(format_type)

            # Mark export as completed in data logger to reset export conditions
            # (No mark_export_completed method in DataLogger; nothing to call here)

            self.logger.info('Data export completed successfully')

        except Exception as e:
            self.logger.error(f'Data export failed: {e}')
    
    def _display_data(self, data_packet: Dict[str, Any]) -> None:
        '''
        Display data packet in real-time.
        Args:
            self: Instance of the class.
            data_packet: Data packet from Shimmer3.
        Returns:
            None
        '''
        # Simple console display, can be enhanced with GUI libraries
        timestamp = data_packet.get('timestamp', 'N/A')
        accel = data_packet.get('accelerometer', {})
        gyro = data_packet.get('gyroscope', {})
        mag = data_packet.get('magnetometer', {})
        
        print(f"\r[{timestamp}] "
              f"Accel: ({accel.get('x', 0):.2f}, {accel.get('y', 0):.2f}, {accel.get('z', 0):.2f}) "
              f"Gyro: ({gyro.get('x', 0):.2f}, {gyro.get('y', 0):.2f}, {gyro.get('z', 0):.2f}) "
              f"Mag: ({mag.get('x', 0):.2f}, {mag.get('y', 0):.2f}, {mag.get('z', 0):.2f})", 
              end='', flush=True)
        
    async def cleanup(self) -> None:
        '''
        Cleanup resources on shutdown.
        Args:
            self: Instance of the class.
        Returns:
            None
        '''
        self.logger.info('Cleaning up resources...')

        if self.shimmer_client:
            await self.shimmer_client.disconnect()

        if self.data_logger:
            await self.data_logger.close()

        self.logger.info('Cleanup completed.')

    def stop_streaming(self) -> None:
        '''
        Stop the streaming process gracefully.
        Args:
            self: Instance of the class.
        Returns:
            None
        '''
        self.logger.info('Stopping streaming...')
        self.running = False

    async def _export_loop(self):
        """Background export loop"""
        # Fix inconsistent config access
        export_interval = safe_config_get(self.config, 'export.interval_seconds', 60)
        
        while self.shimmer_client and self.shimmer_client.streaming:
            try:
                await asyncio.sleep(export_interval)
                await self.export_data()
            except Exception as e:
                # Fix inconsistent config access
                continue_on_error = safe_config_get(self.config, 'error_handling.continue_on_error', True)
                self.logger.error(f"Export loop error: {e}")
                if not continue_on_error:
                    break

    async def start(self):        
        # Start background export only when enabled
        if isinstance(self.config, dict) and self.config.get('export.enable_background_loop', False):
            asyncio.create_task(self._export_loop())  

def signal_handler(signum: int, frame, app: ShimmerStreamer) -> None:
    '''
    Handles shutdown signals
    Args:
        signum: Signal number.
        frame: Current stack frame.
        app: Application instance.
    Returns:
        None
    '''
    print(f'\nReceived signal {signum}, shutting down...')
    app.stop_streaming()

async def main() -> None:
    '''
    Main entry point for the application.
    '''
    parser = argparse.ArgumentParser(description='Shimmer3 IMU Data Streamer')
    parser.add_argument('-c', '--config', type=str,
                        help='Path to configuration file')
    parser.add_argument('-d', '--device', type=str,
                        help='Shimmer3 device ID or port')
    parser.add_argument('-r', '--rate', type=float, default=51.2,
                        help='Sampling rate in Hz')
    parser.add_argument('-t', '--duration', type=int,
                        help='Recording duration to stream data in seconds')
    parser.add_argument('--export-only', action='store_true',
                         help='Only export existing data, do not stream')
    
    args = parser.parse_args()

    try:
        # Initialize application
        app = ShimmerStreamer(args.config)

        # Setup signal handlers with proper lambda functions
        signal.signal(signal.SIGINT, lambda s, f: signal_handler(s, f, app))
        signal.signal(signal.SIGTERM, lambda s, f: signal_handler(s, f, app))

        # Override config with command line arguments
        if args.device:
            if app.config is not None:
                if 'shimmer' not in app.config:
                    app.config['shimmer'] = {}
                app.config['shimmer']['port'] = args.device
        if args.rate:
            if app.config is not None:
                if 'shimmer' not in app.config:
                    app.config['shimmer'] = {}
                app.config['shimmer']['sampling_rate'] = args.rate
        
        app.initialize_components()

        if args.export_only:
            # Export existing data only
            await app.export_data()
        else:
            # Connect and start streaming
            await app.connect_shimmer()

            # Set duration if specified
            if args.duration:
                async def stop_after_duration():
                    await asyncio.sleep(args.duration)
                    app.stop_streaming()
                    app.logger.info(f'Stopping after {args.duration} seconds')

                asyncio.create_task(stop_after_duration())
            
            await app.start_streaming()

        print('\nApplication completed successfully.')

    except KeyboardInterrupt:
        print('\nApplication interrupted by user')
    except Exception as e:
        print(f'\nApplication Error: {e}')
        sys.exit(1)

if __name__ == '__main__':
    asyncio.run(main())
