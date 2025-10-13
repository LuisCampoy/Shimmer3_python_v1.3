# Shimmer3 IMU Data Streaming, Logging, and Export Tool
# Main entry point for the application
# Script created on 10/08/2025
# Last revision on 10/12/2025

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Fixed imports - use absolute imports for entry point compatibility
try:
    # Try relative imports first (when imported as module)
    from .shimmer_client import ShimmerClient
    from .data_logger import DataLogger
    from .data_exporter import DataExporter
    from .utils.config import Config
except ImportError:
    # Fall back to absolute imports (when run as script)
    from shimmer_client import ShimmerClient
    from data_logger import DataLogger
    from data_exporter import DataExporter
    from utils.config import Config

# Create Class for the application
class Shimmer3StreamerApp:
    def __init__(self, config_path: Optional[str] = None):
        self.config = Config(config_path)
        self.shimmer_client: Optional[ShimmerClient] = None
        self.data_logger: Optional[DataLogger] = None
        self.data_exporter: Optional[DataExporter] = None
        self.running = False
 
        # Setting up logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        '''
        Set up application logging.
        Args:
            self: Instance of the class.
        Returns:
            None
        '''
        log_dir: Path = Path(self.config.get('logging.log_directory', 'logs'))
        log_dir.mkdir(parents=True, exist_ok=True)

        log_level: str = self.config.get('logging.level', 'INFO')
        timestamp: str = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file: Path = log_dir / f'shimmer3_streamer_{timestamp}.log'

        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

        self.logger: logging.Logger = logging.getLogger(__name__)
        self.logger.info('Shimmer3 Streamer Application Initialized')

    async def initialize_components(self) -> None:
        '''
        Initialize Shimmer client, data logger, and data exporter.
        Args:
            self: Instance of the class.
        Returns:
            None
        '''
        try:
            # Initialize Shimmer Client
            self.shimmer_client = ShimmerClient(
                device_id=self.config.get('shimmer.device_id'),
                port=self.config.get('shimmer.port', '/dev/ttyUSB0'),
                baudrate=self.config.get('shimmer.baudrate', 115200)
            )

            # Initialize data_logger
            self.data_logger = DataLogger(
                output_dir=self.config.get('data.raw_directory', 'data/raw'),
                file_format=self.config.get('data.format', 'csv')
            )

            # Initialize data exporter
            self.data_exporter = DataExporter(
                input_dir=self.config.get('data.raw_directory', 'data/raw'),
                output_dir=self.config.get('data.processed_directory', 'data/processed'),
            )

            self.logger.info('All components initialized successfully')
        
        except Exception as e:
            self.logger.error(f'Failed to initialize components: {e}')
            raise

    async def connect_shimmer(self) -> None:
        '''
        Connect to the Shimmer3 device.
        Args:
            self: Instance of the class.
        Returns:
            None
        '''
        try:
            if not self.shimmer_client:
                raise RuntimeError("Shimmer client not initialized")

            connected = await self.shimmer_client.connect()
            if not connected:
                self.logger.error("Could not connect to Shimmer3. Aborting initialization.")
                raise RuntimeError("Shimmer3 connection failed")

            # Configure IMU sensors
            sensors = self.config.get('shimmer.sensors', ['accelerometer', 'gyroscope', 'magnetometer'])
            sampling_rate = self.config.get('shimmer.sampling_rate', 51.2)

            configured = await self.shimmer_client.configure_sensors(sensors, sampling_rate)
            if not configured:
                self.logger.error("Could not configure Shimmer3 sensors. Aborting initialization.")
                raise RuntimeError("Shimmer3 sensor configuration failed")

            self.logger.info(f'Shimmer3 connected and configured with sensors: {sensors}')

        except Exception as e:
            self.logger.error(f'Failed to connect/configure Shimmer3: {e}')
            raise
    
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
                        if self.config.get('display.real_time', False):
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
                    if not self.config.get('error_handling.continue_on_error', True):
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

            export_formats = self.config.get('export.formats', ['csv', 'hdf5'])  # Fixed: was 'export.fromat'

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

        print(f"\rTime: {timestamp} | "
              f"Accel: X={accel.get('x', 0):.2f} Y={accel.get('y', 0):.2f} Z={accel.get('z', 0):.2f} | "
              f"Gyro: X={gyro.get('x', 0):.3f} Y={gyro.get('y', 0):.3f} Z={gyro.get('z', 0):.3f}",
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
        while True:
            try:
                if self.data_logger and self.data_logger.should_export():
                    self.logger.info("Starting data export...")
                    if self.data_exporter:
                        await self.data_exporter.export('csv')
                        await self.data_exporter.export('hdf5')
                    else:
                        self.logger.error("Data exporter not initialized")
                    if hasattr(self.data_logger, "mark_export_completed"):
                        self.data_logger.mark_export_completed()
                    self.logger.info("Data export completed successfully")
            except Exception as e:
                self.logger.error(f"Export failed: {e}")
            # Sleep a bit to avoid hammering the exporter
            await asyncio.sleep(self.config.get('export.interval', 30.0)) # export frequency from config file (changed from 2 to 30)

    async def start(self):        
        # Start background export only when enabled
        if self.config.get('export.enable_background_loop', False):
            asyncio.create_task(self._export_loop())  

def signal_handler(signum: int, frame, app: Shimmer3StreamerApp) -> None:
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
        app = Shimmer3StreamerApp(args.config)

        # Setup signal handlers with proper lambda functions
        signal.signal(signal.SIGINT, lambda s, f: signal_handler(s, f, app))
        signal.signal(signal.SIGTERM, lambda s, f: signal_handler(s, f, app))

        # Override config with command line arguments
        if args.device:
            app.config.set('shimmer.port', args.device)
        if args.rate:
            app.config.set('shimmer.sampling_rate', args.rate)
        
        await app.initialize_components()

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
