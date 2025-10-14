#!/usr/bin/env python3
"""
Entry point for Shimmer3 IMU Data Streamer
"""
import sys
import os
import asyncio
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from main import ShimmerStreamer

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Shimmer3 IMU Data Streaming Tool')
    parser.add_argument('-c', '--config', default='config/shimmer_config.json',
                       help='Path to configuration file')
    parser.add_argument('-d', '--device', help='Shimmer3 device port or address')
    parser.add_argument('-r', '--rate', type=float, help='Sampling rate in Hz')
    parser.add_argument('-t', '--duration', type=float, help='Recording duration in seconds')
    parser.add_argument('--export-only', action='store_true', 
                       help='Export mode only (no streaming)')
    
    args = parser.parse_args()
    
    try:
        # Initialize application with config path
        app = ShimmerStreamer(args.config)
        
        # Override config with command line arguments using consistent access
        if args.device and app.config is not None:
            app.config.set('shimmer.port', args.device)
            print(f"Device port overridden to: {args.device}")
            
        if args.rate and app.config is not None:
            app.config.set('shimmer.sampling_rate', args.rate)
            print(f"Sampling rate overridden to: {args.rate} Hz")
        
        # Initialize components
        app.initialize_components()

        if args.export_only:
            # Export existing data only
            print("Running in export-only mode...")
            await app.export_data()
            print("Export completed.")
        else:
            # Connect and start streaming
            print("Connecting to Shimmer3 device...")
            connected = await app.connect_shimmer()
            
            if not connected:
                print("Failed to connect to Shimmer3 device. Check configuration and device availability.")
                sys.exit(1)

            # Set duration if specified
            if args.duration:
                print(f"Recording for {args.duration} seconds...")
                async def stop_after_duration():
                    await asyncio.sleep(args.duration)
                    app.stop_streaming()
                asyncio.create_task(stop_after_duration())
            else:
                print("Starting continuous streaming (Ctrl+C to stop)...")
            
            await app.start_streaming()

        print('\nApplication completed successfully.')

    except KeyboardInterrupt:
        print('\nApplication interrupted by user')
    except Exception as e:
        print(f'\nApplication Error: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())