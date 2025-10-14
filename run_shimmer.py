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
    parser.add_argument('--passive-sniff-seconds', type=float, default=0.0,
                       help='If > 0 and passive_connect enabled, sniff raw bytes for this many seconds then report and exit (no streaming).')
    parser.add_argument('--preserve-connection', action='store_true',
                       help='Do not close serial/RFCOMM on normal exit (sets shimmer.preserve_connection=true).')
    parser.add_argument('--keepalive', action='store_true',
                       help='Enable periodic keepalive version requests (sets shimmer.keepalive_enabled=true).')
    parser.add_argument('--keepalive-interval', type=float, default=None,
                       help='Override keepalive interval seconds (sets shimmer.keepalive_interval_seconds).')
    
    args = parser.parse_args()
    
    try:
        # Initialize application with config path
        app = ShimmerStreamer(args.config)
        
        # Override config with command line arguments using consistent access
        if args.device and app.config is not None:
            if 'shimmer' not in app.config:
                app.config['shimmer'] = {}
            app.config['shimmer']['port'] = args.device
            print(f"Device port overridden to: {args.device}")
            
        if args.rate and app.config is not None:
            if 'shimmer' not in app.config:
                app.config['shimmer'] = {}
            app.config['shimmer']['sampling_rate'] = args.rate
            print(f"Sampling rate overridden to: {args.rate} Hz")
        if args.preserve_connection and app.config is not None:
            if 'shimmer' not in app.config:
                app.config['shimmer'] = {}
            app.config['shimmer']['preserve_connection'] = True
            print("Preserve connection enabled (will skip automatic disconnect).")
        if args.keepalive and app.config is not None:
            if 'shimmer' not in app.config:
                app.config['shimmer'] = {}
            app.config['shimmer']['keepalive_enabled'] = True
            print("Keepalive loop enabled.")
        if args.keepalive_interval and app.config is not None:
            if 'shimmer' not in app.config:
                app.config['shimmer'] = {}
            app.config['shimmer']['keepalive_interval_seconds'] = args.keepalive_interval
            print(f"Keepalive interval set to {args.keepalive_interval}s")
        
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

            # Passive sniff mode
            if args.passive_sniff_seconds > 0:
                shimmer_cfg = app.config.get('shimmer', {}) if isinstance(app.config, dict) else {}
                passive_enabled = shimmer_cfg.get('passive_connect', False)
                if not passive_enabled:
                    print("Passive sniff requested but passive_connect is False in config. Enable it under shimmer.passive_connect.")
                    sys.exit(1)
                # perform sniff using underlying client
                if not app.shimmer_client:
                    print("Shimmer client not initialized; cannot perform passive sniff.")
                    sys.exit(1)
                result = await app.shimmer_client.passive_sniff(seconds=args.passive_sniff_seconds)
                print(f"Passive sniff completed: bytes={result['bytes_captured']}, truncated={result['truncated']}, hex_preview={result['hex_preview']}")
                print("Exiting after passive sniff.")
                sys.exit(0)

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