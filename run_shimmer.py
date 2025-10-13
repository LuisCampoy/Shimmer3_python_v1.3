#!/usr/bin/env python3
"""
Entry point for Shimmer3 IMU Data Streamer
"""
import sys
import asyncio
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

try:
    # Import and run main
    from main import main
    
    if __name__ == "__main__":
        asyncio.run(main())
        
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all dependencies are installed:")
    print("  pip install -r requirements.txt")
    print("  pip install -e .")
    sys.exit(1)
except KeyboardInterrupt:
    print("\nApplication interrupted by user")
    sys.exit(0)
except Exception as e:
    print(f"Error running application: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)