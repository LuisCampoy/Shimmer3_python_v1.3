#!/usr/bin/env python3

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils.config import Config
from utils.config_helpers import safe_config_get, validate_config_section

def test_config():
    """Test configuration loading and access"""
    try:
        # Test default config
        config = Config()
        print("✓ Default config created successfully")
        
        # Test config access
        port = safe_config_get(config, 'shimmer.port', '/dev/ttyUSB0')
        print(f"✓ Port: {port}")
        
        rate = safe_config_get(config, 'shimmer.sampling_rate', 51.2)
        print(f"✓ Sampling rate: {rate}")
        
        # Test validation
        is_valid = validate_config_section(config, 'shimmer', ['port'])
        print(f"✓ Configuration validation: {is_valid}")
        
        # Test loading from file
        config_with_file = Config('config/shimmer_config.json')
        print("✓ Config loaded from file successfully")
        
        print("\nAll configuration tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_config()
    sys.exit(0 if success else 1)