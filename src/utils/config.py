# Config File
# Last review: 10/12/2025

import json
from pathlib import Path
from typing import Any, Optional, Dict

class Config:
    """Configuration management class for Shimmer3 application"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration with optional config file"""
        # Enhanced defaults with more comprehensive settings
        self.defaults = {
            'shimmer': {
                'device_id': 'shimmer3_default',
                'port': '/dev/rfcomm0',
                'device_address': '00:06:66:B1:4D:A1',
                'baudrate': 115200,
                'timeout': 5,
                'sampling_rate': 51.2,
                'sensors': ['accelerometer', 'gyroscope', 'magnetometer'],
                'calibration_enabled': True,
                'auto_pair': True,
                'max_pair_attempts': 3
            },
            'data': {
                'raw_directory': 'data/raw',
                'processed_directory': 'data/processed',
                'format': 'csv',
                'buffer_size': 1000,
                'max_file_size_mb': 100,
                'auto_flush_interval': 30,
                'compression': False,
                'backup_enabled': True
            },
            'export': {
                'formats': ['csv', 'hdf5'],
                'include_statistics': True,
                'compression': True,
                'interval_seconds': 60,
                'min_records_threshold': 1000,
                'quarantine_corrupted_files': True,
                'enable_background_loop': False,
                'chunk_size': 50000
            },
            'logging': {
                'level': 'INFO',
                'log_directory': 'logs',
                'max_file_size_mb': 10,
                'backup_count': 5,
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'display': {
                'real_time': False,
                'update_interval': 0.1
            },
            'error_handling': {
                'continue_on_error': True,
                'max_retries': 3
            }
        }
        
        # Initialize config data with defaults
        self.config_data = self.defaults.copy()
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> None:
        """Load configuration from JSON file"""
        try:
            config_file = Path(config_path)
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    # FIX: Start with defaults, then merge loaded config
                    self.config_data = self.defaults.copy()
                    self._merge_configs(loaded_config, self.config_data)
                print(f"Configuration loaded from {config_path}")
            else:
                print(f"Config file {config_path} not found. Using defaults.")
                self.config_data = self.defaults.copy()
        except json.JSONDecodeError as e:
            print(f"Invalid JSON in {config_path}: {e}. Using defaults.")
            self.config_data = self.defaults.copy()
        except Exception as e:
            print(f"Error loading config {config_path}: {e}. Using defaults.")
            self.config_data = self.defaults.copy()

    def _merge_configs(self, source: Dict[str, Any], target: Dict[str, Any]) -> None:
        """Recursively merge source config into target config"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._merge_configs(value, target[key])
            else:
                target[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'logging.level')"""
        keys = key.split('.')
        
        # Try to get from loaded config first
        value = self._get_nested_value(self.config_data, keys)
        if value is not None:
            return value
        
        # Fall back to defaults
        value = self._get_nested_value(self.defaults, keys)
        if value is not None:
            return value
            
        return default
    
    def _get_nested_value(self, data: Dict[str, Any], keys: list) -> Any:
        """Helper method to get nested value from dictionary"""
        try:
            value = data
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return None
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation"""
        keys = key.split('.')
        config = self.config_data
        
        # Navigate to the parent of the final key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            elif not isinstance(config[k], dict):
                config[k] = {}  # Convert non-dict values to dict
            config = config[k]
        
        # Set the final key
        config[keys[-1]] = value
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values merged with defaults"""
        merged = self.defaults.copy()
        self._merge_configs(self.config_data, merged)
        return merged
    
    def validate(self) -> bool:
        """Validate configuration values"""
        try:
            # Check required fields - allow either device_id OR port
            device_id = self.get('shimmer.device_id')
            port = self.get('shimmer.port')
            
            if not device_id and not port:
                print("Warning: No Shimmer device ID or port specified")
                return False
            
            # Validate sampling rate
            sampling_rate = self.get('shimmer.sampling_rate', 0)
            if not 0 < sampling_rate <= 1000:
                print(f"Warning: Invalid sampling rate: {sampling_rate}")
                return False
            
            # Validate directories exist or can be created
            for dir_key in ['data.raw_directory', 'data.processed_directory', 'logging.log_directory']:
                dir_path = Path(self.get(dir_key, ''))
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    print(f"Warning: Cannot create directory {dir_path}: {e}")
                    return False
            
            return True
        except Exception as e:
            print(f"Configuration validation error: {e}")
            return False
    
    def save_config(self, config_path: str) -> None:
        """Save current configuration to JSON file"""
        try:
            config_file = Path(config_path)
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_file, 'w', encoding='utf-8') as f:  # Added encoding
                json.dump(self.config_data, f, indent=2, ensure_ascii=False)
            print(f"Configuration saved to {config_path}")
        except Exception as e:
            print(f"Error saving config to {config_path}: {e}")