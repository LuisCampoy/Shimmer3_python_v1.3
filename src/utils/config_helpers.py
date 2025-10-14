"""Helper functions for consistent configuration access"""

from typing import Any, Dict, Optional, Union
import logging
from pathlib import Path

def safe_config_get(config: Any, key: str, default: Any = None) -> Any:
    """
    Safely get configuration value with fallback for different config types
    
    Args:
        config: Config object, dict, or None
        key: Configuration key (supports dot notation)
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    if config is None:
        return default
        
    # If it's a Config object with .get() method
    if hasattr(config, 'get') and callable(getattr(config, 'get')):
        return config.get(key, default)
    
    # If it's a plain dictionary
    if isinstance(config, dict):
        # Handle dot notation for nested keys
        keys = key.split('.')
        value = config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    # Fallback
    return default

def validate_config_section(config: Any, section: str, required_keys: list, 
                          logger: Optional[logging.Logger] = None) -> bool:
    """
    Validate that a config section has all required keys
    
    Args:
        config: Configuration object
        section: Section name (e.g., 'shimmer', 'data.export')
        required_keys: List of required keys in the section
        logger: Optional logger for warnings
        
    Returns:
        True if all required keys are present
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    missing_keys = []
    for key in required_keys:
        full_key = f"{section}.{key}" if section else key
        if safe_config_get(config, full_key) is None:
            missing_keys.append(full_key)
    
    if missing_keys:
        logger.warning(f"Missing configuration keys: {missing_keys}")
        return False
    
    return True

def get_config_with_defaults(config: Any, defaults: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get configuration values with fallback defaults
    
    Args:
        config: Configuration object
        defaults: Dictionary of default values
        
    Returns:
        Dictionary with configuration values or defaults
    """
    result = {}
    for key, default_value in defaults.items():
        result[key] = safe_config_get(config, key, default_value)
    return result