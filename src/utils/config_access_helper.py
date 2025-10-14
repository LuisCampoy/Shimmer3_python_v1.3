"""Helper functions for consistent configuration access across the Shimmer3 application"""

from typing import Any, Dict, Optional, List
import logging

def safe_config_get(config: Any, key: str, default: Any = None) -> Any:
    """
    Safely get configuration value with fallback for different config types
    
    Args:
        config: Config object, dict, or None
        key: Configuration key (supports dot notation like 'shimmer.port')
        default: Default value if key not found
        
    Returns:
        Configuration value or default
        
    Example:
        >>> safe_config_get(config, 'shimmer.port', '/dev/rfcomm0')
        >>> safe_config_get(config, 'data.buffer_size', 1000)
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

def validate_config_section(config: Any, section: str, required_keys: List[str], 
                          logger: Optional[logging.Logger] = None) -> bool:
    """
    Validate that a config section has all required keys
    
    Args:
        config: Configuration object
        section: Section name (e.g., 'shimmer', 'data')
        required_keys: List of required keys in the section
        logger: Optional logger for warnings
        
    Returns:
        True if all required keys are present
        
    Example:
        >>> validate_config_section(config, 'shimmer', ['port', 'baud_rate'])
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

def get_config_with_defaults(config: Any, key_defaults: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get multiple config values with their defaults in one call
    
    Args:
        config: Configuration object
        key_defaults: Dictionary mapping config keys to default values
        
    Returns:
        Dictionary with resolved configuration values
        
    Example:
        >>> defaults = {
        ...     'shimmer.port': '/dev/rfcomm0',
        ...     'shimmer.baud_rate': 115200,
        ...     'data.buffer_size': 1000
        ... }
        >>> values = get_config_with_defaults(config, defaults)
    """
    result = {}
    for key, default in key_defaults.items():
        result[key] = safe_config_get(config, key, default)
    return result

def normalize_config_paths(config: Any, path_keys: List[str]) -> Dict[str, str]:
    """
    Normalize path configurations to ensure they're strings
    
    Args:
        config: Configuration object
        path_keys: List of config keys that should be file/directory paths
        
    Returns:
        Dictionary with normalized path strings
        
    Example:
        >>> paths = normalize_config_paths(config, [
        ...     'data.output_directory',
        ...     'export.output_directory',
        ...     'logging.log_directory'
        ... ])
    """
    import os
    from pathlib import Path
    
    result = {}
    for key in path_keys:
        value = safe_config_get(config, key)
        if value is not None:
            # Convert Path objects to strings
            if hasattr(value, '__fspath__'):
                result[key] = str(value)
            # Handle dict with path-like keys
            elif isinstance(value, dict):
                for path_key in ['path', 'dir', 'directory']:
                    if path_key in value:
                        result[key] = str(value[path_key])
                        break
                else:
                    result[key] = str(value)
            else:
                result[key] = str(value)
    
    return result