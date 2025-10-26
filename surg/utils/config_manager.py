"""
Utilities - Configuration Manager

This module handles configuration management, environment variables,
and settings for the SURG system.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

from surg.utils.logger import get_logger

logger = get_logger(__name__)

def load_api_keys(keys_file: str = "keys.json") -> Dict[str, str]:
    possible_paths = [
        Path(keys_file),
        Path.cwd() / keys_file,
        Path(__file__).parent.parent.parent / keys_file,
        Path.home() / ".surg" / keys_file
    ]
    
    for path in possible_paths:
        if path.exists():
            logger.debug(f"Loading API keys from {path}")
            try:
                with open(path, 'r') as f:
                    config = json.load(f)
                logger.success(f"API keys loaded from {path}")
                return config
            except Exception as e:
                logger.error(f"Failed to parse {path}: {str(e)}")
                continue
    
    # If not found, try environment variables as fallback
    logger.warning(f"Could not find {keys_file}, attempting to load from environment variables")
    return _load_from_env()

def _load_from_env() -> Dict[str, str]:
    """
    Load API keys from environment variables.
    Returns:
        A dictionary containing API keys loaded from environment variables.
    """
    config = {
        'llm_providers': {},
        'default_provider': 'openai',
        'settings': {
            'temperature': 0.7,
            'max_tokens': 2000,
            'timeout': 60,
            'max_retries': 3
        }
    }

    # OpenAI
    if os.getenv('OPENAI_API_KEY'):
        config['llm_providers']['openai'] = {
            'api_key': os.getenv('OPENAI_API_KEY'),
            'model': os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
            'base_url': 'https://api.openai.com/v1'
        }
    
    # Anthropic
    if os.getenv('ANTHROPIC_API_KEY'):
        config['llm_providers']['anthropic'] = {
            'api_key': os.getenv('ANTHROPIC_API_KEY'),
            'model': os.getenv('ANTHROPIC_MODEL', 'claude-3-5-sonnet-20241022'),
            'base_url': 'https://api.anthropic.com/v1'
        }
    
    # Google
    if os.getenv('GOOGLE_API_KEY'):
        config['llm_providers']['google'] = {
            'api_key': os.getenv('GOOGLE_API_KEY'),
            'model': os.getenv('GOOGLE_MODEL', 'gemini-pro'),
            'base_url': 'https://generativelanguage.googleapis.com/v1beta/openai/'
        }
    
    if not config['llm_providers']:
        logger.warning("No API keys found in environment variables")
    
    return config

def get_llm_config(provider: str = None, keys_file: str = "keys.json") -> Optional[Dict[str, Any]]:
    """
    Retrieve LLM configuration for a specified provider.
    
    Args:
        provider: The name of the LLM provider (e.g., 'openai', 'anthropic', 'google').
        keys_file: Path to the JSON file containing API keys and configurations.
    Returns:
        A dictionary containing the configuration for the specified provider, or None if not found.
    Raises:
        ValueError: If the specified provider is not found in the configuration.
    """
    config = load_api_keys(keys_file)
    
    if provider is None:
        provider = config.get('default_provider','openai')
        logger.debug(f"No provider specified, using default: {provider}")
    
    llm_providers = config.get('llm_providers', {})
    
    if provider not in llm_providers:
        error_msg = f"LLM provider '{provider}' not found in configuration."
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    llm_config = llm_providers[provider]
    llm_config.update(config.get('settings', {}))
    
    logger.success(f"LLM configuration loaded for provider: {provider}")
    return llm_config


def load_config(config_file: str = "config.yaml") -> Dict[str, Any]:
    """
    Load general configuration from YAML file.
    
    Args:
        config_file: Path to config file (YAML or JSON)
        
    Returns:
        Configuration dictionary
    """
    possible_paths = [
        Path(config_file),
        Path.cwd() / config_file,
        Path(__file__).parent.parent.parent / config_file,
        Path.home() / ".surg" / config_file
    ]
    
    for path in possible_paths:
        if path.exists():
            logger.debug(f"Loading config from {path}")
            try:
                with open(path, 'r') as f:
                    if path.suffix in ['.yaml', '.yml']:
                        config = yaml.safe_load(f)
                    else:
                        config = json.load(f)
                logger.success(f"Config loaded from {path}")
                return config
            except Exception as e:
                logger.error(f"Failed to parse {path}: {str(e)}")
                continue
    
    logger.warning(f"Config file {config_file} not found, using defaults")
    return _get_default_config()


def _get_default_config() -> Dict[str, Any]:
    """
    Get default configuration.
    
    Returns:
        Default configuration dictionary
    """
    return {
        'logging': {
            'level': 'INFO',
            'log_dir': 'logs',
            'max_file_size': 10485760,  # 10MB
            'backup_count': 5
        },
        'data': {
            'cache_dir': '.cache',
            'datasets_dir': 'datasets'
        },
        'training': {
            'default_epochs': 10,
            'batch_size': 256,
            'learning_rate': 0.001,
            'early_stopping_patience': 5
        },
        'evaluation': {
            'k_values': [5, 10, 20],
            'metrics': ['precision', 'recall', 'ndcg']
        },
        'genai': {
            'enable': False,
            'default_provider': 'openai',
            'temperature': 0.7,
            'max_tokens': 2000
        }
    }


def save_config(config: Dict[str, Any], config_file: str = "config.yaml") -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary to save
        config_file: Path to save config file
    """
    path = Path(config_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(path, 'w') as f:
            if path.suffix in ['.yaml', '.yml']:
                yaml.dump(config, f, default_flow_style=False)
            else:
                json.dump(config, f, indent=2)
        logger.success(f"Configuration saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save configuration: {str(e)}")
        raise


def get_env_var(key: str, default: Any = None, required: bool = False) -> Any:
    """
    Get environment variable with optional default and required check.
    
    Args:
        key: Environment variable name
        default: Default value if not found
        required: Whether the variable is required
        
    Returns:
        Environment variable value or default
        
    Raises:
        ValueError: If required=True and variable not found
    """
    value = os.getenv(key, default)
    
    if required and value is None:
        raise ValueError(f"Required environment variable '{key}' not set")
    
    return value


def update_config(updates: Dict[str, Any], config_file: str = "config.yaml") -> Dict[str, Any]:
    """
    Update existing configuration with new values.
    
    Args:
        updates: Dictionary with updates to apply
        config_file: Path to config file
        
    Returns:
        Updated configuration dictionary
    """
    config = load_config(config_file)
    
    # Deep update
    def deep_update(base: Dict, updates: Dict) -> Dict:
        for key, value in updates.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                base[key] = deep_update(base[key], value)
            else:
                base[key] = value
        return base
    
    config = deep_update(config, updates)
    save_config(config, config_file)
    
    logger.info(f"Configuration updated with {len(updates)} changes")
    return config


class ConfigManager:
    """
    Configuration manager for SURG system.
    
    Provides centralized access to all configuration settings
    with validation and type checking.
    """
    
    def __init__(self, config_file: str = "config.yaml", keys_file: str = "keys.json"):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to main config file
            keys_file: Path to API keys file
        """
        self.config_file = config_file
        self.keys_file = keys_file
        self._config = None
        self._keys = None
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get general configuration (lazy load)."""
        if self._config is None:
            self._config = load_config(self.config_file)
        return self._config
    
    @property
    def keys(self) -> Dict[str, Any]:
        """Get API keys configuration (lazy load)."""
        if self._keys is None:
            self._keys = load_api_keys(self.keys_file)
        return self._keys
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key (supports dot notation).
        
        Args:
            key: Configuration key (e.g., 'logging.level')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key (supports dot notation).
        
        Args:
            key: Configuration key (e.g., 'logging.level')
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        logger.debug(f"Configuration updated: {key} = {value}")
    
    def get_llm_config(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """Get LLM provider configuration."""
        return get_llm_config(provider, self.keys_file)
    
    def reload(self) -> None:
        """Reload configuration from files."""
        self._config = None
        self._keys = None
        logger.info("Configuration reloaded")
    
    def save(self) -> None:
        """Save current configuration to file."""
        if self._config is not None:
            save_config(self._config, self.config_file)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"ConfigManager(config_file='{self.config_file}', keys_file='{self.keys_file}')"


# Global config manager instance
_config_manager = None


def get_config_manager(config_file: str = "config.yaml", 
                       keys_file: str = "keys.json") -> ConfigManager:
    """
    Get global configuration manager instance (singleton).
    
    Args:
        config_file: Path to main config file
        keys_file: Path to API keys file
        
    Returns:
        ConfigManager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_file, keys_file)
    return _config_manager
