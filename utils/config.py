"""
Configuration management utilities.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any
from omegaconf import OmegaConf


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to config file
    
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif config_path.suffix == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    return config


def save_config(config: Dict[str, Any], output_path: str):
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save config
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        if output_path.suffix in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False)
        elif output_path.suffix == '.json':
            json.dump(config, f, indent=2)


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
    
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def get_default_config() -> Dict[str, Any]:
    """Get default configuration for ExamHandOCR."""
    return {
        'data': {
            'data_root': './data/ExamHandOCR',
            'annotation_file': './data/annotations.json',
            'batch_size': 32,
            'num_workers': 4,
            'pin_memory': True,
        },
        'model': {
            'type': 'trocr',  # crnn, abinet, trocr, vit_ocr
            'pretrained': True,
            'pretrained_path': None,
            'ssl_pretrained': False,
            'ssl_pretrained_path': None,
        },
        'training': {
            'epochs': 50,
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'warmup_epochs': 5,
            'grad_clip': 1.0,
            'save_interval': 10,
            'eval_interval': 1,
        },
        'ssl': {
            'enabled': False,
            'epochs': 100,
            'learning_rate': 1.5e-4,
            'warmup_epochs': 40,
            'mask_ratio': 0.75,
            'patch_size': 16,
        },
        'evaluation': {
            'calculate_oqs': True,
            'calculate_ri': True,
            'evaluate_tracks': True,
        },
        'output': {
            'output_dir': './outputs',
            'log_dir': './logs',
            'tensorboard_dir': './tensorboard',
        }
    }
