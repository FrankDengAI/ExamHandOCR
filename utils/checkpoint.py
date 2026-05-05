"""
Checkpoint saving and loading utilities.
"""

import torch
from pathlib import Path
from typing import Dict, Optional


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch: int,
    metrics: Dict,
    filepath: str,
    additional_info: Optional[Dict] = None,
):
    """
    Save training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch
        metrics: Metrics dictionary
        filepath: Path to save checkpoint
        additional_info: Additional information to save
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
    }
    
    if additional_info:
        checkpoint.update(additional_info)
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    model,
    filepath: str,
    optimizer=None,
    scheduler=None,
    strict: bool = True,
) -> Dict:
    """
    Load training checkpoint.
    
    Args:
        model: Model to load state into
        filepath: Path to checkpoint
        optimizer: Optional optimizer to load state
        scheduler: Optional scheduler to load state
        strict: Whether to strictly enforce state dict matching
    
    Returns:
        Dictionary with epoch and metrics
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location='cpu')
    
    # Load model state
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=strict)
    else:
        model.load_state_dict(checkpoint, strict=strict)
    
    # Load optimizer state
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Checkpoint loaded from {filepath}")
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'metrics': checkpoint.get('metrics', {}),
    }


def load_pretrained_encoder(model, pretrained_path: str, strict: bool = False):
    """
    Load pretrained encoder weights (e.g., from MAE pre-training).
    
    Args:
        model: Model with encoder
        pretrained_path: Path to pretrained encoder
        strict: Whether strict loading
    """
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    
    # Extract encoder state
    if 'encoder_state_dict' in checkpoint:
        encoder_state = checkpoint['encoder_state_dict']
    elif 'model_state_dict' in checkpoint:
        # Try to extract encoder from full model
        full_state = checkpoint['model_state_dict']
        encoder_state = {k.replace('encoder.', ''): v for k, v in full_state.items() if k.startswith('encoder.')}
    else:
        encoder_state = checkpoint
    
    # Load into model's encoder
    # This assumes model has an 'encoder' attribute
    if hasattr(model, 'encoder'):
        model.encoder.load_state_dict(encoder_state, strict=strict)
        print(f"Pretrained encoder loaded from {pretrained_path}")
    else:
        print("Warning: Model does not have 'encoder' attribute, skipping encoder loading")
