"""
Self-supervised pre-training script using MAE.
Implements the SSL pre-training described in Section 7.1.
"""

import os
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models import MaskedAutoencoder
from data import SSLDataset, get_ssl_dataloader
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.logger import setup_logger


def pretrain_ssl(
    model: MaskedAutoencoder,
    train_loader,
    config: Dict,
    device='cuda',
    output_dir='./outputs_ssl',
):
    """
    Pre-train model using Masked Autoencoding.
    
    Paper configuration (Section 7.1):
    - Pre-training: 100 epochs on 3.15M unannotated images
    - Optimizer: AdamW, lr=1.5e-4, warmup 40 epochs
    - Mask ratio: 0.75
    - Patch size: 16
    
    Args:
        model: MAE model
        train_loader: DataLoader for unannotated images
        config: Training configuration
        device: Device to train on
        output_dir: Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup
    logger = setup_logger('ssl_pretrain', output_dir / 'ssl_train.log')
    logger.info(f"Starting SSL pre-training with config: {config}")
    
    writer = SummaryWriter(output_dir / 'tensorboard_ssl')
    
    model = model.to(device)
    
    # Optimizer
    lr = config.get('learning_rate', 1.5e-4)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=config.get('weight_decay', 0.05),
        betas=(0.9, 0.95),
    )
    
    # Scheduler with warmup
    num_epochs = config.get('epochs', 100)
    warmup_epochs = config.get('warmup_epochs', 40)
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [SSL]')
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['images'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            loss, pred, mask = model(images)
            
            # Backward
            loss.backward()
            
            # Gradient clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            # Log
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            if global_step % 100 == 0:
                writer.add_scalar('ssl/loss', loss.item(), global_step)
                writer.add_scalar('ssl/lr', optimizer.param_groups[0]['lr'], global_step)
                
                # Log mask ratio
                mask_ratio = mask.sum() / mask.numel()
                writer.add_scalar('ssl/mask_ratio', mask_ratio, global_step)
        
        avg_loss = epoch_loss / num_batches
        logger.info(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, LR={optimizer.param_groups[0]['lr']:.6f}")
        
        writer.add_scalar('ssl/epoch_loss', avg_loss, epoch)
        
        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                {'loss': avg_loss},
                output_dir / 'best_mae.pth'
            )
        
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                {'loss': avg_loss},
                output_dir / f'mae_epoch_{epoch+1}.pth'
            )
        
        scheduler.step()
    
    writer.close()
    logger.info("SSL pre-training completed!")
    
    return model


def extract_encoder_for_finetuning(mae_checkpoint_path: str, output_path: str):
    """
    Extract encoder from trained MAE for downstream fine-tuning.
    
    Args:
        mae_checkpoint_path: Path to MAE checkpoint
        output_path: Path to save encoder checkpoint
    """
    checkpoint = torch.load(mae_checkpoint_path, map_location='cpu')
    
    # Extract encoder state dict
    encoder_state_dict = {}
    for key, value in checkpoint['model_state_dict'].items():
        if key.startswith('encoder.'):
            # Remove 'encoder.' prefix
            new_key = key[8:]
            encoder_state_dict[new_key] = value
    
    # Save
    torch.save({
        'encoder_state_dict': encoder_state_dict,
        'epoch': checkpoint.get('epoch', 0),
    }, output_path)
    
    print(f"Extracted encoder saved to {output_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='Path to unannotated images')
    parser.add_argument('--image_list', type=str, default=None, help='Optional list of images')
    parser.add_argument('--output_dir', type=str, default='./outputs_ssl')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1.5e-4)
    parser.add_argument('--warmup_epochs', type=int, default=40)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=8)
    
    args = parser.parse_args()
    
    # Create MAE model
    model = MaskedAutoencoder(
        img_size=(384, 128),
        patch_size=args.patch_size,
        mask_ratio=args.mask_ratio,
        embed_dim=768,
        encoder_depth=12,
        decoder_dim=512,
        decoder_depth=8,
    )
    
    # Create dataset
    from data.transforms import get_ssl_transforms
    
    train_transform = get_ssl_transforms(patch_size=args.patch_size)
    
    train_dataset = SSLDataset(
        data_root=args.data_root,
        image_list=args.image_list,
        transform=train_transform,
    )
    
    train_loader = get_ssl_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    # Config
    config = {
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'warmup_epochs': args.warmup_epochs,
        'weight_decay': 0.05,
    }
    
    # Train
    model = pretrain_ssl(model, train_loader, config, args.device, args.output_dir)
