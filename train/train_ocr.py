"""
OCR model training script.
Implements training loop for CRNN, ABINet, TrOCR, and ViT-OCR.
"""

import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from data import get_dataloader
from metrics import calculate_cer, calculate_esa_cer
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.logger import setup_logger


def train_ocr_model(
    model,
    train_loader,
    val_loader,
    tokenizer,
    config: Dict,
    device='cuda',
    output_dir='./outputs',
):
    """
    Train an OCR model.
    
    Args:
        model: OCR model (CRNN, ABINet, TrOCR, or ViT-OCR)
        train_loader: Training data loader
        val_loader: Validation data loader
        tokenizer: Tokenizer for encoding/decoding
        config: Training configuration dict
        device: Device to train on
        output_dir: Directory to save outputs
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    logger = setup_logger('train_ocr', output_dir / 'train.log')
    logger.info(f"Starting training with config: {config}")
    
    # TensorBoard
    writer = SummaryWriter(output_dir / 'tensorboard')
    
    # Model to device
    model = model.to(device)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get('learning_rate', 1e-4),
        weight_decay=config.get('weight_decay', 0.01),
    )
    
    # Scheduler (cosine decay with warmup)
    num_epochs = config.get('epochs', 50)
    warmup_epochs = config.get('warmup_epochs', 5)
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Loss function
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True) if config.get('use_ctc') else nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    
    # Training loop
    best_cer = float('inf')
    global_step = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_cer = 0.0
        num_train_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, batch in enumerate(pbar):
            images = batch['images'].to(device)
            transcriptions = batch['transcriptions']
            
            # Encode targets
            targets = tokenizer.batch_encode(transcriptions, padding=True).to(device)
            
            # Forward
            optimizer.zero_grad()
            
            if config.get('model_type') == 'crnn':
                # CRNN uses CTC loss
                logits = model(images)  # (T, B, num_classes)
                log_probs = nn.functional.log_softmax(logits, dim=-1)
                
                # CTC inputs need to be (T, B, C)
                T, B, C = log_probs.shape
                input_lengths = torch.full((B,), T, dtype=torch.long)
                target_lengths = (targets != tokenizer.pad_id).sum(dim=1)
                
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
            else:
                # Other models use cross-entropy
                outputs = model(images, targets[:, :-1])  # Exclude last token for input
                if isinstance(outputs, dict):
                    loss = outputs['loss']
                else:
                    logits = outputs
                    loss = nn.functional.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        targets[:, 1:].reshape(-1),
                        ignore_index=tokenizer.pad_id,
                    )
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            num_train_batches += 1
            global_step += 1
            
            # Log
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            writer.add_scalar('train/loss', loss.item(), global_step)
            
            # Periodic evaluation on small batch
            if batch_idx % 100 == 0 and batch_idx > 0:
                with torch.no_grad():
                    if config.get('model_type') == 'crnn':
                        preds = model.predict(images[:4])
                        pred_texts = tokenizer.batch_decode([torch.tensor(p) for p in preds])
                    else:
                        generated = model.generate(images[:4]) if hasattr(model, 'generate') else model(images[:4])
                        pred_texts = tokenizer.batch_decode(generated) if hasattr(tokenizer, 'batch_decode') else [''] * 4
                    
                    ref_texts = transcriptions[:4]
                    batch_cer = calculate_cer(pred_texts, ref_texts)
                    writer.add_scalar('train/cer', batch_cer, global_step)
        
        avg_train_loss = train_loss / num_train_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_refs = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for batch in pbar:
                images = batch['images'].to(device)
                transcriptions = batch['transcriptions']
                
                # Generate predictions
                if config.get('model_type') == 'crnn':
                    pred_ids = model.predict(images)
                    pred_texts = tokenizer.batch_decode([torch.tensor(p) for p in pred_ids])
                else:
                    generated = model.generate(images) if hasattr(model, 'generate') else model(images)
                    if isinstance(generated, dict):
                        generated = generated['logits'].argmax(dim=-1)
                    pred_texts = tokenizer.batch_decode(generated)
                
                all_preds.extend(pred_texts)
                all_refs.extend(transcriptions)
        
        # Calculate metrics
        val_cer = calculate_cer(all_preds, all_refs)
        val_esa_cer = calculate_esa_cer(all_preds, all_refs, alpha=3.0)
        
        logger.info(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val CER={val_cer:.4f}, Val ESA-CER={val_esa_cer:.4f}")
        
        # TensorBoard
        writer.add_scalar('val/loss', val_loss / len(val_loader), epoch)
        writer.add_scalar('val/cer', val_cer, epoch)
        writer.add_scalar('val/esa_cer', val_esa_cer, epoch)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
        
        # Save checkpoint
        if val_cer < best_cer:
            best_cer = val_cer
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                {'cer': val_cer, 'esa_cer': val_esa_cer},
                output_dir / 'best_model.pth'
            )
            logger.info(f"Saved best model with CER={val_cer:.4f}")
        
        # Regular checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                {'cer': val_cer, 'esa_cer': val_esa_cer},
                output_dir / f'checkpoint_epoch_{epoch+1}.pth'
            )
        
        scheduler.step()
    
    writer.close()
    logger.info("Training completed!")
    
    return model


if __name__ == '__main__':
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='trocr', choices=['crnn', 'abinet', 'trocr', 'vit_ocr'])
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--annotation_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # Import model
    from models import CRNN, ABINet, TrOCRModel, ViTOCR
    
    # Create model
    model_map = {
        'crnn': lambda: CRNN(num_classes=5000),
        'abinet': lambda: ABINet(num_classes=5000),
        'trocr': lambda: TrOCRModel(),
        'vit_ocr': lambda: ViTOCR(num_classes=5000),
    }
    
    model = model_map[args.model]()
    
    # Create data loaders
    from data import ExamHandOCRDataset, get_dataloader
    from data.transforms import get_train_transforms, get_val_transforms
    
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()
    
    train_dataset = ExamHandOCRDataset(
        data_root=args.data_root,
        split='train-sup',
        annotation_file=args.annotation_file,
        transform=train_transform,
    )
    
    val_dataset = ExamHandOCRDataset(
        data_root=args.data_root,
        split='val',
        annotation_file=args.annotation_file,
        transform=val_transform,
    )
    
    train_loader = get_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Train
    config = {
        'model_type': args.model,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'use_ctc': args.model == 'crnn',
    }
    
    train_ocr_model(model, train_loader, val_loader, None, config, args.device, args.output_dir)
