"""
Layout analysis evaluation script.
Evaluates text-line segmentation models.
"""

import json
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from metrics.layout_metrics import compute_layout_metrics, calculate_miou


def evaluate_layout(
    model,
    test_loader,
    device='cuda',
) -> Dict:
    """
    Evaluate layout model on test set.
    
    Args:
        model: Layout model (U-Net, Mask R-CNN, DETR)
        test_loader: Test data loader
        device: Device to run on
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    model = model.to(device)
    
    all_predictions = []
    all_ground_truths = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating Layout'):
            images = batch['images'].to(device)
            gt_masks = batch['masks']
            
            # Forward pass
            outputs = model(images)
            
            # Convert outputs to predictions
            if isinstance(outputs, dict):
                # DETR-style output
                pred_masks = outputs.get('pred_masks', None)
            else:
                # U-Net style output (logits)
                pred_probs = torch.softmax(outputs, dim=1)
                pred_masks = (pred_probs.argmax(dim=1) == 1).cpu().numpy()
            
            # Store predictions and ground truths
            for i in range(len(images)):
                pred_mask = pred_masks[i] if pred_masks is not None else None
                gt_mask = gt_masks[i].cpu().numpy()
                
                all_predictions.append({'masks': pred_mask})
                all_ground_truths.append({'masks': gt_mask})
    
    # Calculate metrics
    metrics = compute_layout_metrics(all_predictions, all_ground_truths)
    
    return metrics


def save_layout_results(results: Dict, output_path: str):
    """Save layout evaluation results."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Layout results saved to {output_path}")


def print_layout_results(results: Dict):
    """Print layout evaluation results."""
    print("\n" + "="*60)
    print("LAYOUT EVALUATION RESULTS")
    print("="*60)
    print(f"\nMetrics:")
    print(f"  mIoU:      {results.get('mIoU', 0)*100:.2f}%")
    print(f"  F1@0.5:    {results.get('F1@0.5', 0)*100:.2f}%")
    print(f"  F1@0.75:   {results.get('F1@0.75', 0)*100:.2f}%")
    print("="*60 + "\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--model_type', type=str, default='unet', choices=['unet', 'detr'])
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--annotation_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, default='layout_results.json')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # Load model
    from models import UNetLayout, DETRLayout
    from utils.checkpoint import load_checkpoint
    
    model_map = {
        'unet': lambda: UNetLayout(in_channels=1, num_classes=2),
        'detr': lambda: DETRLayout(in_channels=1, num_queries=100),
    }
    
    model = model_map[args.model_type]()
    load_checkpoint(model, args.model_path)
    
    # Create dataset
    from data import LayoutDataset, get_dataloader
    from data.transforms import get_layout_transforms
    
    test_dataset = LayoutDataset(
        data_root=args.data_root,
        split='test-sup',
        annotation_file=args.annotation_file,
        transform=get_layout_transforms(is_train=False),
    )
    
    test_loader = get_dataloader(test_dataset, batch_size=args.batch_size, shuffle=False, task='layout')
    
    # Evaluate
    results = evaluate_layout(model, test_loader, args.device)
    
    # Print and save
    print_layout_results(results)
    save_layout_results(results, args.output_file)
