"""
Visualization utilities for predictions and analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import cv2


def visualize_predictions(
    images: List[np.ndarray],
    predictions: List[str],
    references: List[str],
    output_path: str,
    max_samples: int = 16,
    figsize: Tuple[int, int] = (20, 10),
):
    """
    Visualize OCR predictions vs references.
    
    Args:
        images: List of grayscale images
        predictions: List of predicted texts
        references: List of reference texts
        output_path: Path to save visualization
        max_samples: Maximum number of samples to visualize
        figsize: Figure size
    """
    num_samples = min(len(images), max_samples)
    cols = 4
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes.flatten()
    
    for idx in range(num_samples):
        ax = axes[idx]
        
        # Display image
        ax.imshow(images[idx], cmap='gray')
        ax.axis('off')
        
        # Add text
        pred_text = predictions[idx][:50] + '...' if len(predictions[idx]) > 50 else predictions[idx]
        ref_text = references[idx][:50] + '...' if len(references[idx]) > 50 else references[idx]
        
        title = f"Pred: {pred_text}\nRef: {ref_text}"
        
        # Highlight if different
        if predictions[idx] != references[idx]:
            ax.set_title(title, color='red', fontsize=8)
        else:
            ax.set_title(title, color='green', fontsize=8)
    
    # Hide unused subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {output_path}")


def plot_confusion_matrix(
    y_true: List,
    y_pred: List,
    labels: List[str],
    output_path: str,
    normalize: bool = True,
):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
        output_path: Path to save plot
        normalize: Whether to normalize
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', 
                xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_cer_by_subject(
    subject_cers: Dict[str, float],
    output_path: str,
):
    """
    Plot CER by subject.
    
    Args:
        subject_cers: Dictionary mapping subject to CER
        output_path: Path to save plot
    """
    subjects = list(subject_cers.keys())
    cers = [subject_cers[s] * 100 for s in subjects]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(subjects, cers, color='steelblue')
    plt.xlabel('Subject')
    plt.ylabel('CER (%)')
    plt.title('Character Error Rate by Subject')
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_quality_tier_analysis(
    tier_results: Dict,
    output_path: str,
):
    """
    Plot CER across quality tiers.
    
    Args:
        tier_results: Dictionary with tier results
        output_path: Path to save plot
    """
    tiers = ['High-OQS', 'Medium-OQS', 'Low-OQS']
    cers = [tier_results.get(tier, {}).get('CER', 0) * 100 for tier in tiers]
    
    plt.figure(figsize=(10, 6))
    colors = ['green', 'orange', 'red']
    bars = plt.bar(tiers, cers, color=colors)
    plt.xlabel('Quality Tier')
    plt.ylabel('CER (%)')
    plt.title('CER by Operational Quality Tier')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_annotation_efficiency_curve(
    sizes: List[int],
    cers: List[float],
    output_path: str,
):
    """
    Plot annotation efficiency curve.
    
    Args:
        sizes: List of annotation set sizes
        cers: List of CER values
        output_path: Path to save plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, [c * 100 for c in cers], marker='o', linewidth=2, markersize=8)
    plt.xscale('log')
    plt.xlabel('Number of Labeled Images')
    plt.ylabel('CER (%)')
    plt.title('Annotation Efficiency Curve')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_comparison_table(
    model_results: Dict[str, Dict],
    output_path: str,
):
    """
    Create comparison table of different models.
    
    Args:
        model_results: Dictionary mapping model name to results
        output_path: Path to save table (CSV)
    """
    import pandas as pd
    
    rows = []
    for model_name, results in model_results.items():
        row = {
            'Model': model_name,
            'CER (%)': f"{results.get('CER', 0) * 100:.2f}",
            'WER (%)': f"{results.get('WER', 0) * 100:.2f}",
            'ESA-CER (%)': f"{results.get('ESA-CER', 0) * 100:.2f}",
            'RI': f"{results.get('RI', 0):.3f}" if 'RI' in results else 'N/A',
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    
    print(f"Comparison table saved to {output_path}")
    return df


def visualize_attention_maps(
    image: np.ndarray,
    attention_weights: np.ndarray,
    output_path: str,
    patch_size: int = 16,
):
    """
    Visualize attention maps from transformer models.
    
    Args:
        image: Input image
        attention_weights: Attention weights (num_heads, num_patches, num_patches)
        output_path: Path to save visualization
        patch_size: Patch size
    """
    # Average across heads
    attn = attention_weights.mean(axis=0)
    
    # Get attention from CLS token to patches
    cls_attn = attn[0, 1:]  # Skip CLS token itself
    
    # Reshape to image shape
    h, w = image.shape
    num_patches_h = h // patch_size
    num_patches_w = w // patch_size
    
    cls_attn = cls_attn.reshape(num_patches_h, num_patches_w)
    
    # Upsample to image size
    cls_attn = cv2.resize(cls_attn, (w, h))
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    axes[1].imshow(image, cmap='gray')
    axes[1].imshow(cls_attn, cmap='jet', alpha=0.6)
    axes[1].set_title('Attention Map')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_curves(
    history: Dict[str, List],
    output_path: str,
):
    """
    Plot training and validation curves.
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_cer', 'val_cer', etc.
        output_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss curves
    if 'train_loss' in history and 'val_loss' in history:
        axes[0, 0].plot(history['train_loss'], label='Train')
        axes[0, 0].plot(history['val_loss'], label='Val')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # CER curves
    if 'train_cer' in history and 'val_cer' in history:
        axes[0, 1].plot(history['train_cer'], label='Train')
        axes[0, 1].plot(history['val_cer'], label='Val')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('CER')
        axes[0, 1].set_title('CER Curves')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate
    if 'learning_rate' in history:
        axes[1, 0].plot(history['learning_rate'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
    
    # ESA-CER
    if 'val_esa_cer' in history:
        axes[1, 1].plot(history['val_esa_cer'])
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('ESA-CER')
        axes[1, 1].set_title('ESA-CER (Validation)')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
