"""
Robustness Index (RI) and Generalization Gap (GG) calculations.
Implements evaluation metrics from Section 5.2 and 5.3.
"""

import numpy as np
from typing import List, Dict, Tuple


class RobustnessIndex:
    """
    Robustness Index (RI) measures model sensitivity to image quality.
    
    Formula:
        RI = 1 - (CER_Low-OQS - CER_High-OQS) / CER_High-OQS
    
    Interpretation:
        RI = 1: Model is insensitive to image quality
        RI < 0: CER more than doubles from high to low quality
    
    Args:
        oqs_scores: List of OQS scores for test images
        cer_scores: List of CER scores for test images
    """
    
    def __init__(self):
        pass
    
    def calculate(
        self,
        oqs_scores: List[float],
        cer_scores: List[float],
    ) -> float:
        """
        Calculate RI from paired OQS and CER scores.
        
        Args:
            oqs_scores: List of OQS scores (higher = better quality)
            cer_scores: List of CER scores
        
        Returns:
            Robustness Index
        """
        if len(oqs_scores) != len(cer_scores):
            raise ValueError("OQS and CER score lists must have same length")
        
        if len(oqs_scores) == 0:
            return 0.0
        
        # Sort by OQS
        sorted_pairs = sorted(zip(oqs_scores, cer_scores), key=lambda x: x[0])
        
        # Split into tertiles
        n = len(sorted_pairs)
        tertile_size = max(n // 3, 1)
        
        # High quality (top tertile)
        high_oqs_cers = [cer for _, cer in sorted_pairs[-tertile_size:]]
        cer_high = np.mean(high_oqs_cers) if high_oqs_cers else 0.0
        
        # Low quality (bottom tertile)
        low_oqs_cers = [cer for _, cer in sorted_pairs[:tertile_size]]
        cer_low = np.mean(low_oqs_cers) if low_oqs_cers else 0.0
        
        # Calculate RI
        if cer_high == 0:
            return 1.0 if cer_low == 0 else -1.0
        
        ri = 1.0 - (cer_low - cer_high) / cer_high
        
        return ri
    
    def calculate_by_tier(
        self,
        oqs_scores: List[float],
        cer_scores: List[float],
    ) -> Dict[str, float]:
        """
        Calculate CER and RI by quality tier.
        
        Returns:
            Dictionary with per-tier statistics
        """
        if len(oqs_scores) != len(cer_scores):
            raise ValueError("Score lists must have same length")
        
        # Sort by OQS
        sorted_pairs = sorted(zip(oqs_scores, cer_scores), key=lambda x: x[0])
        n = len(sorted_pairs)
        tertile_size = max(n // 3, 1)
        
        # Split into tiers
        high_pairs = sorted_pairs[-tertile_size:]
        medium_pairs = sorted_pairs[tertile_size:-tertile_size]
        low_pairs = sorted_pairs[:tertile_size]
        
        # Calculate stats
        stats = {
            'High-OQS': {
                'CER': np.mean([cer for _, cer in high_pairs]),
                'count': len(high_pairs),
            },
            'Medium-OQS': {
                'CER': np.mean([cer for _, cer in medium_pairs]),
                'count': len(medium_pairs),
            },
            'Low-OQS': {
                'CER': np.mean([cer for _, cer in low_pairs]),
                'count': len(low_pairs),
            },
        }
        
        # Calculate overall RI
        stats['RI'] = self.calculate(oqs_scores, cer_scores)
        
        return stats


class GeneralizationGap:
    """
    Generalization Gap (GG) for cross-session evaluation.
    
    Formula:
        GG = |CER_source - CER_target|
    
    Lower GG indicates better cross-session generalization.
    """
    
    @staticmethod
    def calculate(cer_source: float, cer_target: float) -> float:
        """
        Calculate Generalization Gap.
        
        Args:
            cer_source: CER on source domain
            cer_target: CER on target domain
        
        Returns:
            Generalization Gap
        """
        return abs(cer_source - cer_target)
    
    @staticmethod
    def calculate_batch(
        source_preds: List[str],
        source_refs: List[str],
        target_preds: List[str],
        target_refs: List[str],
        metric_fn,
    ) -> Tuple[float, float, float]:
        """
        Calculate GG from predictions.
        
        Args:
            source_preds: Predictions on source domain
            source_refs: References for source domain
            target_preds: Predictions on target domain
            target_refs: References for target domain
            metric_fn: Metric function (e.g., calculate_cer)
        
        Returns:
            Tuple of (CER_source, CER_target, GG)
        """
        cer_source = metric_fn(source_preds, source_refs)
        cer_target = metric_fn(target_preds, target_refs)
        gg = GeneralizationGap.calculate(cer_source, cer_target)
        
        return cer_source, cer_target, gg


def calculate_ri(
    oqs_scores: List[float],
    cer_scores: List[float],
) -> float:
    """
    Convenience function to calculate Robustness Index.
    
    Args:
        oqs_scores: List of OQS scores
        cer_scores: List of CER scores
    
    Returns:
        Robustness Index
    """
    calculator = RobustnessIndex()
    return calculator.calculate(oqs_scores, cer_scores)


def calculate_gg(cer_source: float, cer_target: float) -> float:
    """
    Convenience function to calculate Generalization Gap.
    
    Args:
        cer_source: Source domain CER
        cer_target: Target domain CER
    
    Returns:
        Generalization Gap
    """
    return GeneralizationGap.calculate(cer_source, cer_target)


def evaluate_cross_session(
    model,
    source_loader,
    target_loader,
    tokenizer,
    device='cuda',
) -> Dict[str, float]:
    """
    Evaluate model on cross-session generalization task.
    
    Args:
        model: OCR model
        source_loader: DataLoader for source domain
        target_loader: DataLoader for target domain
        tokenizer: Tokenizer for decoding
        device: Device to run on
    
    Returns:
        Dictionary with CER_source, CER_target, GG
    """
    from .cer_wer import calculate_cer
    
    def evaluate_loader(loader):
        all_preds = []
        all_refs = []
        
        model.eval()
        with torch.no_grad():
            for batch in loader:
                images = batch['images'].to(device)
                references = batch['transcriptions']
                
                # Generate predictions
                generated = model.generate(images)
                predictions = tokenizer.batch_decode(generated)
                
                all_preds.extend(predictions)
                all_refs.extend(references)
        
        return calculate_cer(all_preds, all_refs)
    
    cer_source = evaluate_loader(source_loader)
    cer_target = evaluate_loader(target_loader)
    gg = calculate_gg(cer_source, cer_target)
    
    return {
        'CER_source': cer_source,
        'CER_target': cer_target,
        'GG': gg,
    }


# Import torch for evaluate_cross_session
import torch
