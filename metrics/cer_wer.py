"""
Character Error Rate (CER) and Word Error Rate (WER) calculations.

Standard evaluation metrics for optical character recognition (OCR).
These metrics provide the foundation for assessing model performance.

Definitions:
- CER: (Substitutions + Insertions + Deletions) / Total Reference Characters
- WER: (Substitutions + Insertions + Deletions) / Total Reference Words

For handwritten Chinese text:
- CER operates on individual characters (每个汉字)
- WER operates on whitespace-delimited segments (typically sentences)
"""

import editdistance
from typing import List, Dict, Tuple, Optional
import numpy as np


def calculate_cer(predictions: List[str], references: List[str]) -> float:
    """
    Calculate Character Error Rate (CER).
    
    CER measures the edit distance between predicted and reference text
    at the character level, normalized by reference length.
    
    Edit operations include:
    - Substitution: Character replaced with different character
    - Insertion: Extra character in prediction
    - Deletion: Missing character in prediction
    
    Formula:
        CER = (S + I + D) / N
        where S=substitutions, I=insertions, D=deletions, N=reference length
    
    Args:
        predictions: List of predicted strings
        references: List of ground truth reference strings
    
    Returns:
        CER as a float between 0.0 and 1.0
    
    Raises:
        ValueError: If predictions and references have different lengths
    
    Example:
        >>> predictions = ["hello", "world"]
        >>> references = ["hallo", "world"]
        >>> calculate_cer(predictions, references)
        0.1  # 1 substitution out of 10 total characters
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Predictions ({len(predictions)}) and references ({len(references)}) "
            "must have the same length"
        )
    
    total_distance = 0
    total_chars = 0
    
    for pred, ref in zip(predictions, references):
        # Convert strings to character lists
        pred_chars = list(pred)
        ref_chars = list(ref)
        
        # Calculate Levenshtein distance
        distance = editdistance.eval(pred_chars, ref_chars)
        
        total_distance += distance
        total_chars += len(ref_chars)
    
    if total_chars == 0:
        return 0.0
    
    cer = total_distance / total_chars
    return cer


def calculate_wer(predictions: List[str], references: List[str]) -> float:
    """
    Calculate Word Error Rate (WER).
    
    WER measures edit distance at the word level. For Chinese text,
    "words" are typically whitespace-delimited segments, which may be
    phrases or sentences depending on tokenization.
    
    Formula:
        WER = (S + I + D) / N
        where S=substitutions, I=insertions, D=deletions, N=reference words
    
    Args:
        predictions: List of predicted strings
        references: List of ground truth reference strings
    
    Returns:
        WER as a float between 0.0 and 1.0
    
    Note:
        For Chinese text without explicit word boundaries, WER is computed
        on whitespace-delimited segments, which may be suboptimal.
        CER is generally preferred for Chinese OCR evaluation.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Predictions ({len(predictions)}) and references ({len(references)}) "
            "must have the same length"
        )
    
    total_distance = 0
    total_words = 0
    
    for pred, ref in zip(predictions, references):
        # Split on whitespace to get "words"
        pred_words = pred.split()
        ref_words = ref.split()
        
        # Calculate edit distance at word level
        distance = editdistance.eval(pred_words, ref_words)
        
        total_distance += distance
        total_words += len(ref_words)
    
    if total_words == 0:
        return 0.0
    
    wer = total_distance / total_words
    return wer


def calculate_cer_per_sample(
    predictions: List[str],
    references: List[str],
) -> List[float]:
    """
    Calculate CER for each sample individually.
    
    Useful for analyzing per-sample errors and computing statistics
    across the dataset (e.g., mean, std, percentiles).
    
    Args:
        predictions: List of predicted strings
        references: List of reference strings
    
    Returns:
        List of CER values (one per sample)
    """
    cers = []
    
    for pred, ref in zip(predictions, references):
        pred_chars = list(pred)
        ref_chars = list(ref)
        
        if len(ref_chars) == 0:
            # Empty reference case
            cer = 0.0 if len(pred_chars) == 0 else 1.0
        else:
            distance = editdistance.eval(pred_chars, ref_chars)
            cer = distance / len(ref_chars)
        
        cers.append(cer)
    
    return cers


def calculate_cer_by_subject(
    predictions: List[str],
    references: List[str],
    subjects: List[str],
) -> Dict[str, float]:
    """
    Calculate CER grouped by academic subject.
    
    This analysis reveals subject-specific performance differences,
    particularly important for ExamHandOCR where Mathematics and Chemistry
    show significantly higher CER due to mathematical expressions.
    
    Paper results (Table in Section 7.4):
        Mathematics: 10.02% CER (8.3× human rate)
        Chinese: 4.21% CER
        English: 3.94% CER
    
    Args:
        predictions: List of predicted strings
        references: List of reference strings
        subjects: List of subject labels for each sample
    
    Returns:
        Dictionary mapping subject name to CER
    """
    # Group by subject
    subject_stats = {}
    
    for pred, ref, subj in zip(predictions, references, subjects):
        if subj not in subject_stats:
            subject_stats[subj] = {'distance': 0, 'chars': 0}
        
        pred_chars = list(pred)
        ref_chars = list(ref)
        
        distance = editdistance.eval(pred_chars, ref_chars)
        
        subject_stats[subj]['distance'] += distance
        subject_stats[subj]['chars'] += len(ref_chars)
    
    # Calculate per-subject CER
    subject_cers = {}
    for subj, stats in subject_stats.items():
        if stats['chars'] > 0:
            subject_cers[subj] = stats['distance'] / stats['chars']
        else:
            subject_cers[subj] = 0.0
    
    return subject_cers


def calculate_cer_by_style(
    predictions: List[str],
    references: List[str],
    styles: List[str],
) -> Dict[str, float]:
    """
    Calculate CER grouped by handwriting style.
    
    Paper results (Table 7, Section 4.5):
        Cursive/Mixed: 14.27% CER (highest error rate)
        Running: 8.42% CER
        Regular: 7.14% CER
        Print-like: 6.03% CER (lowest error rate)
    
    Args:
        predictions: List of predicted strings
        references: List of reference strings
        styles: List of handwriting style labels
    
    Returns:
        Dictionary mapping style to CER
    """
    style_stats = {}
    
    for pred, ref, style in zip(predictions, references, styles):
        if style not in style_stats:
            style_stats[style] = {'distance': 0, 'chars': 0}
        
        pred_chars = list(pred)
        ref_chars = list(ref)
        
        distance = editdistance.eval(pred_chars, ref_chars)
        style_stats[style]['distance'] += distance
        style_stats[style]['chars'] += len(ref_chars)
    
    style_cers = {}
    for style, stats in style_stats.items():
        if stats['chars'] > 0:
            style_cers[style] = stats['distance'] / stats['chars']
        else:
            style_cers[style] = 0.0
    
    return style_cers


def calculate_cer_confidence_interval(
    predictions: List[str],
    references: List[str],
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Calculate CER with confidence interval using bootstrap resampling.
    
    Provides uncertainty estimates for the CER metric.
    
    Args:
        predictions: List of predicted strings
        references: List of reference strings
        confidence: Confidence level (default 0.95 for 95% CI)
    
    Returns:
        Tuple of (mean_cer, lower_bound, upper_bound)
    """
    per_sample_cers = calculate_cer_per_sample(predictions, references)
    
    # Bootstrap resampling
    n_bootstrap = 1000
    n_samples = len(per_sample_cers)
    bootstrap_means = []
    
    np.random.seed(42)
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        bootstrap_sample = [per_sample_cers[i] for i in indices]
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    # Calculate confidence interval
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, alpha/2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
    mean = np.mean(per_sample_cers)
    
    return mean, lower, upper


def calculate_cer_statistics(predictions: List[str], references: List[str]) -> Dict:
    """
    Calculate comprehensive CER statistics.
    
    Args:
        predictions: List of predicted strings
        references: List of reference strings
    
    Returns:
        Dictionary with various CER statistics
    """
    per_sample_cers = calculate_cer_per_sample(predictions, references)
    
    stats = {
        'mean': np.mean(per_sample_cers),
        'std': np.std(per_sample_cers),
        'median': np.median(per_sample_cers),
        'min': np.min(per_sample_cers),
        'max': np.max(per_sample_cers),
        'percentile_25': np.percentile(per_sample_cers, 25),
        'percentile_75': np.percentile(per_sample_cers, 75),
        'percentile_90': np.percentile(per_sample_cers, 90),
        'percentile_95': np.percentile(per_sample_cers, 95),
    }
    
    return stats


class CERMetric:
    """
    Class-based CER metric for tracking during training.
    
    Accumulates predictions and references over batches, then
    computes final CER at epoch end.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset accumulated statistics."""
        self.all_predictions = []
        self.all_references = []
    
    def update(self, predictions: List[str], references: List[str]):
        """Accumulate batch results."""
        self.all_predictions.extend(predictions)
        self.all_references.extend(references)
    
    def compute(self) -> float:
        """Compute final CER."""
        if len(self.all_predictions) == 0:
            return 0.0
        return calculate_cer(self.all_predictions, self.all_references)
    
    def compute_by_sample(self) -> List[float]:
        """Get per-sample CERs."""
        return calculate_cer_per_sample(self.all_predictions, self.all_references)


class WERMetric:
    """
    Class-based WER metric for tracking during training.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset accumulated statistics."""
        self.all_predictions = []
        self.all_references = []
    
    def update(self, predictions: List[str], references: List[str]):
        """Accumulate batch results."""
        self.all_predictions.extend(predictions)
        self.all_references.extend(references)
    
    def compute(self) -> float:
        """Compute final WER."""
        if len(self.all_predictions) == 0:
            return 0.0
        return calculate_wer(self.all_predictions, self.all_references)
