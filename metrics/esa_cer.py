"""
ExamScore-Aware Character Error Rate (ESA-CER) metric.

Novel evaluation metric introduced in ExamHandOCR (Section 5.1).

Motivation:
In examination grading, OCR errors in mathematical tokens (digits, operators,
variables) are substantially more consequential than errors in narrative text.
A misread digit in a physics derivation changes the numerical answer, while
a misread character in a history essay typically does not affect the score.

Formula (from Section 5.1):
    ESA-CER = Σ(w_i × 1[char_i is erroneous]) / Σ(w_i)

where the per-character weight is:
    w_i = α if char_i is a math-token (inside \( ... \))
    w_i = 1 otherwise

with α = 3.0 calibrated from grading rubric analysis across 500 sample
question-answer pairs.

Key Properties:
1. ESA-CER = CER when α = 1 (reduces to standard CER)
2. ESA-CER > CER when α > 1 (amplifies math errors)
3. Consistently reveals 1.30× amplification across all models (Table 1)
"""

import re
import editdistance
from typing import List, Dict, Optional, Tuple, Union
import numpy as np


class ESA_CER:
    """
    ExamScore-Aware Character Error Rate calculator.
    
    This metric differentially weights errors in mathematical tokens
to better align OCR accuracy with examination grading consequences.
    
    Mathematical Token Definition:
    - Digits (0-9)
    - Operators (+, -, ×, ÷, =, etc.)
    - Greek letters (α, β, γ, etc.)
    - LaTeX commands (\frac, \sqrt, etc.)
    - Content inside \( ... \) delimiters
    
    Implementation:
    1. Parse text to identify math regions (\( ... \))
    2. Compute per-character weights
    3. Calculate weighted edit distance
    4. Normalize by total weight
    
    Paper Calibration:
    - α = 3.0 (from rubric analysis of 500 Q&A pairs)
    - Consistent 1.30× ESA-CER/CER ratio across models
    
    Args:
        alpha: Weight for mathematical tokens (default: 3.0)
    """
    
    def __init__(self, alpha: float = 3.0):
        self.alpha = alpha
        
        # Define mathematical symbols and tokens
        self.math_symbols = set([
            # Basic operators
            '+', '-', '*', '/', '=', '≠', '≈', '<', '>', '≤', '≥', '≡', '±',
            # Superscript/subscript markers
            '^', '_',
            # Greek letters (lowercase)
            'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ',
            'ν', 'ξ', 'ο', 'π', 'ρ', 'σ', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω',
            # Greek letters (uppercase)
            'Γ', 'Δ', 'Θ', 'Λ', 'Ξ', 'Π', 'Σ', 'Φ', 'Ψ', 'Ω',
            # Mathematical notation
            '√', '∫', '∑', '∏', '∂', '·', '×', '÷', '∞', '∇',
            '∈', '∉', '∪', '∩', '⊂', '⊃', '⊆', '⊇',
            # Delimiters
            '(', ')', '[', ']', '{', '}',
        ])
        
        # LaTeX math commands
        self.latex_commands = [
            'frac', 'sqrt', 'sum', 'int', 'prod', 'lim', 'infty',
            'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'theta',
            'lambda', 'mu', 'pi', 'sigma', 'omega', 'Delta', 'Sigma',
            'left', 'right', 'cdot', 'times', 'div', 'pm', 'mp',
            'leq', 'geq', 'neq', 'approx', 'equiv',
            'rightarrow', 'leftarrow', 'Rightarrow', 'Leftarrow',
        ]
    
    def is_math_token(self, char: str, in_math_mode: bool) -> bool:
        """
        Determine if a character is a mathematical token.
        
        A token is considered mathematical if:
        1. It is inside LaTeX math delimiters (\( ... \)), OR
        2. It is a recognized math symbol, OR
        3. It is a digit (0-9), OR
        4. It is a LaTeX command
        
        Args:
            char: The character to check
            in_math_mode: Whether currently inside \( ... \)
        
        Returns:
            True if mathematical token, False otherwise
        """
        # Inside math delimiters - everything is mathematical
        if in_math_mode:
            return True
        
        # Check against known math symbols
        if char in self.math_symbols:
            return True
        
        # Digits are considered mathematical (grade-sensitive)
        if char.isdigit():
            return True
        
        # Check for LaTeX commands
        if char.startswith('\\') and len(char) > 1:
            cmd = char[1:]
            if cmd in self.latex_commands:
                return True
        
        return False
    
    def tokenize_with_math_regions(self, text: str) -> List[Tuple[str, bool]]:
        """
        Tokenize text and identify mathematical regions.
        
        Parses LaTeX math delimiters (\( ... \)) to track when we are
        in math mode, as all tokens inside math mode are weighted higher.
        
        Args:
            text: Input text with possible LaTeX math expressions
        
        Returns:
            List of (token, is_math_region) tuples
        """
        tokens = []
        i = 0
        in_math = False
        
        while i < len(text):
            # Check for LaTeX math delimiters
            # Inline math: \( ... \)
            if text[i:i+2] == '\\(':
                in_math = True
                i += 2
                continue
            elif text[i:i+2] == '\\)':
                in_math = False
                i += 2
                continue
            
            # Display math: \[ ... \]
            if text[i:i+2] == '\\[':
                in_math = True
                i += 2
                continue
            elif text[i:i+2] == '\\]':
                in_math = False
                i += 2
                continue
            
            # Add token with math region flag
            tokens.append((text[i], in_math))
            i += 1
        
        return tokens
    
    def compute_weights(self, text: str) -> List[float]:
        """
        Compute per-character weights for a text.
        
        Args:
            text: Input text
        
        Returns:
            List of weights (one per character)
        """
        tokens = self.tokenize_with_math_regions(text)
        weights = []
        
        for char, in_math in tokens:
            if in_math or self.is_math_token(char, in_math):
                weights.append(self.alpha)
            else:
                weights.append(1.0)
        
        return weights
    
    def calculate_alignment(
        self,
        prediction: str,
        reference: str,
    ) -> Tuple[List[int], List[int], List[Tuple[int, int, str]]]:
        """
        Compute alignment between prediction and reference.
        
        Uses edit distance algorithm to find optimal alignment,
        identifying which characters are substituted, inserted, or deleted.
        
        Args:
            prediction: Predicted text
            reference: Ground truth text
        
        Returns:
            Tuple of:
            - pred_aligned: Indices of aligned prediction characters
            - ref_aligned: Indices of aligned reference characters
            - operations: List of (pred_idx, ref_idx, op_type) where
              op_type is 'match', 'subst', 'insert', or 'delete'
        """
        pred_chars = list(prediction)
        ref_chars = list(reference)
        
        # Compute edit distance with backtracking
        m, n = len(pred_chars), len(ref_chars)
        
        # DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred_chars[i-1] == ref_chars[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],      # Deletion
                        dp[i][j-1],      # Insertion
                        dp[i-1][j-1],    # Substitution
                    )
        
        # Backtrack to find alignment
        operations = []
        i, j = m, n
        
        while i > 0 or j > 0:
            if i > 0 and j > 0 and pred_chars[i-1] == ref_chars[j-1]:
                # Match
                operations.append((i-1, j-1, 'match'))
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
                # Substitution
                operations.append((i-1, j-1, 'subst'))
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
                # Deletion (in prediction)
                operations.append((i-1, -1, 'delete'))
                i -= 1
            else:
                # Insertion (in prediction)
                operations.append((-1, j-1, 'insert'))
                j -= 1
        
        operations.reverse()
        
        # Extract aligned indices
        pred_aligned = [op[0] for op in operations if op[0] >= 0]
        ref_aligned = [op[1] for op in operations if op[1] >= 0]
        
        return pred_aligned, ref_aligned, operations
    
    def calculate(
        self,
        prediction: str,
        reference: str,
        return_details: bool = False,
    ) -> Union[float, Tuple[float, Dict]]:
        """
        Calculate ESA-CER for a single prediction-reference pair.
        
        This is the core computation implementing the formula from Section 5.1.
        
        Algorithm:
        1. Compute character-level alignment (edit distance)
        2. Identify erroneous positions (substitutions, insertions, deletions)
        3. Get weights for reference characters
        4. Compute weighted sum of errors / weighted sum of all characters
        
        Args:
            prediction: Predicted transcription
            reference: Ground truth transcription
            return_details: Whether to return detailed breakdown
        
        Returns:
            ESA-CER score (and optionally details dict)
        """
        # Tokenize and get math region flags
        ref_tokens = self.tokenize_with_math_regions(reference)
        
        # Compute reference weights
        ref_weights = []
        for char, in_math in ref_tokens:
            if in_math or self.is_math_token(char, in_math):
                ref_weights.append(self.alpha)
            else:
                ref_weights.append(1.0)
        
        # Get alignment
        _, ref_aligned, operations = self.calculate_alignment(prediction, reference)
        
        # Compute weighted error
        weighted_error = 0.0
        total_weight = sum(ref_weights)
        
        # Track errors in reference positions
        ref_error_positions = set()
        
        for pred_idx, ref_idx, op_type in operations:
            if op_type == 'subst':
                # Substitution affects reference position
                ref_error_positions.add(ref_idx)
            elif op_type == 'delete':
                # Deletion means missing reference character
                ref_error_positions.add(ref_idx)
            elif op_type == 'insert':
                # Insertion adds extra character (affects next reference pos)
                # We count this as affecting the closest reference position
                if ref_idx < len(ref_weights):
                    ref_error_positions.add(ref_idx)
        
        # Sum weighted errors
        for i, weight in enumerate(ref_weights):
            if i in ref_error_positions:
                weighted_error += weight
        
        if total_weight == 0:
            esa_cer = 0.0 if len(prediction) == 0 else 1.0
        else:
            esa_cer = weighted_error / total_weight
        
        if return_details:
            # Compute unweighted CER for comparison
            pred_chars = list(prediction)
            ref_chars = list(reference)
            unweighted_distance = editdistance.eval(pred_chars, ref_chars)
            unweighted_cer = unweighted_distance / len(ref_chars) if ref_chars else 0.0
            
            details = {
                'esa_cer': esa_cer,
                'unweighted_cer': unweighted_cer,
                'amplification_factor': esa_cer / unweighted_cer if unweighted_cer > 0 else 1.0,
                'total_chars': len(ref_chars),
                'math_tokens': sum(1 for _, in_math in ref_tokens if in_math),
                'normal_tokens': sum(1 for _, in_math in ref_tokens if not in_math),
                'total_weight': total_weight,
                'weighted_errors': weighted_error,
                'alpha': self.alpha,
            }
            return esa_cer, details
        
        return esa_cer
    
    def calculate_batch(
        self,
        predictions: List[str],
        references: List[str],
        return_details: bool = False,
    ) -> Union[float, Tuple[float, List[Dict]]]:
        """
        Calculate ESA-CER for a batch of predictions.
        
        Args:
            predictions: List of predicted transcriptions
            references: List of ground truth transcriptions
            return_details: Whether to return per-sample details
        
        Returns:
            Average ESA-CER (and optionally per-sample details)
        """
        if len(predictions) != len(references):
            raise ValueError(
                f"Predictions ({len(predictions)}) and references ({len(references)}) "
                "must have the same length"
            )
        
        scores = []
        details_list = []
        
        for pred, ref in zip(predictions, references):
            if return_details:
                score, details = self.calculate(pred, ref, return_details=True)
                details_list.append(details)
            else:
                score = self.calculate(pred, ref)
            scores.append(score)
        
        avg_esa_cer = np.mean(scores) if scores else 0.0
        
        if return_details:
            return avg_esa_cer, details_list
        return avg_esa_cer


def calculate_esa_cer(
    predictions: List[str],
    references: List[str],
    alpha: float = 3.0,
) -> float:
    """
    Convenience function to calculate ESA-CER.
    
    Args:
        predictions: List of predicted strings
        references: List of reference strings
        alpha: Weight for mathematical tokens (default 3.0 from paper)
    
    Returns:
        Average ESA-CER across all samples
    
    Example:
        >>> predictions = ["计算 \\(1+1=2\\)"]
        >>> references = ["计算 \\(1+1=3\\)"]
        >>> calculate_esa_cer(predictions, references, alpha=3.0)
        # Higher than CER because error is in math region
    """
    metric = ESA_CER(alpha=alpha)
    return metric.calculate_batch(predictions, references)


def extract_math_expressions(text: str) -> List[str]:
    """
    Extract LaTeX math expressions from text.
    
    Args:
        text: Input text with LaTeX delimiters
    
    Returns:
        List of math expressions (without delimiters)
    """
    # Pattern for \( ... \)
    pattern = r'\\\((.*?)\\\)'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches


def calculate_math_expression_rate(
    predictions: List[str],
    references: List[str],
) -> float:
    """
    Calculate Math Expression Recognition Rate (ExpRate).
    
    ExpRate = fraction of expressions with zero edit distance
    This is a stricter metric than CER for mathematical content.
    
    Paper results (Section 7.3, Task 2):
        Models struggle more with full expression recognition than
        individual character recognition, highlighting the need for
        structural understanding.
    
    Args:
        predictions: List of predicted strings
        references: List of reference strings
    
    Returns:
        ExpRate as float between 0.0 and 1.0
    """
    correct = 0
    total = 0
    
    for pred, ref in zip(predictions, references):
        pred_math = extract_math_expressions(pred)
        ref_math = extract_math_expressions(ref)
        
        # Match expressions (simplified - assumes aligned order)
        for p, r in zip(pred_math, ref_math):
            if editdistance.eval(p, r) == 0:
                correct += 1
            total += 1
    
    if total == 0:
        return 0.0
    
    return correct / total


def calculate_esa_cer_by_subject(
    predictions: List[str],
    references: List[str],
    subjects: List[str],
    alpha: float = 3.0,
) -> Dict[str, float]:
    """
    Calculate ESA-CER grouped by subject.
    
    Shows how the exam-score-aware metric varies across subjects
    with different amounts of mathematical content.
    
    Paper results (Section 7.4):
        Mathematics: ESA-CER significantly higher than CER due to
        high proportion of grade-sensitive mathematical tokens.
    
    Args:
        predictions: List of predicted strings
        references: List of reference strings
        subjects: List of subject labels
        alpha: Weight for math tokens
    
    Returns:
        Dictionary mapping subject to ESA-CER
    """
    metric = ESA_CER(alpha=alpha)
    
    subject_scores = {}
    subject_weights = {}
    
    for pred, ref, subj in zip(predictions, references, subjects):
        if subj not in subject_scores:
            subject_scores[subj] = {'weighted_error': 0.0, 'total_weight': 0.0}
        
        # Calculate per-sample
        ref_tokens = metric.tokenize_with_math_regions(ref)
        ref_weights = [metric.alpha if in_math or metric.is_math_token(c, in_math) 
                      else 1.0 for c, in_math in ref_tokens]
        
        # Compute alignment and errors
        _, ref_aligned, operations = metric.calculate_alignment(pred, ref)
        
        ref_error_positions = set()
        for pred_idx, ref_idx, op_type in operations:
            if op_type in ['subst', 'delete']:
                if ref_idx >= 0:
                    ref_error_positions.add(ref_idx)
            elif op_type == 'insert':
                if ref_idx < len(ref_weights):
                    ref_error_positions.add(ref_idx)
        
        # Accumulate weighted errors
        for i, weight in enumerate(ref_weights):
            subject_scores[subj]['total_weight'] += weight
            if i in ref_error_positions:
                subject_scores[subj]['weighted_error'] += weight
    
    # Compute per-subject ESA-CER
    result = {}
    for subj, stats in subject_scores.items():
        if stats['total_weight'] > 0:
            result[subj] = stats['weighted_error'] / stats['total_weight']
        else:
            result[subj] = 0.0
    
    return result
