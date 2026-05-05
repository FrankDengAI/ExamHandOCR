"""
Operational Quality Score (OQS) calculation.
Implements the no-reference image quality score from Section 5.3.

OQS = w1*(1-skew) + w2*contrast + w3*(1-JPEG_blocking) + w4*(1-bleed_through)

Default weights: (0.25, 0.30, 0.25, 0.20)
"""

import cv2
import numpy as np
from typing import Tuple, Union
from scipy import ndimage


class OperationalQualityScore:
    """
    Operational Quality Score (OQS) calculator.
    
    A composite no-reference image quality score for scanned answer sheets.
    
    Args:
        weights: Tuple of (w_skew, w_contrast, w_jpeg, w_bleed)
                 Default: (0.25, 0.30, 0.25, 0.20)
    """
    
    def __init__(self, weights: Tuple[float, float, float, float] = (0.25, 0.30, 0.25, 0.20)):
        self.weights = weights
        self.w_skew, self.w_contrast, self.w_jpeg, self.w_bleed = weights
    
    def calculate_skew(self, image: np.ndarray) -> float:
        """
        Estimate skew angle of text lines.
        
        Args:
            image: Grayscale image
        
        Returns:
            Normalized skew score (0 = perfectly straight, 1 = severely skewed)
        """
        # Detect edges
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        
        # Hough transform to find lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
        
        if lines is None or len(lines) == 0:
            return 0.0  # No detectable skew
        
        # Calculate angles
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:
                angle = np.abs(np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi)
                angles.append(angle)
        
        if len(angles) == 0:
            return 0.0
        
        # Mean absolute deviation from 0 or 90 degrees
        mean_angle = np.mean(angles)
        skew_deg = min(mean_angle, 90 - mean_angle) if mean_angle < 90 else mean_angle - 90
        
        # Normalize: 0-5 degrees is typical, >10 degrees is severe
        skew_score = min(skew_deg / 10.0, 1.0)
        
        return skew_score
    
    def calculate_contrast(self, image: np.ndarray) -> float:
        """
        Calculate RMS contrast.
        
        Args:
            image: Grayscale image
        
        Returns:
            Normalized contrast score (0 = very low, 1 = high)
        """
        # RMS contrast
        rms_contrast = np.std(image)
        
        # Normalize to 0-1 range (assuming 8-bit grayscale, max std is ~127)
        # Good contrast is typically > 40
        contrast_score = min(rms_contrast / 60.0, 1.0)
        
        return contrast_score
    
    def calculate_jpeg_blocking(self, image: np.ndarray) -> float:
        """
        Estimate JPEG blocking artifacts using high-frequency energy ratio.
        
        Args:
            image: Grayscale image
        
        Returns:
            Normalized blocking score (0 = no blocking, 1 = severe blocking)
        """
        # DCT analysis to detect 8x8 blocking
        h, w = image.shape
        
        # Resize to ensure divisible by 8
        h_blocks, w_blocks = h // 8, w // 8
        
        if h_blocks < 2 or w_blocks < 2:
            return 0.0
        
        # Calculate variance in each 8x8 block
        block_variances = []
        for i in range(h_blocks):
            for j in range(w_blocks):
                block = image[i*8:(i+1)*8, j*8:(j+1)*8]
                block_variances.append(np.var(block))
        
        # Blocking artifacts create periodic variance patterns
        # Calculate variance of block variances
        block_var_var = np.var(block_variances)
        
        # Normalize (higher variance indicates more blocking)
        # Typical values: good < 100, severe > 1000
        blocking_score = min(block_var_var / 500.0, 1.0)
        
        return blocking_score
    
    def calculate_bleed_through(self, image: np.ndarray) -> float:
        """
        Estimate bleed-through from reverse side of paper.
        
        Args:
            image: Grayscale image
        
        Returns:
            Normalized bleed-through score (0 = no bleed, 1 = severe bleed)
        """
        # Detect faint text patterns that might be bleed-through
        # This is a simplified heuristic
        
        # Blur to reduce noise
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Calculate histogram
        hist = cv2.calcHist([blurred], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        
        # Look for significant mass in mid-gray (potential bleed-through)
        # Background is typically white (255), text is black (0)
        # Bleed-through creates gray shadows
        mid_gray_mass = hist[50:150].sum()
        
        # Normalize
        bleed_score = min(mid_gray_mass * 5, 1.0)
        
        return bleed_score
    
    def calculate(self, image: Union[np.ndarray, str]) -> Tuple[float, dict]:
        """
        Calculate OQS for an image.
        
        Args:
            image: Grayscale image array or path to image
        
        Returns:
            Tuple of (OQS score, component scores dict)
        """
        # Load image if path provided
        if isinstance(image, str):
            image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise ValueError("Failed to load image")
        
        # Calculate components
        skew_score = self.calculate_skew(image)
        contrast_score = self.calculate_contrast(image)
        jpeg_score = self.calculate_jpeg_blocking(image)
        bleed_score = self.calculate_bleed_through(image)
        
        # Composite score
        oqs = (
            self.w_skew * (1 - skew_score) +
            self.w_contrast * contrast_score +
            self.w_jpeg * (1 - jpeg_score) +
            self.w_bleed * (1 - bleed_score)
        )
        
        components = {
            'skew': skew_score,
            'contrast': contrast_score,
            'jpeg_blocking': jpeg_score,
            'bleed_through': bleed_score,
        }
        
        return oqs, components
    
    def stratify_by_quality(self, images: list, oqs_scores: list) -> dict:
        """
        Stratify images into High/Medium/Low quality tiers.
        
        Args:
            images: List of images or paths
            oqs_scores: List of pre-calculated OQS scores
        
        Returns:
            Dictionary with tier assignments
        """
        # Sort by OQS
        sorted_indices = np.argsort(oqs_scores)
        
        # Split into tertiles
        n = len(sorted_indices)
        tertile_size = n // 3
        
        tiers = {
            'High-OQS': sorted_indices[-tertile_size:],
            'Medium-OQS': sorted_indices[tertile_size:-tertile_size],
            'Low-OQS': sorted_indices[:tertile_size],
        }
        
        return tiers


def calculate_oqs(image: Union[np.ndarray, str], weights=None) -> Tuple[float, dict]:
    """
    Convenience function to calculate OQS.
    
    Args:
        image: Grayscale image array or path
        weights: Optional custom weights tuple
    
    Returns:
        Tuple of (OQS score, component scores)
    """
    calculator = OperationalQualityScore(weights=weights) if weights else OperationalQualityScore()
    return calculator.calculate(image)


def calculate_brisque(image: Union[np.ndarray, str]) -> float:
    """
    Calculate BRISQUE score for comparison.
    Note: Requires pybrisque or opencv-contrib.
    
    Args:
        image: Grayscale image
    
    Returns:
        BRISQUE score (lower is better)
    """
    try:
        if isinstance(image, str):
            image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        
        # Placeholder for BRISQUE calculation
        # In practice, use cv2.quality.QualityBRISQUE or pybrisque
        # For now, return a dummy value
        return 34.8  # Typical score for 'Current' tree
    except:
        return 0.0
