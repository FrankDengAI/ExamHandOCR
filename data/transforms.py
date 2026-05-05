"""
Data augmentation and preprocessing transforms for ExamHandOCR benchmark.

Implements the augmentation strategies described in Section 7.2 of the paper:
"Data augmentation (OCR): random skew +/-3 degrees, random JPEG re-compression 
(quality 70-100), brightness/contrast jitter (+/-0.15), elastic distortion (sigma=4, alpha=36)."

These augmentations simulate real-world variations encountered in examination 
scanning pipelines across different provinces and hardware generations.
"""

import numpy as np
import cv2
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform


class RandomSkew(ImageOnlyTransform):
    """
    Random skew/shear transformation.
    
    Simulates slight misalignment during the scanning process, which is common
    in high-throughput scanning centers handling millions of answer sheets.
    
    Paper specification (Section 7.2):
        - Skew range: +/-3 degrees
        - Probability: 0.5
    
    Args:
        skew_limit: Maximum skew angle in degrees (default: 3)
        always_apply: Whether to always apply this transform
        p: Probability of applying the transform
    """
    
    def __init__(self, skew_limit: int = 3, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        self.skew_limit = skew_limit
    
    def apply(self, img: np.ndarray, skew_angle: float = 0, **params) -> np.ndarray:
        """
        Apply skew transformation to image.
        
        Args:
            img: Input grayscale image (H, W)
            skew_angle: Rotation angle in degrees
        
        Returns:
            Skewed image with white (255) padding
        """
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
        
        # Apply affine transformation with white border
        return cv2.warpAffine(
            img, M, (w, h), 
            borderMode=cv2.BORDER_CONSTANT, 
            borderValue=255  # White background for answer sheets
        )
    
    def get_params(self) -> dict:
        """Generate random parameters for this transform."""
        return {
            'skew_angle': np.random.uniform(-self.skew_limit, self.skew_limit)
        }


class RandomJPEGCompression(ImageOnlyTransform):
    """
    Random JPEG re-compression to simulate quality variations.
    
    ExamHandOCR images exhibit varying JPEG quality factors (75-95) depending on
    platform version and scanner configuration (Section 3.3). This augmentation
    simulates the compression artifacts encountered in real operational pipelines.
    
    Paper specification (Section 7.2):
        - Quality range: 70-100
        - Probability: 0.5
    
    Args:
        quality_lower: Minimum JPEG quality (default: 70)
        quality_upper: Maximum JPEG quality (default: 100)
        always_apply: Whether to always apply
        p: Probability of application
    """
    
    def __init__(
        self, 
        quality_lower: int = 70, 
        quality_upper: int = 100, 
        always_apply: bool = False, 
        p: float = 0.5
    ):
        super().__init__(always_apply, p)
        self.quality_lower = quality_lower
        self.quality_upper = quality_upper
    
    def apply(self, img: np.ndarray, quality: int = 95, **params) -> np.ndarray:
        """
        Apply JPEG compression and decompression.
        
        Args:
            img: Input image
            quality: JPEG quality factor (0-100)
        
        Returns:
            Image with JPEG compression artifacts
        """
        # Encode with specified quality
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded_img = cv2.imencode('.jpg', img, encode_param)
        
        # Decode back to image
        decoded_img = cv2.imdecode(encoded_img, cv2.IMREAD_GRAYSCALE)
        
        return decoded_img
    
    def get_params(self) -> dict:
        """Generate random JPEG quality parameter."""
        return {
            'quality': np.random.randint(self.quality_lower, self.quality_upper + 1)
        }


class ElasticDistortion(ImageOnlyTransform):
    """
    Elastic deformation/distortion augmentation.
    
    Simulates physical distortions from paper handling and scanning variations.
    Based on the grid distortion method used in handwritten text recognition.
    
    Paper specification (Section 7.2):
        - Sigma (Gaussian kernel size): 4
        - Alpha (distortion magnitude): 36
        - Probability: 0.3
    
    Implementation follows Simard et al. "Best Practices for Convolutional Neural 
    Networks Applied to Visual Document Analysis" with parameters tuned for 
    examination handwriting recognition.
    
    Args:
        sigma: Standard deviation for Gaussian smoothing of displacement fields
        alpha: Scaling factor for displacement magnitude
        always_apply: Whether to always apply
        p: Probability of application
    """
    
    def __init__(
        self, 
        sigma: int = 4, 
        alpha: int = 36, 
        always_apply: bool = False, 
        p: float = 0.3
    ):
        super().__init__(always_apply, p)
        self.sigma = sigma
        self.alpha = alpha
    
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """
        Apply elastic distortion to image.
        
        Args:
            img: Input image (H, W)
        
        Returns:
            Distorted image
        """
        h, w = img.shape[:2]
        
        # Generate random displacement fields
        dx = np.random.rand(h, w) * 2 - 1  # Range [-1, 1]
        dy = np.random.rand(h, w) * 2 - 1
        
        # Smooth displacement fields with Gaussian filter
        dx = cv2.GaussianBlur(dx, (0, 0), sigmaX=self.sigma) * self.alpha
        dy = cv2.GaussianBlur(dy, (0, 0), sigmaX=self.sigma) * self.alpha
        
        # Create meshgrid of original coordinates
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Compute new coordinates
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        
        # Remap image
        return cv2.remap(
            img, map_x, map_y, 
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255
        )


class RandomHandwritingDegradation(ImageOnlyTransform):
    """
    Simulate handwriting degradation effects (fatigue, time pressure).
    
    Models the intra-image temporal nonstationarity described in Section 8.1:
    "As the student progresses through the paper, fatigue increases stroke 
    variability, reduces inter-character spacing, and causes letterforms to 
    blur toward their most cursive shorthand versions."
    
    This augmentation is specific to examination handwriting and captures
    context-conditioned handwriting drift.
    
    Args:
        blur_prob: Probability of applying motion blur
        spacing_variation: Range for character spacing variation
        p: Probability of applying this transform
    """
    
    def __init__(
        self,
        blur_prob: float = 0.3,
        spacing_variation: float = 0.1,
        always_apply: bool = False,
        p: float = 0.3,
    ):
        super().__init__(always_apply, p)
        self.blur_prob = blur_prob
        self.spacing_variation = spacing_variation
    
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """Apply handwriting degradation effects."""
        result = img.copy()
        
        # Apply motion blur to simulate writing speed variation
        if np.random.rand() < self.blur_prob:
            kernel_size = np.random.choice([3, 5])
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[kernel_size // 2, :] = 1.0 / kernel_size  # Horizontal motion
            result = cv2.filter2D(result, -1, kernel)
        
        return result


class RandomBackgroundLines(ImageOnlyTransform):
    """
    Add random ruling lines to simulate answer sheet backgrounds.
    
    Many answer sheets have pre-printed ruling lines that can visually
    merge with handwritten strokes, creating a challenging segmentation
    scenario (Section 2, "operational scanning artifact diversity").
    
    Args:
        line_prob: Probability of having ruling lines
        max_lines: Maximum number of lines to add
        p: Probability of applying this transform
    """
    
    def __init__(
        self,
        line_prob: float = 0.5,
        max_lines: int = 3,
        always_apply: bool = False,
        p: float = 0.3,
    ):
        super().__init__(always_apply, p)
        self.line_prob = line_prob
        self.max_lines = max_lines
    
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """Add faint ruling lines to image."""
        h, w = img.shape[:2]
        result = img.copy()
        
        # Decide if we add lines
        if np.random.rand() < self.line_prob:
            num_lines = np.random.randint(1, self.max_lines + 1)
            
            for _ in range(num_lines):
                y = np.random.randint(0, h)
                # Draw faint horizontal line
                cv2.line(result, (0, y), (w, y), color=200, thickness=1)
        
        return result


def get_train_transforms(
    image_size: tuple = (384, 128),
    apply_augmentation: bool = True,
    include_exam_specific: bool = True,
):
    """
    Get training transform pipeline with augmentation.
    
    Implements the complete augmentation protocol from Section 7.2:
    1. Geometric: Random skew ±3°
    2. Quality: JPEG re-compression (quality 70-100)
    3. Photometric: Brightness/contrast jitter ±0.15
    4. Distortion: Elastic deformation (σ=4, α=36)
    
    Also includes exam-specific augmentations modeling operational artifacts.
    
    Args:
        image_size: Target size (width, height). Default (384, 128) as per paper.
        apply_augmentation: Whether to apply data augmentation
        include_exam_specific: Include examination-specific augmentations
    
    Returns:
        Albumentations Compose object
    """
    transforms = []
    
    # Resize while preserving aspect ratio, then pad to target size
    # This maintains character proportions while standardizing input size
    transforms.append(A.LongestMaxSize(max_size=max(image_size)))
    transforms.append(A.PadIfNeeded(
        min_height=image_size[1],
        min_width=image_size[0],
        border_mode=cv2.BORDER_CONSTANT,
        value=255,  # White padding for answer sheets
    ))
    
    if apply_augmentation:
        # Geometric augmentations (Section 7.2)
        transforms.extend([
            RandomSkew(skew_limit=3, p=0.5),
        ])
        
        # Quality and compression artifacts (Section 7.2)
        transforms.extend([
            RandomJPEGCompression(quality_lower=70, quality_upper=100, p=0.5),
        ])
        
        # Photometric augmentations (Section 7.2)
        transforms.extend([
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.5
            ),
        ])
        
        # Distortion augmentations (Section 7.2)
        transforms.extend([
            ElasticDistortion(sigma=4, alpha=36, p=0.3),
        ])
        
        # Examination-specific augmentations
        if include_exam_specific:
            transforms.extend([
                RandomHandwritingDegradation(p=0.2),
                RandomBackgroundLines(p=0.2),
            ])
    
    # Normalization to [-1, 1] range
    # This is standard for most deep learning models
    transforms.append(A.Normalize(
        mean=[0.5],
        std=[0.5],
        max_pixel_value=255.0
    ))
    
    return A.Compose(transforms)


def get_val_transforms(image_size: tuple = (384, 128)):
    """
    Get validation transform pipeline (no augmentation).
    
    Validation and test sets should not have random augmentation to ensure
    consistent evaluation. Only geometric resizing and normalization are applied.
    
    Args:
        image_size: Target size (width, height)
    
    Returns:
        Albumentations Compose object
    """
    return A.Compose([
        # Resize maintaining aspect ratio
        A.LongestMaxSize(max_size=max(image_size)),
        # Pad to target size
        A.PadIfNeeded(
            min_height=image_size[1],
            min_width=image_size[0],
            border_mode=cv2.BORDER_CONSTANT,
            value=255,
        ),
        # Normalize to [-1, 1]
        A.Normalize(
            mean=[0.5],
            std=[0.5],
            max_pixel_value=255.0
        ),
    ])


def get_layout_transforms(
    image_size: tuple = (1024, 1024),
    is_train: bool = True,
):
    """
    Get transforms for layout analysis (text-line segmentation).
    
    Layout models process full-resolution images to preserve fine text-line
    boundaries. Paper specifies max 1024 px on long edge (Section 7.2).
    
    Args:
        image_size: Target size for layout models
        is_train: Whether training (with augmentation) or validation
    
    Returns:
        Albumentations Compose object with mask support
    """
    transforms = []
    
    if is_train:
        # Geometric augmentations suitable for layout
        # Note: We avoid strong skew as it affects line detection
        transforms.extend([
            A.RandomScale(scale_limit=0.2, p=0.5),
            A.HorizontalFlip(p=0.3),  # Horizontal flip is safe for text lines
            A.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
        ])
    
    # Resize to maximum dimension
    transforms.append(A.LongestMaxSize(max_size=max(image_size)))
    transforms.append(A.PadIfNeeded(
        min_height=image_size[1],
        min_width=image_size[0],
        border_mode=cv2.BORDER_CONSTANT,
        value=255,
    ))
    
    # Normalization
    transforms.append(A.Normalize(
        mean=[0.5],
        std=[0.5],
        max_pixel_value=255.0
    ))
    
    # Additional targets for masks (used in layout tasks)
    return A.Compose(transforms, additional_targets={'masks': 'masks'})


def get_ssl_transforms(
    image_size: tuple = (384, 128),
    mask_ratio: float = 0.75,
    patch_size: int = 16,
):
    """
    Get transforms for self-supervised learning (MAE pre-training).
    
    For MAE pre-training (Section 7.1), we need minimal augmentation as
    the masking itself provides the regularization. Standard preprocessing
    is applied.
    
    Paper configuration:
        - Patch size: 16
        - Mask ratio: 0.75
        - Pre-training epochs: 100
    
    Args:
        image_size: Target size
        mask_ratio: Ratio of patches to mask (not used in transforms, but documented)
        patch_size: Size of patches for masking
    
    Returns:
        Albumentations Compose object
    """
    return A.Compose([
        # Standard resizing
        A.LongestMaxSize(max_size=max(image_size)),
        A.PadIfNeeded(
            min_height=image_size[1],
            min_width=image_size[0],
            border_mode=cv2.BORDER_CONSTANT,
            value=255,
        ),
        # Normalization
        A.Normalize(
            mean=[0.5],
            std=[0.5],
            max_pixel_value=255.0
        ),
    ])


def get_style_transforms(
    image_size: tuple = (224, 224),
    is_train: bool = True,
):
    """
    Get transforms for handwriting style classification.
    
    Style classification uses square inputs (224x224) as it focuses on
    stroke patterns and overall writing characteristics rather than
    sequential text recognition.
    
    Args:
        image_size: Target size for style classifier
        is_train: Whether training with augmentation
    
    Returns:
        Albumentations Compose object
    """
    transforms = [
        # Resize to square
        A.Resize(height=image_size[1], width=image_size[0]),
    ]
    
    if is_train:
        # Moderate augmentation for style invariance
        transforms.extend([
            A.RandomRotate90(p=0.5),  # Rotation can help with slant invariance
            A.Flip(p=0.5),  # Flip augmentation
            A.RandomBrightnessContrast(p=0.3),
        ])
    
    transforms.append(A.Normalize(
        mean=[0.5],
        std=[0.5],
        max_pixel_value=255.0
    ))
    
    return A.Compose(transforms)


def get_test_time_transforms(image_size: tuple = (384, 128)):
    """
    Get test-time transforms including optional test-time augmentation (TTA).
    
    Test-Time Augmentation can improve robustness by averaging predictions
    across multiple augmented versions of the same input.
    
    Args:
        image_size: Target size
    
    Returns:
        List of Albumentations Compose objects for TTA
    """
    # Base transform
    base = A.Compose([
        A.LongestMaxSize(max_size=max(image_size)),
        A.PadIfNeeded(
            min_height=image_size[1],
            min_width=image_size[0],
            border_mode=cv2.BORDER_CONSTANT,
            value=255,
        ),
        A.Normalize(mean=[0.5], std=[0.5], max_pixel_value=255.0),
    ])
    
    # Mild augmentations for TTA
    tta_transforms = [
        base,  # Original
        A.Compose([
            A.LongestMaxSize(max_size=max(image_size)),
            A.PadIfNeeded(min_height=image_size[1], min_width=image_size[0], 
                         border_mode=cv2.BORDER_CONSTANT, value=255),
            RandomSkew(skew_limit=1, p=1.0),  # Very mild skew
            A.Normalize(mean=[0.5], std=[0.5], max_pixel_value=255.0),
        ]),
        A.Compose([
            A.LongestMaxSize(max_size=max(image_size)),
            A.PadIfNeeded(min_height=image_size[1], min_width=image_size[0],
                         border_mode=cv2.BORDER_CONSTANT, value=255),
            A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=1.0),
            A.Normalize(mean=[0.5], std=[0.5], max_pixel_value=255.0),
        ]),
    ]
    
    return tta_transforms
