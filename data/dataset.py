"""
Dataset classes for ExamHandOCR benchmark.
Implements PyTorch Dataset classes for loading annotated and unannotated data.
Supports hierarchical data organization: batch_session -> examinee_id -> response_region
"""

import os
import json
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2

from .transforms import get_train_transforms, get_val_transforms


class ExamHandOCRDataset(Dataset):
    """
    Main dataset class for annotated ExamHandOCR data.
    
    This dataset loads the 8,640 professionally annotated images from the benchmark
    subset, with stratified sampling across 8 subjects, 5 exam types, and 3 quality tiers.
    
    Data Organization (Section 3.4):
        ExamHandOCR/
        ├── <BATCH_ID>/                   # e.g., "2024GKZH001_MATH"
        │   ├── <EXAMINEE_PSID>/          # 16-digit HMAC-SHA256 pseudonym
        │   │   ├── 01.jpg                # Response region 1
        │   │   └── ...                   # Typically 8 regions per examinee
    
    Annotation Protocol (Section 3.5):
        - 48 certified examination teachers (教师资格证持有者)
        - Dual annotation with senior adjudication
        - 99.2% inter-annotator agreement (character-level)
        - Full Unicode/LaTeX transcriptions
        - Pixel-level text-line polygon masks
    
    Args:
        data_root: Root directory containing the dataset
        split: Data split identifier ('train-sup', 'val', 'test-sup')
        annotation_file: Path to JSON file with annotations
        transform: Optional albumentations transform pipeline
        max_length: Maximum sequence length for text tokenization
        load_images: Whether to actually load images (False for metadata-only access)
        subjects: Optional filter for specific subjects (list of subject names)
        exam_types: Optional filter for specific exam types
    """
    
    # Valid subjects in the dataset (Section 3.1)
    VALID_SUBJECTS = [
        'Chinese', 'Mathematics', 'English', 'Physics', 
        'Chemistry', 'History', 'Geography', 'Biology'
    ]
    
    # Valid examination types (Section 3.1)
    VALID_EXAM_TYPES = [
        'Gaokao', 'Zhongkao', 'Mock', 'Joint', 'Standardized'
    ]
    
    def __init__(
        self,
        data_root: str,
        split: str,
        annotation_file: str,
        transform: Optional[Any] = None,
        max_length: int = 512,
        load_images: bool = True,
        subjects: Optional[List[str]] = None,
        exam_types: Optional[List[str]] = None,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.max_length = max_length
        self.load_images = load_images
        self.subjects_filter = subjects
        self.exam_types_filter = exam_types
        
        # Validate split name
        valid_splits = ['train-sup', 'val', 'test-sup', 'train-unsup', 'cross-A', 'cross-B']
        if split not in valid_splits:
            raise ValueError(f"Invalid split '{split}'. Must be one of {valid_splits}")
        
        # Load annotation file (Section 3.5)
        with open(annotation_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        
        # Filter by split and optional subject/exam_type filters
        self.samples = self._filter_samples()
        
        # Compute statistics
        self._compute_statistics()
        
        print(f"[ExamHandOCRDataset] Loaded {len(self.samples)} samples for split '{split}'")
        if subjects:
            print(f"  Filtered by subjects: {subjects}")
        if exam_types:
            print(f"  Filtered by exam types: {exam_types}")
    
    def _filter_samples(self) -> List[Dict]:
        """
        Filter annotations based on split and optional criteria.
        
        Returns:
            List of filtered sample dictionaries
        """
        filtered = []
        for ann in self.annotations:
            # Filter by split
            if ann.get('split', 'train-sup') != self.split:
                continue
            
            # Filter by subject if specified
            if self.subjects_filter:
                if ann.get('subject') not in self.subjects_filter:
                    continue
            
            # Filter by exam type if specified
            if self.exam_types_filter:
                if ann.get('exam_type') not in self.exam_types_filter:
                    continue
            
            filtered.append(ann)
        
        return filtered
    
    def _compute_statistics(self):
        """Compute and print dataset statistics."""
        self.stats = {
            'total': len(self.samples),
            'by_subject': {},
            'by_exam_type': {},
            'by_style': {},
            'by_quality': {},
        }
        
        for sample in self.samples:
            # Count by subject
            subj = sample.get('subject', 'Unknown')
            self.stats['by_subject'][subj] = self.stats['by_subject'].get(subj, 0) + 1
            
            # Count by exam type
            exam = sample.get('exam_type', 'Unknown')
            self.stats['by_exam_type'][exam] = self.stats['by_exam_type'].get(exam, 0) + 1
            
            # Count by handwriting style
            style = sample.get('handwriting_style', 'Unknown')
            self.stats['by_style'][style] = self.stats['by_style'].get(style, 0) + 1
            
            # Count by quality tier
            qual = sample.get('image_quality_tier', 'Medium')
            self.stats['by_quality'][qual] = self.stats['by_quality'].get(qual, 0) + 1
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Sample index
        
        Returns:
            Dictionary containing:
                - image: Transformed image tensor (if load_images=True)
                - image_path: Absolute path to image file
                - transcription: Ground truth text (Unicode with LaTeX math delimiters)
                - subject: Academic subject
                - exam_type: Examination type
                - handwriting_style: Handwriting style classification
                - batch_id: Examination batch identifier
                - examinee_psid: Pseudonymous examinee ID (HMAC-SHA256)
                - region_index: Response region index within answer sheet
                - image_quality_tier: Quality classification (High/Medium/Low)
                - oqs: Operational Quality Score if available
        """
        sample = self.samples[idx]
        
        # Construct image path following hierarchical organization (Section 3.4)
        # Format: <data_root>/<BATCH_ID>/<EXAMINEE_PSID>/<region_index>.jpg
        image_path = self.data_root / sample.get('image_path', '')
        
        # Load and transform image
        if self.load_images:
            image = self._load_image(image_path)
            
            # Apply albumentations transform pipeline
            if self.transform:
                transformed = self.transform(image=np.array(image))
                image = transformed['image']
            
            # Convert to torch tensor if not already done by transform
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image).unsqueeze(0).float()  # Add channel dim
        else:
            image = None
        
        return {
            'image': image,
            'image_path': str(image_path),
            'transcription': sample.get('transcription', ''),
            'subject': sample.get('subject', 'Unknown'),
            'exam_type': sample.get('exam_type', 'Unknown'),
            'handwriting_style': sample.get('handwriting_style', 'Unknown'),
            'batch_id': sample.get('batch_id', 'Unknown'),
            'examinee_psid': sample.get('examinee_psid', 'Unknown'),
            'region_index': sample.get('region_index', 0),
            'image_quality_tier': sample.get('image_quality_tier', 'Medium'),
            'oqs': sample.get('oqs', None),  # Operational Quality Score (Section 5.3)
        }
    
    def _load_image(self, image_path: Path) -> np.ndarray:
        """
        Load image from disk.
        
        Images are scanned at 300 DPI on A4 paper (Section 3.3):
        - Color mode: Grayscale / 8-bit
        - JPEG quality factor: 75-95 (platform-version dependent)
        
        Args:
            image_path: Path to image file
        
        Returns:
            Grayscale image as numpy array (H, W)
        
        Raises:
            ValueError: If image cannot be loaded
        """
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(
                f"Failed to load image: {image_path}. "
                f"File may be missing or corrupted."
            )
        return image
    
    def get_statistics(self) -> Dict[str, Any]:
        """Return dataset statistics."""
        return self.stats
    
    def print_statistics(self):
        """Print detailed dataset statistics."""
        print("\n" + "="*60)
        print(f"Dataset Statistics for Split '{self.split}'")
        print("="*60)
        print(f"Total samples: {self.stats['total']}")
        
        print("\nBy Subject:")
        for subj, count in sorted(self.stats['by_subject'].items()):
            pct = count / self.stats['total'] * 100
            print(f"  {subj:15s}: {count:5d} ({pct:5.1f}%)")
        
        print("\nBy Exam Type:")
        for exam, count in sorted(self.stats['by_exam_type'].items()):
            pct = count / self.stats['total'] * 100
            print(f"  {exam:15s}: {count:5d} ({pct:5.1f}%)")
        
        print("\nBy Handwriting Style:")
        for style, count in sorted(self.stats['by_style'].items()):
            pct = count / self.stats['total'] * 100
            print(f"  {style:15s}: {count:5d} ({pct:5.1f}%)")
        
        print("\nBy Quality Tier:")
        for qual, count in sorted(self.stats['by_quality'].items()):
            pct = count / self.stats['total'] * 100
            print(f"  {qual:15s}: {count:5d} ({pct:5.1f}%)")
        print("="*60 + "\n")


class SSLDataset(Dataset):
    """
    Dataset for semi-supervised learning (SSL) pre-training.
    
    This dataset provides access to the 3.15 million unannotated images
    from the train-unsup split, used for MAE pre-training (Section 7.1).
    
    The SSL paradigm is central to ExamHandOCR: with only 8,640 labeled
    images available, models must learn robust representations from the
    massive pool of unlabeled data.
    
    Paper configuration for SSL (Section 7.1):
        - Masked Autoencoding (MAE) objective
        - Patch size: 16
        - Mask ratio: 0.75
        - Pre-training epochs: 100
        - Learning rate: 1.5e-4 with 40-epoch warmup
    
    Args:
        data_root: Root directory containing unannotated images
        image_list: Optional specification of images to include
            - None: scan all JPEG files recursively
            - str ending with .txt: load paths from text file
            - str ending with .json: load from JSON file
            - List[str]: use provided list of paths
        transform: Albumentations transform pipeline (typically includes random masking)
        max_images: Maximum number of images to load (for debugging/development)
        pipeline_branch: Which pipeline branch ('Current' or 'Current_jst')
    """
    
    def __init__(
        self,
        data_root: str,
        image_list: Optional[Union[str, List[str]]] = None,
        transform: Optional[Any] = None,
        max_images: Optional[int] = None,
        pipeline_branch: str = 'Current',
    ):
        self.data_root = Path(data_root)
        self.transform = transform
        self.pipeline_branch = pipeline_branch
        
        # Load image paths based on input type
        self.image_paths = self._load_image_list(image_list)
        
        # Limit number of images if specified
        if max_images is not None:
            if max_images < len(self.image_paths):
                # Randomly sample for reproducibility
                import random
                random.seed(42)
                self.image_paths = random.sample(self.image_paths, max_images)
        
        # Verify that images exist
        self._validate_images()
        
        print(f"[SSLDataset] Loaded {len(self.image_paths)} unannotated images")
        print(f"  Pipeline branch: {pipeline_branch}")
        if max_images:
            print(f"  Limited to {max_images} images")
    
    def _load_image_list(self, image_list: Optional[Union[str, List[str]]]) -> List[str]:
        """
        Load list of image paths from various input formats.
        
        Args:
            image_list: Input specification (None, file path, or list)
        
        Returns:
            List of absolute image paths
        """
        if image_list is None:
            # Recursively scan for all JPEG files
            pattern = str(self.data_root / "**" / "*.jpg")
            paths = glob.glob(pattern, recursive=True)
            return sorted(paths)
        
        elif isinstance(image_list, str):
            if image_list.endswith('.txt'):
                # Load from text file (one path per line)
                with open(image_list, 'r', encoding='utf-8') as f:
                    paths = [line.strip() for line in f if line.strip()]
                return paths
            
            elif image_list.endswith('.json'):
                # Load from JSON file
                with open(image_list, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return [d.get('image_path', d) for d in data]
                    else:
                        raise ValueError("JSON file must contain a list of paths or objects")
            
            else:
                raise ValueError(f"Unsupported file format: {image_list}")
        
        elif isinstance(image_list, list):
            # Use provided list directly
            return image_list
        
        else:
            raise TypeError(f"image_list must be None, str, or List[str], got {type(image_list)}")
    
    def _validate_images(self):
        """Validate that image paths exist and are readable."""
        invalid_paths = []
        for path in self.image_paths[:100]:  # Check first 100 for efficiency
            if not Path(path).exists():
                invalid_paths.append(path)
        
        if invalid_paths:
            print(f"[Warning] {len(invalid_paths)} images not found (showing first 5):")
            for p in invalid_paths[:5]:
                print(f"  - {p}")
    
    def __len__(self) -> int:
        """Return the number of unannotated images."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single unannotated image.
        
        Args:
            idx: Image index
        
        Returns:
            Dictionary containing:
                - image: Image tensor (transformed)
                - image_path: Absolute path to image file
        """
        image_path = self.image_paths[idx]
        
        # Load image in grayscale (Section 3.3)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            # Handle corrupted/missing images gracefully
            # Return a different image from the dataset
            print(f"[Warning] Failed to load image: {image_path}")
            return self.__getitem__((idx + 1) % len(self))
        
        # Apply transforms (e.g., random masking for MAE)
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return {
            'image': image,
            'image_path': image_path,
        }


class LayoutDataset(Dataset):
    """
    Dataset for text-line segmentation (layout analysis) task.
    
    This dataset provides pixel-level polygon annotations for text-line
    segmentation, supporting both instance segmentation and semantic
    segmentation approaches (Section 7.5).
    
    Annotation format:
        - Polygon vertices: List of (x, y) coordinates
        - Each polygon represents one text line
        - Multiple lines per image (typically 1-8 lines)
    
    Supported models:
        - U-Net: Semantic segmentation baseline
        - Mask R-CNN: Instance segmentation
        - DETR: Transformer-based segmentation
    
    Args:
        data_root: Root directory of the dataset
        split: Data split ('train-sup', 'val', 'test-sup')
        annotation_file: Path to annotation JSON with polygon masks
        transform: Albumentations transform (supports 'masks' additional target)
    """
    
    def __init__(
        self,
        data_root: str,
        split: str,
        annotation_file: str,
        transform: Optional[Any] = None,
    ):
        self.data_root = Path(data_root)
        self.transform = transform
        
        # Load annotations with polygon masks (Section 3.5)
        with open(annotation_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        
        # Filter to samples with mask annotations in the specified split
        self.samples = [
            ann for ann in self.annotations
            if ann.get('split', 'train-sup') == split
            and 'text_line_masks' in ann  # Must have polygon annotations
            and len(ann.get('text_line_masks', [])) > 0
        ]
        
        # Compute statistics
        num_lines = [len(s.get('text_line_masks', [])) for s in self.samples]
        self.line_stats = {
            'mean': np.mean(num_lines) if num_lines else 0,
            'std': np.std(num_lines) if num_lines else 0,
            'min': np.min(num_lines) if num_lines else 0,
            'max': np.max(num_lines) if num_lines else 0,
        }
        
        print(f"[LayoutDataset] Loaded {len(self.samples)} samples for split '{split}'")
        print(f"  Avg text lines per image: {self.line_stats['mean']:.2f}")
    
    def __len__(self) -> int:
        """Return the number of samples with layout annotations."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample with layout annotations.
        
        Args:
            idx: Sample index
        
        Returns:
            Dictionary containing:
                - image: Image tensor
                - masks: Binary masks (N, H, W) where N is number of text lines
                - image_path: Path to image file
                - num_lines: Number of text lines in image
        """
        sample = self.samples[idx]
        
        # Load image
        image_path = self.data_root / sample.get('image_path', '')
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Load polygon masks and convert to binary masks
        polygon_data = sample.get('text_line_masks', [])
        masks = self._polygons_to_masks(polygon_data, image.shape)
        
        # Apply transforms
        if self.transform:
            # Albumentations supports additional targets like 'masks'
            transformed = self.transform(image=image, masks=masks)
            image = transformed['image']
            masks = transformed['masks']
        
        return {
            'image': image,
            'masks': masks,
            'image_path': str(image_path),
            'num_lines': len(polygon_data),
        }
    
    def _polygons_to_masks(
        self,
        polygon_data: List[List[Tuple[int, int]]],
        image_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Convert polygon annotations to binary masks.
        
        Args:
            polygon_data: List of polygons, each polygon is a list of (x, y) tuples
            image_shape: (height, width) of the image
        
        Returns:
            Binary masks as numpy array (num_lines, height, width)
        """
        h, w = image_shape[:2]
        masks = []
        
        for polygon in polygon_data:
            # Create empty mask
            mask = np.zeros((h, w), dtype=np.uint8)
            
            # Convert polygon to numpy array
            points = np.array(polygon, dtype=np.int32)
            
            # Fill polygon
            cv2.fillPoly(mask, [points], 1)
            
            masks.append(mask)
        
        # Return stacked masks or single empty mask if no lines
        if masks:
            return np.stack(masks)
        else:
            return np.zeros((1, h, w), dtype=np.uint8)


class StyleClassificationDataset(Dataset):
    """
    Dataset for handwriting style classification (Task 4, Section 6).
    
    Classification into 5 style categories:
        1. Regular (楷书): Standard, clearly written characters
        2. Running (行书): Semi-cursive, common in time-pressured exams
        3. Print-like: Handwriting mimicking printed text
        4. Cursive (草书): Highly abbreviated, difficult to read
        5. Mixed: Combination of styles within same response
    
    Style distribution in annotated subset (Section 4.5):
        - Running: 42.0% (most common due to time pressure)
        - Regular: 33.0%
        - Print-like: 17.0%
        - Cursive/Mixed: 8.0% (disproportionately high CER)
    
    Args:
        data_root: Root directory of the dataset
        split: Data split
        annotation_file: Path to annotation JSON with style labels
        transform: Albumentations transform for image preprocessing
    """
    
    # Style category mapping (Section 4.5, Table 7)
    STYLE_MAP = {
        'Regular': 0,        # Kaishu (楷书) - standard script
        'Running': 1,        # Xingshu (行书) - semi-cursive, most common
        'Print-like': 2,     # Print-style handwriting
        'Cursive': 3,        # Caoshu (草书) - highly abbreviated
        'Mixed': 4,          # Combination of styles
    }
    
    # Inverse mapping for decoding
    ID_TO_STYLE = {v: k for k, v in STYLE_MAP.items()}
    
    def __init__(
        self,
        data_root: str,
        split: str,
        annotation_file: str,
        transform: Optional[Any] = None,
    ):
        self.data_root = Path(data_root)
        self.transform = transform
        
        # Load annotations
        with open(annotation_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        
        # Filter to samples with style annotations
        self.samples = [
            ann for ann in self.annotations
            if ann.get('split', 'train-sup') == split
            and 'handwriting_style' in ann
            and ann['handwriting_style'] in self.STYLE_MAP
        ]
        
        # Compute style distribution
        self.style_counts = {}
        for sample in self.samples:
            style = sample['handwriting_style']
            self.style_counts[style] = self.style_counts.get(style, 0) + 1
        
        print(f"[StyleClassificationDataset] Loaded {len(self.samples)} samples")
        
        # Print style distribution
        print("  Style distribution:")
        for style, count in sorted(self.style_counts.items()):
            pct = count / len(self.samples) * 100
            style_id = self.STYLE_MAP[style]
            print(f"    {style:12s} (ID={style_id}): {count:5d} ({pct:5.1f}%)")
    
    def __len__(self) -> int:
        """Return the number of samples with style annotations."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample with style label.
        
        Args:
            idx: Sample index
        
        Returns:
            Dictionary containing:
                - image: Image tensor
                - label: Style class label (0-4)
                - style_name: Style name string
                - image_path: Path to image file
        """
        sample = self.samples[idx]
        
        # Load image
        image_path = self.data_root / sample.get('image_path', '')
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Get style label
        style_name = sample['handwriting_style']
        style_label = self.STYLE_MAP.get(style_name, 1)  # Default to Running (most common)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=np.array(image))
            image = transformed['image']
        
        return {
            'image': image,
            'label': style_label,
            'style_name': style_name,
            'image_path': str(image_path),
        }
    
    def get_class_weights(self) -> np.ndarray:
        """
        Compute class weights for balanced loss.
        
        Returns:
            Array of weights inversely proportional to class frequency
        """
        counts = np.array([self.style_counts.get(self.ID_TO_STYLE[i], 0) 
                          for i in range(len(self.STYLE_MAP))])
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * len(weights)
        return weights
