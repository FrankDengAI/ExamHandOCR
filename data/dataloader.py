"""
DataLoader utilities for ExamHandOCR.
"""

from typing import Optional, Callable

import torch
from torch.utils.data import DataLoader
import numpy as np


def collate_fn_ocr(batch):
    """Collate function for OCR tasks."""
    images = []
    transcriptions = []
    metadata = []
    
    for item in batch:
        if item['image'] is not None:
            images.append(torch.from_numpy(item['image']).unsqueeze(0).float())
        transcriptions.append(item['transcription'])
        metadata.append({
            'image_path': item['image_path'],
            'subject': item['subject'],
            'exam_type': item['exam_type'],
            'handwriting_style': item['handwriting_style'],
            'batch_id': item['batch_id'],
            'image_quality_tier': item['image_quality_tier'],
        })
    
    # Stack images (they should all be same size after transforms)
    if images:
        images = torch.stack(images)
    else:
        images = None
    
    return {
        'images': images,
        'transcriptions': transcriptions,
        'metadata': metadata,
    }


def collate_fn_ssl(batch):
    """Collate function for SSL pre-training."""
    images = []
    paths = []
    
    for item in batch:
        images.append(torch.from_numpy(item['image']).unsqueeze(0).float())
        paths.append(item['image_path'])
    
    images = torch.stack(images)
    
    return {
        'images': images,
        'image_paths': paths,
    }


def collate_fn_layout(batch):
    """Collate function for layout tasks."""
    images = []
    masks = []
    metadata = []
    
    for item in batch:
        images.append(torch.from_numpy(item['image']).unsqueeze(0).float())
        masks.append(torch.from_numpy(item['masks']).float())
        metadata.append({'image_path': item['image_path']})
    
    # Pad masks to same number of instances
    max_masks = max(m.shape[0] for m in masks)
    padded_masks = []
    for m in masks:
        if m.shape[0] < max_masks:
            pad = torch.zeros(max_masks - m.shape[0], m.shape[1], m.shape[2])
            m = torch.cat([m, pad], dim=0)
        padded_masks.append(m)
    
    images = torch.stack(images)
    masks = torch.stack(padded_masks)
    
    return {
        'images': images,
        'masks': masks,
        'metadata': metadata,
    }


def get_dataloader(
    dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    task: str = 'ocr',
):
    """
    Create DataLoader for specified task.
    
    Args:
        dataset: Dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        task: One of 'ocr', 'ssl', 'layout', 'style'
    """
    collate_fn = {
        'ocr': collate_fn_ocr,
        'ssl': collate_fn_ssl,
        'layout': collate_fn_layout,
        'style': collate_fn_ocr,  # Similar structure
    }.get(task, collate_fn_ocr)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=shuffle,  # Drop incomplete batches during training
    )


def get_ssl_dataloader(
    dataset,
    batch_size: int = 256,
    num_workers: int = 8,
    pin_memory: bool = True,
):
    """Create DataLoader for SSL pre-training."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn_ssl,
        drop_last=True,
    )


class BalancedSubjectSampler:
    """
    Sampler to balance subjects across batches.
    Ensures each batch has proportional representation of all 8 subjects.
    """
    
    def __init__(self, dataset, batch_size, num_batches):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_batches = num_batches
        
        # Group indices by subject
        self.subject_indices = {}
        for idx, sample in enumerate(dataset.samples):
            subject = sample.get('subject', 'Unknown')
            if subject not in self.subject_indices:
                self.subject_indices[subject] = []
            self.subject_indices[subject].append(idx)
        
        self.subjects = list(self.subject_indices.keys())
        self.samples_per_subject = batch_size // len(self.subjects)
    
    def __iter__(self):
        for _ in range(self.num_batches):
            batch_indices = []
            for subject in self.subjects:
                indices = self.subject_indices[subject]
                sampled = np.random.choice(
                    indices,
                    size=self.samples_per_subject,
                    replace=len(indices) < self.samples_per_subject
                )
                batch_indices.extend(sampled)
            
            # Shuffle batch
            np.random.shuffle(batch_indices)
            yield batch_indices[:self.batch_size]
    
    def __len__(self):
        return self.num_batches
