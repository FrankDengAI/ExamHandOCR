#!/usr/bin/env python3
"""
Script to prepare annotation files from raw annotation data.
Converts professional transcriptions to training-ready format.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
from collections import defaultdict


def load_raw_annotations(annotation_dir: str) -> List[Dict]:
    """
    Load raw annotation files.
    
    Expected format:
    - JSON files with transcription, polygons, metadata
    """
    annotation_dir = Path(annotation_dir)
    annotations = []
    
    for ann_file in annotation_dir.glob('*.json'):
        with open(ann_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            annotations.append(data)
    
    return annotations


def stratified_split(
    annotations: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Dict[str, List[Dict]]:
    """
    Create stratified train/val/test split.
    
    Stratification by:
    - Examination type (Gaokao, Zhongkao, Mock, Joint, Standardized)
    - Subject (Chinese, Math, English, Physics, Chemistry, History, Geography, Biology)
    - Quality tier (High, Medium, Low)
    """
    # Group by strata
    strata = defaultdict(list)
    
    for ann in annotations:
        exam_type = ann.get('exam_type', 'Unknown')
        subject = ann.get('subject', 'Unknown')
        quality = ann.get('quality_tier', 'Medium')
        
        key = (exam_type, subject, quality)
        strata[key].append(ann)
    
    # Split each stratum
    splits = {'train-sup': [], 'val': [], 'test-sup': []}
    
    for stratum_key, stratum_anns in strata.items():
        n = len(stratum_anns)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        # Shuffle
        import random
        random.shuffle(stratum_anns)
        
        # Split
        splits['train-sup'].extend(stratum_anns[:n_train])
        splits['val'].extend(stratum_anns[n_train:n_train+n_val])
        splits['test-sup'].extend(stratum_anns[n_train+n_val:])
    
    return splits


def create_splits_by_session(
    annotations: List[Dict],
) -> Dict[str, List[Dict]]:
    """
    Create splits at batch-session level to prevent leakage.
    
    All images from a given examination session go to the same split.
    """
    # Group by batch session
    sessions = defaultdict(list)
    for ann in annotations:
        batch_id = ann.get('batch_id', 'Unknown')
        sessions[batch_id].append(ann)
    
    # Randomly assign sessions to splits
    import random
    session_ids = list(sessions.keys())
    random.shuffle(session_ids)
    
    n = len(session_ids)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    
    train_sessions = set(session_ids[:n_train])
    val_sessions = set(session_ids[n_train:n_train+n_val])
    test_sessions = set(session_ids[n_train+n_val:])
    
    splits = {'train-sup': [], 'val': [], 'test-sup': []}
    
    for batch_id, anns in sessions.items():
        if batch_id in train_sessions:
            splits['train-sup'].extend(anns)
        elif batch_id in val_sessions:
            splits['val'].extend(anns)
        else:
            splits['test-sup'].extend(anns)
    
    return splits


def build_vocabulary(annotations: List[Dict], min_freq: int = 2) -> Dict:
    """
    Build vocabulary from training annotations.
    
    Args:
        annotations: List of annotation dictionaries
        min_freq: Minimum frequency for character inclusion
    
    Returns:
        Vocabulary dictionary
    """
    from collections import Counter
    
    char_counter = Counter()
    
    for ann in annotations:
        transcription = ann.get('transcription', '')
        char_counter.update(transcription)
    
    # Filter by frequency
    vocab = ['<pad>', '<sos>', '<eos>', '<unk>', '<math>', '</math>']
    
    for char, freq in char_counter.most_common():
        if freq >= min_freq:
            vocab.append(char)
    
    return {
        'vocab': vocab,
        'char_to_id': {c: i for i, c in enumerate(vocab)},
        'id_to_char': {i: c for i, c in enumerate(vocab)},
        'vocab_size': len(vocab),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing raw annotation files')
    parser.add_argument('--output_file', type=str, default='annotations.json',
                       help='Output annotation file')
    parser.add_argument('--split_method', type=str, default='session',
                       choices=['session', 'stratified'],
                       help='Method for creating splits')
    parser.add_argument('--build_vocab', action='store_true',
                       help='Build and save vocabulary')
    parser.add_argument('--vocab_file', type=str, default='vocab.json',
                       help='Output vocabulary file')
    
    args = parser.parse_args()
    
    # Load raw annotations
    print(f"Loading annotations from {args.input_dir}...")
    annotations = load_raw_annotations(args.input_dir)
    print(f"Loaded {len(annotations)} annotations")
    
    # Create splits
    print(f"Creating splits using {args.split_method} method...")
    if args.split_method == 'session':
        splits = create_splits_by_session(annotations)
    else:
        splits = stratified_split(annotations)
    
    # Assign splits to annotations
    for split_name, split_anns in splits.items():
        for ann in split_anns:
            ann['split'] = split_name
    
    # Save annotations
    print(f"Saving annotations to {args.output_file}...")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)
    
    # Build vocabulary
    if args.build_vocab:
        print("Building vocabulary...")
        train_anns = splits['train-sup']
        vocab = build_vocabulary(train_anns)
        
        with open(args.vocab_file, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        
        print(f"Vocabulary size: {vocab['vocab_size']}")
        print(f"Saved vocabulary to {args.vocab_file}")
    
    # Print statistics
    print("\nDataset Statistics:")
    for split_name, split_anns in splits.items():
        print(f"  {split_name}: {len(split_anns)} samples")
        
        # Count by subject
        subjects = defaultdict(int)
        for ann in split_anns:
            subjects[ann.get('subject', 'Unknown')] += 1
        
        for subj, count in sorted(subjects.items()):
            print(f"    - {subj}: {count}")
    
    print("\nAnnotation preparation completed!")


if __name__ == '__main__':
    main()
