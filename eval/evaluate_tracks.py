"""
Evaluation scripts for the three benchmark tracks:
1. Semi-Supervised Long-Form OCR (SSL-OCR)
2. Cross-Session Generalization (CSG)
3. Operational-Fidelity Evaluation (OFE)
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from metrics import calculate_cer, calculate_esa_cer, calculate_gg
from metrics.oqs import OperationalQualityScore
from metrics.ri import calculate_ri


def evaluate_semi_supervised_track(
    model,
    labeled_loaders: Dict[str, torch.utils.data.DataLoader],
    tokenizer,
    labeled_sizes: List[int] = None,
    device='cuda',
) -> Dict:
    """
    Evaluate on Semi-Supervised Long-Form OCR Track (Track 1).
    
    Paper protocol:
    - Models may access all ~3.15M unannotated images during SSL pre-training
    - Supervised fine-tuning restricted to labeled data
    - Primary evaluation on test-sup
    - Annotation-efficiency sub-track: sweep labeled set size from 100 to 6,048
    
    Args:
        model: OCR model (pre-trained with SSL)
        labeled_loaders: Dict mapping split names to data loaders
        tokenizer: Tokenizer
        labeled_sizes: List of labeled set sizes for efficiency curve
        device: Device
    
    Returns:
        Dictionary with results for each labeled size
    """
    results = {
        'track': 'SSL-OCR',
        'annotation_efficiency_curve': {},
    }
    
    # Evaluate on full test set
    if 'test-sup' in labeled_loaders:
        print("Evaluating on test-sup...")
        test_results = evaluate_split(model, labeled_loaders['test-sup'], tokenizer, device)
        results['test_sup'] = test_results
    
    # Annotation-efficiency sub-track
    if labeled_sizes:
        for size in labeled_sizes:
            if f'train_{size}' in labeled_loaders:
                print(f"Evaluating with {size} labeled samples...")
                loader = labeled_loaders[f'train_{size}']
                train_results = evaluate_split(model, loader, tokenizer, device)
                results['annotation_efficiency_curve'][size] = train_results
    
    return results


def evaluate_cross_session_track(
    model,
    source_loader,
    target_loader,
    tokenizer,
    adaptation_loader=None,
    device='cuda',
) -> Dict:
    """
    Evaluate on Cross-Session Generalization Track (Track 2).
    
    Paper protocol:
    - Train on cross-A (Current tree), evaluate zero-shot on cross-B (Current_jst)
    - Key metric: Generalization Gap (GG) = |CER_source - CER_target|
    - Adaptation sub-track: permit 5% unlabeled target data for adaptation
    
    Args:
        model: OCR model
        source_loader: Source domain (e.g., Current) test loader
        target_loader: Target domain (e.g., Current_jst) test loader
        tokenizer: Tokenizer
        adaptation_loader: Optional unlabeled target data for adaptation
        device: Device
    
    Returns:
        Dictionary with CER_source, CER_target, GG
    """
    print("Evaluating Cross-Session Generalization...")
    
    # Zero-shot evaluation
    source_results = evaluate_split(model, source_loader, tokenizer, device)
    target_results = evaluate_split(model, target_loader, tokenizer, device)
    
    cer_source = source_results['CER']
    cer_target = target_results['CER']
    gg = calculate_gg(cer_source, cer_target)
    
    results = {
        'track': 'Cross-Session Generalization',
        'direction': 'Current -> Current_jst',
        'zero_shot': {
            'CER_source': cer_source,
            'CER_target': cer_target,
            'GG': gg,
            'ESA-CER_source': source_results.get('ESA-CER', cer_source),
            'ESA-CER_target': target_results.get('ESA-CER', cer_target),
        }
    }
    
    # Adaptation sub-track
    if adaptation_loader:
        print("Evaluating with adaptation...")
        # Perform adaptation (e.g., test-time training, batch norm adaptation)
        adapted_model = adapt_model(model, adaptation_loader, device)
        
        adapted_results = evaluate_split(adapted_model, target_loader, tokenizer, device)
        cer_adapted = adapted_results['CER']
        gg_adapted = calculate_gg(cer_source, cer_adapted)
        
        results['with_adaptation'] = {
            'CER_target': cer_adapted,
            'GG': gg_adapted,
            'improvement': cer_target - cer_adapted,
        }
    
    return results


def evaluate_operational_fidelity_track(
    model,
    test_loader,
    tokenizer,
    device='cuda',
) -> Dict:
    """
    Evaluate on Operational-Fidelity Evaluation Track (Track 3).
    
    Paper protocol:
    - Stratify test-sup by OQS into High/Medium/Low quality tertiles
    - Report CER, ESA-CER, and Robustness Index (RI) per tertile
    
    Args:
        model: OCR model
        test_loader: Test data loader
        tokenizer: Tokenizer
        device: Device
    
    Returns:
        Dictionary with per-tier and overall metrics
    """
    print("Evaluating Operational Fidelity...")
    
    oqs_calculator = OperationalQualityScore()
    
    # Collect predictions and OQS
    all_predictions = []
    all_references = []
    all_oqs_scores = []
    all_per_sample_cer = []
    
    model.eval()
    model = model.to(device)
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Collecting predictions and OQS'):
            images = batch['images'].to(device)
            transcriptions = batch['transcriptions']
            
            # Generate predictions
            if hasattr(model, 'generate'):
                generated = model.generate(images)
                if isinstance(generated, dict):
                    generated = generated['logits'].argmax(dim=-1)
                predictions = tokenizer.batch_decode(generated)
            elif hasattr(model, 'predict'):
                pred_ids = model.predict(images)
                predictions = tokenizer.batch_decode([torch.tensor(p) for p in pred_ids])
            else:
                logits = model(images)
                if isinstance(logits, dict):
                    logits = logits['logits']
                preds = logits.argmax(dim=-1)
                predictions = tokenizer.batch_decode(preds)
            
            # Calculate per-sample CER
            import editdistance
            for pred, ref in zip(predictions, transcriptions):
                pred_chars = list(pred)
                ref_chars = list(ref)
                cer = editdistance.eval(pred_chars, ref_chars) / len(ref_chars) if ref_chars else 0
                all_per_sample_cer.append(cer)
            
            all_predictions.extend(predictions)
            all_references.extend(transcriptions)
            
            # Calculate OQS for each image
            for i, img in enumerate(images):
                img_np = img.cpu().numpy()
                if img_np.ndim == 3:
                    img_np = img_np[0]
                if img_np.min() < 0:
                    img_np = (img_np + 1) / 2 * 255
                img_np = img_np.astype('uint8')
                
                oqs, components = oqs_calculator.calculate(img_np)
                all_oqs_scores.append(oqs)
    
    # Stratify by OQS
    sorted_indices = np.argsort(all_oqs_scores)
    n = len(sorted_indices)
    tertile_size = n // 3
    
    tier_indices = {
        'High-OQS': sorted_indices[-tertile_size:],
        'Medium-OQS': sorted_indices[tertile_size:-tertile_size],
        'Low-OQS': sorted_indices[:tertile_size],
    }
    
    results = {
        'track': 'Operational-Fidelity Evaluation',
        'overall': {
            'CER': calculate_cer(all_predictions, all_references),
            'ESA-CER': calculate_esa_cer(all_predictions, all_references),
        },
        'by_tier': {},
    }
    
    # Calculate per-tier metrics
    for tier, indices in tier_indices.items():
        tier_preds = [all_predictions[i] for i in indices]
        tier_refs = [all_references[i] for i in indices]
        tier_oqs = [all_oqs_scores[i] for i in indices]
        tier_cers = [all_per_sample_cer[i] for i in indices]
        
        results['by_tier'][tier] = {
            'CER': calculate_cer(tier_preds, tier_refs),
            'ESA-CER': calculate_esa_cer(tier_preds, tier_refs),
            'mean_OQS': np.mean(tier_oqs),
            'num_samples': len(indices),
        }
    
    # Calculate RI
    ri = calculate_ri(all_oqs_scores, all_per_sample_cer)
    results['overall']['RI'] = ri
    
    return results


def evaluate_split(model, loader, tokenizer, device='cuda') -> Dict:
    """Helper function to evaluate on a single split."""
    model.eval()
    model = model.to(device)
    
    all_preds = []
    all_refs = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating'):
            images = batch['images'].to(device)
            transcriptions = batch['transcriptions']
            
            # Generate
            if hasattr(model, 'generate'):
                generated = model.generate(images)
                if isinstance(generated, dict):
                    generated = generated['logits'].argmax(dim=-1)
                predictions = tokenizer.batch_decode(generated) if tokenizer else [''] * len(generated)
            elif hasattr(model, 'predict'):
                pred_ids = model.predict(images)
                predictions = tokenizer.batch_decode([torch.tensor(p) for p in pred_ids]) if tokenizer else [''] * len(pred_ids)
            else:
                logits = model(images)
                if isinstance(logits, dict):
                    logits = logits['logits']
                preds = logits.argmax(dim=-1)
                predictions = tokenizer.batch_decode(preds) if tokenizer else [''] * len(preds)
            
            all_preds.extend(predictions)
            all_refs.extend(transcriptions)
    
    cer = calculate_cer(all_preds, all_refs)
    esa_cer = calculate_esa_cer(all_preds, all_refs)
    
    return {
        'CER': cer,
        'ESA-CER': esa_cer,
        'num_samples': len(all_preds),
    }


def adapt_model(model, adaptation_loader, device='cuda', epochs=3):
    """
    Lightweight adaptation using unlabeled target data.
    Implements test-time batch normalization re-estimation + self-training.
    
    Args:
        model: Model to adapt
        adaptation_loader: Unlabeled target data
        device: Device
        epochs: Number of adaptation epochs
    
    Returns:
        Adapted model
    """
    model.train()
    model = model.to(device)
    
    # Use a very small learning rate for adaptation
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    print(f"Adapting model for {epochs} epochs...")
    
    for epoch in range(epochs):
        for batch in tqdm(adaptation_loader, desc=f'Adaptation epoch {epoch+1}'):
            images = batch['images'].to(device)
            
            # Self-training: use model's own predictions as pseudo-labels
            with torch.no_grad():
                pseudo_labels = model.generate(images) if hasattr(model, 'generate') else model(images)
            
            # Forward with pseudo-labels
            outputs = model(images, pseudo_labels)
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return model


def print_track_results(results: Dict):
    """Print track evaluation results."""
    print("\n" + "="*70)
    print(f"TRACK EVALUATION: {results.get('track', 'Unknown')}")
    print("="*70)
    
    if 'track' in results and 'SSL-OCR' in results['track']:
        if 'test_sup' in results:
            print(f"\nTest Set Performance:")
            print(f"  CER:     {results['test_sup']['CER']*100:.2f}%")
            print(f"  ESA-CER: {results['test_sup']['ESA-CER']*100:.2f}%")
        
        if 'annotation_efficiency_curve' in results:
            print(f"\nAnnotation Efficiency Curve:")
            for size, metrics in sorted(results['annotation_efficiency_curve'].items()):
                print(f"  {size:6d} samples: CER={metrics['CER']*100:.2f}%")
    
    elif 'track' in results and 'Cross-Session' in results['track']:
        zs = results.get('zero_shot', {})
        print(f"\nZero-Shot Evaluation:")
        print(f"  Source CER: {zs.get('CER_source', 0)*100:.2f}%")
        print(f"  Target CER: {zs.get('CER_target', 0)*100:.2f}%")
        print(f"  Generalization Gap (GG): {zs.get('GG', 0)*100:.2f}%")
        
        if 'with_adaptation' in results:
            adapt = results['with_adaptation']
            print(f"\nWith Adaptation (5% unlabeled target data):")
            print(f"  Target CER: {adapt.get('CER_target', 0)*100:.2f}%")
            print(f"  GG: {adapt.get('GG', 0)*100:.2f}%")
            print(f"  Improvement: {adapt.get('improvement', 0)*100:.2f}%")
    
    elif 'track' in results and 'Operational-Fidelity' in results['track']:
        print(f"\nOverall Metrics:")
        print(f"  CER:     {results['overall']['CER']*100:.2f}%")
        print(f"  ESA-CER: {results['overall']['ESA-CER']*100:.2f}%")
        print(f"  RI:      {results['overall']['RI']:.3f}")
        
        if 'by_tier' in results:
            print(f"\nQuality Tier Analysis:")
            for tier, metrics in results['by_tier'].items():
                print(f"  {tier:15s}: CER={metrics['CER']*100:.2f}%, OQS={metrics['mean_OQS']:.3f}, n={metrics['num_samples']}")
    
    print("="*70 + "\n")
