"""
OCR evaluation script.
Evaluates models on all standard metrics (CER, WER, ESA-CER, RI).
"""

import json
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from metrics import (
    calculate_cer, calculate_wer, calculate_cer_per_sample,
    calculate_esa_cer, calculate_cer_by_subject,
    calculate_oqs, calculate_ri
)


def evaluate_ocr(
    model,
    test_loader,
    tokenizer,
    oqs_calculator=None,
    device='cuda',
) -> Dict:
    """
    Evaluate OCR model on test set.
    
    Args:
        model: OCR model
        test_loader: Test data loader
        tokenizer: Tokenizer for decoding
        oqs_calculator: Optional OQS calculator for RI
        device: Device to run on
    
    Returns:
        Dictionary with all evaluation metrics
    """
    model.eval()
    model = model.to(device)
    
    all_predictions = []
    all_references = []
    all_subjects = []
    all_styles = []
    all_qualities = []
    all_metadata = []
    
    oqs_scores = [] if oqs_calculator else None
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            images = batch['images'].to(device)
            transcriptions = batch['transcriptions']
            metadata = batch['metadata']
            
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
                # Fallback
                outputs = model(images)
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                else:
                    logits = outputs
                preds = logits.argmax(dim=-1)
                predictions = tokenizer.batch_decode(preds)
            
            all_predictions.extend(predictions)
            all_references.extend(transcriptions)
            
            # Collect metadata
            for meta in metadata:
                all_subjects.append(meta.get('subject', 'Unknown'))
                all_styles.append(meta.get('handwriting_style', 'Unknown'))
                all_qualities.append(meta.get('image_quality_tier', 'Medium'))
                all_metadata.append(meta)
            
            # Calculate OQS if calculator provided
            if oqs_calculator:
                for i, img in enumerate(images):
                    # Convert tensor to numpy
                    img_np = img.cpu().numpy()
                    if img_np.ndim == 3:
                        img_np = img_np[0]  # Remove channel dim if grayscale
                    
                    # Denormalize if needed
                    if img_np.min() < 0:
                        img_np = (img_np + 1) / 2 * 255
                    img_np = img_np.astype('uint8')
                    
                    oqs, _ = oqs_calculator.calculate(img_np)
                    oqs_scores.append(oqs)
    
    # Calculate overall metrics
    cer = calculate_cer(all_predictions, all_references)
    wer = calculate_wer(all_predictions, all_references)
    esa_cer = calculate_esa_cer(all_predictions, all_references, alpha=3.0)
    
    results = {
        'overall': {
            'CER': cer,
            'WER': wer,
            'ESA-CER': esa_cer,
            'num_samples': len(all_predictions),
        }
    }
    
    # Per-subject metrics
    subject_cers = calculate_cer_by_subject(all_predictions, all_references, all_subjects)
    results['by_subject'] = {subj: {'CER': cer} for subj, cer in subject_cers.items()}
    
    # Per-style metrics
    style_stats = {}
    for pred, ref, style in zip(all_predictions, all_references, all_styles):
        if style not in style_stats:
            style_stats[style] = {'distance': 0, 'chars': 0}
        
        pred_chars = list(pred)
        ref_chars = list(ref)
        
        import editdistance
        style_stats[style]['distance'] += editdistance.eval(pred_chars, ref_chars)
        style_stats[style]['chars'] += len(ref_chars)
    
    results['by_style'] = {
        style: {'CER': stats['distance'] / stats['chars'] if stats['chars'] > 0 else 0}
        for style, stats in style_stats.items()
    }
    
    # Calculate RI if OQS available
    if oqs_scores:
        from metrics.ri import RobustnessIndex
        ri_calculator = RobustnessIndex()
        
        # Calculate per-sample CER
        per_sample_cers = calculate_cer_per_sample(all_predictions, all_references)
        
        ri_stats = ri_calculator.calculate_by_tier(oqs_scores, per_sample_cers)
        results['quality_tiers'] = ri_stats
        results['overall']['RI'] = ri_stats['RI']
    
    # Save detailed predictions
    results['predictions'] = [
        {
            'prediction': pred,
            'reference': ref,
            'subject': subj,
            'style': style,
            'quality': qual,
        }
        for pred, ref, subj, style, qual in zip(
            all_predictions, all_references, all_subjects, all_styles, all_qualities
        )
    ]
    
    return results


def save_results(results: Dict, output_path: str):
    """Save evaluation results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {output_path}")


def print_results(results: Dict):
    """Print evaluation results in formatted way."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    overall = results['overall']
    print(f"\nOverall Metrics:")
    print(f"  CER:      {overall['CER']*100:.2f}%")
    print(f"  WER:      {overall['WER']*100:.2f}%")
    print(f"  ESA-CER:  {overall['ESA-CER']*100:.2f}%")
    if 'RI' in overall:
        print(f"  RI:       {overall['RI']:.3f}")
    print(f"  Samples:  {overall['num_samples']}")
    
    if 'by_subject' in results:
        print(f"\nPer-Subject CER:")
        for subj, metrics in sorted(results['by_subject'].items()):
            print(f"  {subj:15s}: {metrics['CER']*100:.2f}%")
    
    if 'by_style' in results:
        print(f"\nPer-Style CER:")
        for style, metrics in sorted(results['by_style'].items()):
            print(f"  {style:15s}: {metrics['CER']*100:.2f}%")
    
    if 'quality_tiers' in results:
        print(f"\nQuality Tier Analysis:")
        for tier, metrics in results['quality_tiers'].items():
            if tier != 'RI':
                print(f"  {tier:15s}: CER={metrics['CER']*100:.2f}%, n={metrics['count']}")
    
    print("="*60 + "\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--model_type', type=str, default='trocr', choices=['crnn', 'abinet', 'trocr', 'vit_ocr'])
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--annotation_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, default='results.json')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--calculate_oqs', action='store_true', help='Calculate OQS and RI')
    
    args = parser.parse_args()
    
    # Load model
    from models import CRNN, ABINet, TrOCRModel, ViTOCR
    from utils.checkpoint import load_checkpoint
    
    model_map = {
        'crnn': lambda: CRNN(num_classes=5000),
        'abinet': lambda: ABINet(num_classes=5000),
        'trocr': lambda: TrOCRModel(),
        'vit_ocr': lambda: ViTOCR(num_classes=5000),
    }
    
    model = model_map[args.model_type]()
    load_checkpoint(model, args.model_path)
    
    # Create data loader
    from data import ExamHandOCRDataset, get_dataloader
    from data.transforms import get_val_transforms
    
    test_dataset = ExamHandOCRDataset(
        data_root=args.data_root,
        split='test-sup',
        annotation_file=args.annotation_file,
        transform=get_val_transforms(),
    )
    
    test_loader = get_dataloader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # OQS calculator
    oqs_calc = None
    if args.calculate_oqs:
        from metrics.oqs import OperationalQualityScore
        oqs_calc = OperationalQualityScore()
    
    # Evaluate
    results = evaluate_ocr(model, test_loader, None, oqs_calc, args.device)
    
    # Print and save
    print_results(results)
    save_results(results, args.output_file)
