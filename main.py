#!/usr/bin/env python3
"""
ExamHandOCR: Main entry point for training and evaluation.

Usage:
    # Train OCR model
    python main.py train --model trocr --config configs/default.yaml
    
    # SSL Pre-training
    python main.py pretrain --config configs/ssl_pretrain.yaml
    
    # Evaluate model
    python main.py evaluate --model_path outputs/best_model.pth --config configs/default.yaml
    
    # Evaluate benchmark tracks
    python main.py evaluate_track --track ssl --model_path outputs/best_model.pth
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(
        description='ExamHandOCR: Benchmark Dataset for Examination Handwriting OCR',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Train TrOCR model
  python main.py train --model trocr --data_root ./data --annotation_file ./annotations.json
  
  # Train with SSL pre-training
  python main.py train --model trocr --ssl_pretrained --ssl_path ./mae_best.pth
  
  # Evaluate on test set
  python main.py evaluate --model_path ./best_model.pth --data_root ./data
  
  # Evaluate specific track
  python main.py evaluate_track --track cross_session --model_path ./best_model.pth
        '''
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train OCR model')
    train_parser.add_argument('--model', type=str, default='trocr',
                             choices=['crnn', 'abinet', 'trocr', 'vit_ocr'],
                             help='Model architecture')
    train_parser.add_argument('--config', type=str, default='configs/default.yaml',
                             help='Configuration file')
    train_parser.add_argument('--data_root', type=str, required=True,
                             help='Root directory of dataset')
    train_parser.add_argument('--annotation_file', type=str, required=True,
                             help='Path to annotation JSON file')
    train_parser.add_argument('--output_dir', type=str, default='./outputs',
                             help='Output directory')
    train_parser.add_argument('--batch_size', type=int, default=32,
                             help='Batch size')
    train_parser.add_argument('--epochs', type=int, default=50,
                             help='Number of epochs')
    train_parser.add_argument('--lr', type=float, default=1e-4,
                             help='Learning rate')
    train_parser.add_argument('--device', type=str, default='cuda',
                             help='Device (cuda/cpu)')
    train_parser.add_argument('--ssl_pretrained', action='store_true',
                             help='Use SSL pretrained encoder')
    train_parser.add_argument('--ssl_path', type=str, default=None,
                             help='Path to SSL pretrained checkpoint')
    train_parser.add_argument('--seed', type=int, default=42,
                             help='Random seed')
    
    # Pretrain command (SSL)
    pretrain_parser = subparsers.add_parser('pretrain', help='SSL pre-training with MAE')
    pretrain_parser.add_argument('--config', type=str, default='configs/ssl_pretrain.yaml',
                                help='Configuration file')
    pretrain_parser.add_argument('--data_root', type=str, required=True,
                                help='Root directory of unannotated images')
    pretrain_parser.add_argument('--output_dir', type=str, default='./outputs_ssl',
                                help='Output directory')
    pretrain_parser.add_argument('--batch_size', type=int, default=256,
                                help='Batch size')
    pretrain_parser.add_argument('--epochs', type=int, default=100,
                                help='Number of epochs')
    pretrain_parser.add_argument('--lr', type=float, default=1.5e-4,
                                help='Learning rate')
    pretrain_parser.add_argument('--mask_ratio', type=float, default=0.75,
                                help='MAE mask ratio')
    pretrain_parser.add_argument('--device', type=str, default='cuda',
                                help='Device (cuda/cpu)')
    pretrain_parser.add_argument('--seed', type=int, default=42,
                                help='Random seed')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    eval_parser.add_argument('--model_path', type=str, required=True,
                            help='Path to model checkpoint')
    eval_parser.add_argument('--model_type', type=str, default='trocr',
                            choices=['crnn', 'abinet', 'trocr', 'vit_ocr'],
                            help='Model architecture')
    eval_parser.add_argument('--config', type=str, default='configs/default.yaml',
                            help='Configuration file')
    eval_parser.add_argument('--data_root', type=str, required=True,
                            help='Root directory of dataset')
    eval_parser.add_argument('--annotation_file', type=str, required=True,
                            help='Path to annotation JSON file')
    eval_parser.add_argument('--output_file', type=str, default='results.json',
                            help='Output results file')
    eval_parser.add_argument('--batch_size', type=int, default=32,
                            help='Batch size')
    eval_parser.add_argument('--device', type=str, default='cuda',
                            help='Device (cuda/cpu)')
    eval_parser.add_argument('--calculate_oqs', action='store_true',
                            help='Calculate OQS and RI')
    
    # Evaluate track command
    track_parser = subparsers.add_parser('evaluate_track', help='Evaluate benchmark track')
    track_parser.add_argument('--track', type=str, required=True,
                           choices=['ssl', 'cross_session', 'operational_fidelity', 'all'],
                           help='Track to evaluate')
    track_parser.add_argument('--model_path', type=str, required=True,
                           help='Path to model checkpoint')
    track_parser.add_argument('--model_type', type=str, default='trocr',
                           choices=['crnn', 'abinet', 'trocr', 'vit_ocr'],
                           help='Model architecture')
    track_parser.add_argument('--config', type=str, default='configs/default.yaml',
                           help='Configuration file')
    track_parser.add_argument('--data_root', type=str, required=True,
                           help='Root directory of dataset')
    track_parser.add_argument('--output_file', type=str, default='track_results.json',
                           help='Output results file')
    track_parser.add_argument('--device', type=str, default='cuda',
                           help='Device (cuda/cpu)')
    
    # Inference command
    infer_parser = subparsers.add_parser('inference', help='Run inference on images')
    infer_parser.add_argument('--model_path', type=str, required=True,
                             help='Path to model checkpoint')
    infer_parser.add_argument('--model_type', type=str, default='trocr',
                             choices=['crnn', 'abinet', 'trocr', 'vit_ocr'],
                             help='Model architecture')
    infer_parser.add_argument('--image_path', type=str, required=True,
                             help='Path to image or directory of images')
    infer_parser.add_argument('--output_file', type=str, default='predictions.txt',
                             help='Output file for predictions')
    infer_parser.add_argument('--device', type=str, default='cuda',
                             help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Execute command
    if args.command == 'train':
        from train.train_ocr import train_ocr_model
        from models import CRNN, ABINet, TrOCRModel, ViTOCR
        from data import ExamHandOCRDataset, get_dataloader
        from data.transforms import get_train_transforms, get_val_transforms
        from data.tokenizer import ExamHandOCRTokenizer
        from utils.checkpoint import load_pretrained_encoder
        
        # Create model
        model_map = {
            'crnn': lambda: CRNN(num_classes=5000),
            'abinet': lambda: ABINet(num_classes=5000),
            'trocr': lambda: TrOCRModel(),
            'vit_ocr': lambda: ViTOCR(num_classes=5000),
        }
        
        print(f"Creating {args.model} model...")
        model = model_map[args.model]()
        
        # Load SSL pretrained encoder if specified
        if args.ssl_pretrained and args.ssl_path:
            print(f"Loading SSL pretrained encoder from {args.ssl_path}")
            load_pretrained_encoder(model, args.ssl_path)
        
        # Create tokenizer
        tokenizer = ExamHandOCRTokenizer(max_length=512)
        # In practice, load vocab from training data
        
        # Create datasets
        print("Loading datasets...")
        train_dataset = ExamHandOCRDataset(
            data_root=args.data_root,
            split='train-sup',
            annotation_file=args.annotation_file,
            transform=get_train_transforms(),
        )
        
        val_dataset = ExamHandOCRDataset(
            data_root=args.data_root,
            split='val',
            annotation_file=args.annotation_file,
            transform=get_val_transforms(),
        )
        
        train_loader = get_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = get_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Training config
        config = {
            'model_type': args.model,
            'epochs': args.epochs,
            'learning_rate': args.lr,
            'weight_decay': 0.01,
            'use_ctc': args.model == 'crnn',
        }
        
        # Train
        print("Starting training...")
        model = train_ocr_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            tokenizer=tokenizer,
            config=config,
            device=args.device,
            output_dir=args.output_dir,
        )
        
        print("Training completed!")
    
    elif args.command == 'pretrain':
        from train.train_ssl import pretrain_ssl
        from models.ssl_mae import MaskedAutoencoder
        from data import SSLDataset, get_ssl_dataloader
        from data.transforms import get_ssl_transforms
        
        print("Creating MAE model for SSL pre-training...")
        model = MaskedAutoencoder(
            img_size=(384, 128),
            patch_size=args.mask_ratio * 16 // 0.75,  # Convert ratio to patch size
            mask_ratio=args.mask_ratio,
            embed_dim=768,
            encoder_depth=12,
            decoder_dim=512,
            decoder_depth=8,
        )
        
        # Create dataset
        print("Loading unannotated dataset...")
        train_transform = get_ssl_transforms()
        
        train_dataset = SSLDataset(
            data_root=args.data_root,
            image_list=None,
            transform=train_transform,
        )
        
        train_loader = get_ssl_dataloader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=8,
        )
        
        # Config
        config = {
            'epochs': args.epochs,
            'learning_rate': args.lr,
            'warmup_epochs': 40,
            'weight_decay': 0.05,
        }
        
        # Pre-train
        print("Starting SSL pre-training...")
        model = pretrain_ssl(
            model=model,
            train_loader=train_loader,
            config=config,
            device=args.device,
            output_dir=args.output_dir,
        )
        
        print("SSL pre-training completed!")
    
    elif args.command == 'evaluate':
        from eval.evaluate_ocr import evaluate_ocr, print_results, save_results
        from models import CRNN, ABINet, TrOCRModel, ViTOCR
        from data import ExamHandOCRDataset, get_dataloader
        from data.transforms import get_val_transforms
        from data.tokenizer import ExamHandOCRTokenizer
        from utils.checkpoint import load_checkpoint
        from metrics.oqs import OperationalQualityScore
        
        # Load model
        model_map = {
            'crnn': lambda: CRNN(num_classes=5000),
            'abinet': lambda: ABINet(num_classes=5000),
            'trocr': lambda: TrOCRModel(),
            'vit_ocr': lambda: ViTOCR(num_classes=5000),
        }
        
        print(f"Loading {args.model_type} model from {args.model_path}...")
        model = model_map[args.model_type]()
        load_checkpoint(model, args.model_path)
        
        # Create tokenizer
        tokenizer = ExamHandOCRTokenizer(max_length=512)
        
        # Create dataset
        print("Loading test dataset...")
        test_dataset = ExamHandOCRDataset(
            data_root=args.data_root,
            split='test-sup',
            annotation_file=args.annotation_file,
            transform=get_val_transforms(),
        )
        
        test_loader = get_dataloader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        # OQS calculator
        oqs_calc = OperationalQualityScore() if args.calculate_oqs else None
        
        # Evaluate
        print("Evaluating...")
        results = evaluate_ocr(
            model=model,
            test_loader=test_loader,
            tokenizer=tokenizer,
            oqs_calculator=oqs_calc,
            device=args.device,
        )
        
        # Print and save
        print_results(results)
        save_results(results, args.output_file)
    
    elif args.command == 'evaluate_track':
        from eval.evaluate_tracks import (
            evaluate_semi_supervised_track,
            evaluate_cross_session_track,
            evaluate_operational_fidelity_track,
            print_track_results,
        )
        from models import CRNN, ABINet, TrOCRModel, ViTOCR
        from data import ExamHandOCRDataset, get_dataloader
        from data.transforms import get_val_transforms
        from data.tokenizer import ExamHandOCRTokenizer
        from utils.checkpoint import load_checkpoint
        
        # Load model
        model_map = {
            'crnn': lambda: CRNN(num_classes=5000),
            'abinet': lambda: ABINet(num_classes=5000),
            'trocr': lambda: TrOCRModel(),
            'vit_ocr': lambda: ViTOCR(num_classes=5000),
        }
        
        print(f"Loading {args.model_type} model from {args.model_path}...")
        model = model_map[args.model_type]()
        load_checkpoint(model, args.model_path)
        
        # Create tokenizer
        tokenizer = ExamHandOCRTokenizer(max_length=512)
        
        # Evaluate specific track
        results = {}
        
        if args.track in ['ssl', 'all']:
            # Create data loaders for different labeled set sizes
            # This would require specially prepared data splits
            pass
        
        if args.track in ['cross_session', 'all']:
            print("Evaluating Cross-Session Generalization track...")
            # Load source and target domain loaders
            # results['cross_session'] = evaluate_cross_session_track(...)
            pass
        
        if args.track in ['operational_fidelity', 'all']:
            print("Evaluating Operational-Fidelity track...")
            test_dataset = ExamHandOCRDataset(
                data_root=args.data_root,
                split='test-sup',
                annotation_file=Path(args.data_root) / 'annotations.json',
                transform=get_val_transforms(),
            )
            test_loader = get_dataloader(test_dataset, batch_size=32, shuffle=False)
            
            results['operational_fidelity'] = evaluate_operational_fidelity_track(
                model=model,
                test_loader=test_loader,
                tokenizer=tokenizer,
                device=args.device,
            )
            print_track_results(results['operational_fidelity'])
        
        # Save results
        import json
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Track results saved to {args.output_file}")
    
    elif args.command == 'inference':
        print("Inference not yet implemented.")


if __name__ == '__main__':
    main()
