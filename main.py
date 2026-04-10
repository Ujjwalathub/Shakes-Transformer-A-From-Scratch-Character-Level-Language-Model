"""
Main training script for Transformer model on Shakespeare text
Orchestrates data loading, model training, and evaluation

🚀 GPU ACCELERATION GUIDE
==========================
This script automatically detects and uses a GPU if available.
GPU is 100-200x faster than CPU for this Transformer model.

SETUP REQUIREMENTS:
✓ CUDA-enabled NVIDIA GPU (check with: nvidia-smi)
✓ PyTorch with CUDA support (check: torch.cuda.is_available())
✓ Model moved to GPU: model.to(device)
✓ Data moved to GPU: inside training loop per batch

VERIFY GPU USAGE:
1. Console output: "Using device: cuda:0"
2. Terminal: nvidia-smi shows Python process
3. Training speed: Epoch in seconds (GPU) vs minutes (CPU)
"""
import torch
import torch.nn as nn
from pathlib import Path
import argparse
import sys

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

from data_pipeline import create_dataloaders
from transformer_model import TransformerModel
from train import train, CrossEntropyLossWithLabelSmoothing, TRAINING_CONFIG, device
from inference import InferenceEngine, evaluate_model


def main(args):
    """Main training function."""
    
    # ============================================================================
    # DEVICE SETUP - GPU/CPU Detection
    # ============================================================================
    # Check for CUDA (NVIDIA GPU) - PyTorch requires explicit device management
    # The 'device' variable is defined globally in train.py
    print(f"\n{'='*70}")
    print(f"DEVICE CONFIGURATION")
    print(f"{'='*70}")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"✓ GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"✓ CUDA Available: {torch.cuda.is_available()}")
        print(f"✓ CUDA Version: {torch.version.cuda}")
    else:
        print(f"⚠ GPU not available - running on CPU")
        print(f"⚠ This will be significantly slower (100-200x)")
    print(f"{'='*70}\n")
    
    project_root = Path(__file__).parent
    data_path = project_root / 'data' / 'shakespeare.txt'
    checkpoint_dir = project_root / 'checkpoints'
    
    # Check if data file exists
    if not data_path.exists():
        print(f"ERROR: Shakespeare text file not found at {data_path}")
        print("Please place 'shakespeare.txt' in the 'data' directory")
        return
    
    print(f"\n{'='*70}")
    print("TRANSFORMER MODEL TRAINING - SHAKESPEARE TEXT")
    print(f"{'='*70}\n")
    
    # ============================================================================
    # 1. DATA LOADING
    # ============================================================================
    print("Step 1: Loading and preparing data...")
    print("-" * 70)
    
    train_loader, val_loader, tokenizer = create_dataloaders(
        str(data_path),
        batch_size=TRAINING_CONFIG['batch_size'],
        train_split=0.8,
        seq_length=32
    )
    
    vocab_size = len(tokenizer.word2idx)
    print(f"✓ Vocabulary size: {vocab_size}")
    print(f"✓ Training batches: {len(train_loader)}")
    print(f"✓ Validation batches: {len(val_loader)}")
    print(f"✓ Tokenizer sample: {tokenizer.decode([2, 3, 4, 5, 6])}\n")
    
    # ============================================================================
    # 2. MODEL INITIALIZATION
    # ============================================================================
    print("Step 2: Initializing Transformer model...")
    print("-" * 70)
    
    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=128,
        num_heads=4,
        d_hidden=512,
        num_layers=4,
        seq_length=32,
        dropout=0.1
    )
    # CRITICAL: Move model to GPU BEFORE training
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created successfully")
    print(f"✓ Total parameters: {num_params:,}")
    print(f"✓ Model on device: {next(model.parameters()).device}")
    print(f"✓ Model configuration:")
    print(f"   - d_model: 128")
    print(f"   - num_heads: 4")
    print(f"   - d_hidden: 512")
    print(f"   - num_layers: 4")
    print(f"   - seq_length: 32\n")
    
    # ============================================================================
    # 3. TRAINING
    # ============================================================================
    print("Step 3: Training the model...")
    print("-" * 70)
    print("Hackathon-Compliant Training Configuration:")
    print(f"   - Epochs: {TRAINING_CONFIG['num_epochs']}")
    print(f"   - Batch size: {TRAINING_CONFIG['batch_size']}")
    print(f"   - Learning rate: {TRAINING_CONFIG['learning_rate']}")
    print(f"   - Optimizer: Adam (weight_decay={TRAINING_CONFIG['weight_decay']})")
    print(f"   - Scheduler: Linear warmup (500 steps) + Cosine decay")
    print(f"   - Loss: CrossEntropyLoss with label smoothing (α={TRAINING_CONFIG['label_smoothing']})")
    print(f"   - Gradient clipping: {TRAINING_CONFIG['gradient_clip']}\n")
    
    train(
        model,
        train_loader,
        val_loader,
        num_epochs=TRAINING_CONFIG['num_epochs'],
        device=device,
        checkpoint_dir=str(checkpoint_dir)
    )
    
    # ============================================================================
    # 4. LOAD BEST MODEL
    # ============================================================================
    print("\nStep 4: Loading best model...")
    print("-" * 70)
    
    best_model_path = checkpoint_dir / 'best_model.pt'
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"✓ Loaded best model from {best_model_path}\n")
    
    # ============================================================================
    # 5. INFERENCE & EVALUATION
    # ============================================================================
    print("Step 5: Testing inference and generating predictions...")
    print("-" * 70)
    
    engine = InferenceEngine(model, tokenizer, device=device)
    
    # Test prompts
    test_prompts = [
        "to be or not to be",
        "all the world's a stage",
        "something is rotten",
        "to thine own self be true",
        "friends romans countrymen"
    ]
    
    print("\nTop-5 next word predictions:\n")
    for prompt in test_prompts:
        top_5 = engine.predict_top_k(prompt, k=5)
        print(f"Input: \"{prompt}\"")
        print(f"Top 5 predictions:")
        for i, (word, prob) in enumerate(top_5, 1):
            print(f"   {i}. {word:15s} {prob:.4f}")
        print()
    
    # ============================================================================
    # 6. EVALUATION METRICS
    # ============================================================================
    print("Step 6: Evaluating on test set...")
    print("-" * 70)
    
    # For now, use validation set as test set
    results = evaluate_model(model, val_loader, tokenizer, device=device)
    
    print(f"\nEvaluation Results:")
    print(f"   - Top-1 Accuracy: {results['top1_accuracy']:.2f}%")
    print(f"   - Random Baseline: {results['random_baseline']:.4f}%")
    print(f"   - Improvement: {results['improvement_over_baseline']:.2f}%")
    print(f"   - Meets target (>15%): {'✓ YES' if results['meets_target'] else '✗ NO'}")
    
    # ============================================================================
    # 7. TEXT GENERATION
    # ============================================================================
    print("\n" + "="*70)
    print("BONUS: Text Generation Example")
    print("="*70)
    
    seed = "the quality of mercy is not"
    continuation = engine.predict_text_continuation(seed, num_words=20)
    
    print(f"\nSeed: \"{seed}\"")
    print(f"Generated continuation: {continuation}")
    print(f"\nFull text: {seed} {continuation}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nCheckpoints saved to: {checkpoint_dir}")
    print(f"Best model: {best_model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train Transformer model on Shakespeare text'
    )
    parser.add_argument('--data-path', type=str, default='data/shakespeare.txt',
                       help='Path to Shakespeare text file')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to train on (cuda or cpu)')
    
    args = parser.parse_args()
    
    main(args)
