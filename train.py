"""
Training script for Transformer model
Implements training loop with optimizer, scheduler, and loss functions

🚀 GPU OPTIMIZATION GUIDE
=========================
This module is fully optimized for GPU training. Key requirements:

1. DEVICE VARIABLE: Defined globally at module level
   - Automatically detects CUDA GPU availability
   - Falls back to CPU if no GPU found
   
2. MODEL MIGRATION: Model must be moved to device BEFORE training
   - model = Transformer(CONFIG).to(device)
   
3. DATA MIGRATION (CRITICAL): Every batch must be moved to GPU
   - inputs = inputs.to(device)
   - targets = targets.to(device)
   - Done inside the training loop for EVERY batch
   
4. LOSS FUNCTION: Loss module also moved to device
   - loss_fn = loss_fn.to(device)

Expected Performance:
- CPU: ~1-2 hours for full training on Shakespeare dataset
- GPU (NVIDIA): ~5-10 minutes for full training

DEVICE MISMATCH TROUBLESHOOTING:
If you see "RuntimeError: Expected all tensors to be on the same device":
✓ Check: model.to(device)
✓ Check: inputs.to(device) in loop
✓ Check: targets.to(device) in loop
✓ Debug: print(inputs.device) should show 'cuda:0'
✓ Monitor: nvidia-smi shows GPU memory usage
"""
import torch
import torch.nn as nn
from torch.optim import Adam
import os
import math
from tqdm import tqdm
import json
from pathlib import Path

# ============================================================================
# 1. DEVICE SETUP - GPU/CPU Detection
# ============================================================================
# This ensures GPU is used if available, otherwise falls back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
else:
    print(f"⚠ Running on CPU (GPU not available)")

# ============================================================================
# 2. TRAINING CONFIGURATION
# ============================================================================
# Hackathon-compliant training configuration
TRAINING_CONFIG = {
    'num_epochs': 10,
    'learning_rate': 0.001,
    'batch_size': 32,
    'gradient_clip': 1.0,
    'label_smoothing': 0.1,
    'weight_decay': 0.0001,
}


class WarmupCosineScheduler:
    """Learning rate scheduler with warmup + cosine decay."""
    
    def __init__(self, optimizer, warmup_steps: int = 500, total_steps: int = 10000):
        """
        Args:
            optimizer: PyTorch optimizer
            warmup_steps: Number of warmup steps (default 500)
            total_steps: Total training steps
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0
    
    def step(self):
        """Update learning rate based on current step."""
        self.current_step += 1
        lr = self._get_lr()
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def _get_lr(self) -> float:
        """Calculate learning rate for current step."""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = 0.001 * (self.current_step / self.warmup_steps)
        else:
            # Cosine decay after warmup
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = 0.001 * 0.5 * (1 + math.cos(math.pi * progress))
        
        return max(lr, 1e-6)  # Prevent lr from becoming too small


class CrossEntropyLossWithLabelSmoothing(nn.Module):
    """CrossEntropyLoss with label smoothing."""
    
    def __init__(self, vocab_size: int, smoothing: float = 0.1):
        """
        Args:
            vocab_size: Size of vocabulary
            smoothing: Label smoothing parameter (default 0.1)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.criterion = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, logits, targets):
        """
        Args:
            logits: Model output (batch_size, seq_len, vocab_size)
            targets: Target tokens (batch_size, seq_len)
        
        Returns:
            loss: Scalar loss value
        """
        # Reshape for label smoothing
        batch_size, seq_len, vocab_size = logits.shape
        logits_reshaped = logits.view(-1, vocab_size)
        targets_reshaped = targets.view(-1)
        
        # Create label smoothed targets
        smoothed_targets = torch.full(
            (targets_reshaped.size(0), vocab_size),
            self.smoothing / vocab_size,
            device=targets.device,
            dtype=logits.dtype
        )
        smoothed_targets.scatter_(1, targets_reshaped.unsqueeze(1), 1.0 - self.smoothing)
        
        # Apply log softmax to logits
        log_probs = torch.nn.functional.log_softmax(logits_reshaped, dim=-1)
        
        # Compute KL divergence loss
        loss = self.criterion(log_probs, smoothed_targets)
        
        return loss


def train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device, 
                epoch, clip_grad: float = 1.0):
    """
    Train for one epoch.
    
    CRITICAL GPU REQUIREMENTS:
    ✓ Model already on device via model.to(device)
    ✓ Loss function already on device via loss_fn.to(device)
    ✓ MUST move batch to device: inputs.to(device), targets.to(device)
    
    Args:
        model: Transformer model (must already be on device)
        train_loader: Training data loader
        optimizer: Adam optimizer
        scheduler: Learning rate scheduler
        loss_fn: Loss function with label smoothing (must already be on device)
        device: Device to train on (torch.device('cuda') or torch.device('cpu'))
        epoch: Current epoch number
        clip_grad: Gradient clipping threshold (default 1.0)
    
    Returns:
        avg_loss: Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=True)
    
    for batch_idx, (x, y) in enumerate(pbar):
        # CRITICAL: Move batch tensors to GPU
        # This is where PyTorch CPU-only issues most commonly occur
        x = x.to(device)  # Input sequences to device
        y = y.to(device)  # Target sequences to device
        
        # Forward pass
        logits = model(x)
        loss = loss_fn(logits, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping [Hackathon requirement]
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        # Optimization step
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def evaluate(model, val_loader, loss_fn, device):
    """
    Evaluate model on validation set.
    
    GPU REQUIREMENTS:
    ✓ Model already on device via model.to(device)
    ✓ Loss function already on device via loss_fn.to(device)
    ✓ MUST move batch to device: inputs.to(device), targets.to(device)
    
    Args:
        model: Transformer model (must already be on device)
        val_loader: Validation data loader
        loss_fn: Loss function (must already be on device)
        device: Device to evaluate on (torch.device('cuda') or torch.device('cpu'))
    
    Returns:
        avg_loss: Average validation loss
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Validation"):
            # Move validation batch to device
            x = x.to(device)
            y = y.to(device)
            
            logits = model(x)
            loss = loss_fn(logits, y)
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss


def train(model, train_loader, val_loader, num_epochs: int = 10, 
          device='cuda', checkpoint_dir: str = './checkpoints'):
    """
    Main training loop with GPU optimization.
    
    GPU SETUP CHECKLIST:
    ✓ model must be on device: model = model.to(device) BEFORE calling this function
    ✓ batch data moved inside loop: inputs.to(device), targets.to(device)
    ✓ loss function on device: loss_fn.to(device)
    
    GPU VERIFICATION:
    1. Check device: print(f"Using: {device}")
    2. Check GPU memory: nvidia-smi (run in separate terminal)
    3. Check tensor device: print(inputs.device) should show 'cuda:0'
    4. Check training speed: GPU = seconds/epoch, CPU = minutes/epoch
    
    Args:
        model: Transformer model (MUST already be on device via model.to(device))
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs (default 10) from TRAINING_CONFIG
        device: Device to train on (default 'cuda') - pass torch.device object
        checkpoint_dir: Directory to save checkpoints
    """
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Print GPU info at start of training
    print(f"\n{'='*70}")
    print(f"TRAINING START - Device Configuration")
    print(f"{'='*70}")
    print(f"Device: {device}")
    if isinstance(device, str):
        device = torch.device(device)
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"{'='*70}\n")
    
    # Optimizer
    optimizer = Adam(model.parameters(), lr=TRAINING_CONFIG['learning_rate'], 
                     weight_decay=TRAINING_CONFIG['weight_decay'])
    
    # Learning rate scheduler (assuming roughly 1000 batches per epoch)
    total_steps = num_epochs * 1000  # Rough estimate
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=500, total_steps=total_steps)
    
    # Loss function with label smoothing
    loss_fn = CrossEntropyLossWithLabelSmoothing(
        vocab_size=model.vocab_size, 
        smoothing=TRAINING_CONFIG['label_smoothing']
    )
    loss_fn = loss_fn.to(device)  # Move loss function to device
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'best_val_loss': float('inf'),
        'best_epoch': 0
    }
    
    # Training loop (10 epochs)
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, device,
            epoch, clip_grad=TRAINING_CONFIG['gradient_clip']
        )
        history['train_loss'].append(train_loss)
        
        # Validate
        val_loss = evaluate(model, val_loader, loss_fn, device)
        history['val_loss'].append(val_loss)
        
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < history['best_val_loss']:
            history['best_val_loss'] = val_loss
            history['best_epoch'] = epoch
            
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
            torch.save(model.state_dict(), best_model_path)
            print(f"✓ Best model saved to {best_model_path}")
        
        # Save checkpoint every epoch
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history
        }, checkpoint_path)
    
    # Save training history
    history_path = os.path.join(checkpoint_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        # Convert to serializable format
        history_serializable = {
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss'],
            'best_val_loss': float(history['best_val_loss']),
            'best_epoch': history['best_epoch']
        }
        json.dump(history_serializable, f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"Best validation loss: {history['best_val_loss']:.6f} (Epoch {history['best_epoch']})")
    print(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == "__main__":
    # This is a module to be imported into main training script
    print("Use this module by importing it in your main training script")
    print("Example:")
    print("  from train import train")
    print("  from transformer_model import TransformerModel")
    print("  from data_pipeline import create_dataloaders")
    print("  ")
    print("  train_loader, val_loader, tokenizer = create_dataloaders('data/shakespeare.txt')")
    print("  model = TransformerModel(vocab_size=len(tokenizer.word2idx))")
    print("  train(model, train_loader, val_loader, num_epochs=10)")
