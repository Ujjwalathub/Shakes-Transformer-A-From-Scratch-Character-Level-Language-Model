"""
Inference and Evaluation script
Implements top-K prediction and model evaluation

GPU OPTIMIZATION:
================
This module is optimized for GPU inference. Key optimizations include:

1. DEVICE INITIALIZATION: The InferenceEngine automatically moves the model
   to the specified device (GPU by default) via model.to(device).
   
2. BATCH PROCESSING: Input tensors are moved to GPU before forward pass
   via x.to(self.device).
   
3. MODEL CHECKPOINT LOADING: When loading checkpoints, use map_location=device
   to load directly to GPU without intermediate CPU copy.
   
4. OUTPUT HANDLING: Output is moved back to CPU for Python processing via
   .cpu() or .item() methods.

Expected Performance:
- Single inference: ~1-2 milliseconds on GPU vs ~10-50ms on CPU
- Batch inference: Much faster throughput on GPU due to parallelization

DEVICE MISMATCH DEBUG:
If you encounter device mismatch errors, ensure:
- model is on device via model.to(device)
- inputs are on device via x.to(device)
- checkpoint is loaded with map_location=device
"""
import torch
import torch.nn.functional as F
from typing import List, Tuple
import os


class InferenceEngine:
    """Inference engine for the Transformer model."""
    
    def __init__(self, model, tokenizer, device='cuda'):
        """
        Args:
            model: Trained Transformer model
            tokenizer: WordTokenizer instance
            device: Device to run inference on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def predict_top_k(self, input_text: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Predict top-K most likely next words given input text.
        
        Args:
            input_text: Input text string
            k: Number of top predictions (default 5)
        
        Returns:
            List of tuples (word, probability) sorted by probability (descending)
        """
        with torch.no_grad():
            # Tokenize input
            tokens = self.tokenizer.encode(input_text)
            
            # Pad or truncate to seq_length
            seq_length = self.model.seq_length
            if len(tokens) < seq_length:
                # Pad with PAD token
                tokens = tokens + [self.tokenizer.word2idx['<PAD>']] * (seq_length - len(tokens))
            else:
                # Take last seq_length tokens
                tokens = tokens[-seq_length:]
            
            # Convert to tensor
            x = torch.tensor([tokens], dtype=torch.long).to(self.device)
            
            # Forward pass
            logits = self.model(x)
            
            # Get logits for the last position
            last_logits = logits[0, -1, :]  # (vocab_size,)
            
            # Get probabilities
            probs = F.softmax(last_logits, dim=-1)
            
            # Get top-K
            top_k_probs, top_k_indices = torch.topk(probs, k)
            
            # Convert to words and probabilities
            results = []
            for idx, prob in zip(top_k_indices.cpu().numpy(), top_k_probs.cpu().numpy()):
                word = self.tokenizer.idx2word.get(int(idx), '<UNK>')
                results.append((word, float(prob)))
            
            return results
    
    def predict_text_continuation(self, input_text: str, num_words: int = 10) -> str:
        """
        Generate text continuation by predicting next words one at a time.
        
        Args:
            input_text: Input text string
            num_words: Number of words to generate
        
        Returns:
            Generated text continuation
        """
        with torch.no_grad():
            # Tokenize input
            tokens = self.tokenizer.encode(input_text)
            seq_length = self.model.seq_length
            
            generated_tokens = []
            
            for _ in range(num_words):
                # Prepare input
                if len(tokens) < seq_length:
                    x = tokens + [self.tokenizer.word2idx['<PAD>']] * (seq_length - len(tokens))
                else:
                    x = tokens[-seq_length:]
                
                x = torch.tensor([x], dtype=torch.long).to(self.device)
                
                # Forward pass
                logits = self.model(x)
                
                # Get prediction for last position
                last_logits = logits[0, -1, :]
                probs = F.softmax(last_logits, dim=-1)
                
                # Sample next token (greedy: take argmax)
                next_token = torch.argmax(probs).item()
                
                # Stop if we predict PAD or UNK
                if next_token in [self.tokenizer.word2idx['<PAD>'], 
                                 self.tokenizer.word2idx['<UNK>']]:
                    break
                
                generated_tokens.append(next_token)
                tokens.append(next_token)
            
            # Decode tokens to text
            generated_text = self.tokenizer.decode(generated_tokens)
            return generated_text


def evaluate_top1_accuracy(model, test_loader, tokenizer, device='cuda') -> float:
    """
    Evaluate top-1 accuracy on test set.
    
    Args:
        model: Trained Transformer model
        test_loader: Test data loader
        tokenizer: WordTokenizer instance
        device: Device to evaluate on
    
    Returns:
        Top-1 accuracy (%)
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            
            # Forward pass
            logits = model(x)
            
            # Get top-1 predictions for each position
            predictions = torch.argmax(logits, dim=-1)
            
            # Compare with targets
            correct += (predictions == y).sum().item()
            total += y.numel()
    
    accuracy = (correct / total) * 100 if total > 0 else 0.0
    return accuracy


def evaluate_model(model, test_loader, tokenizer, device='cuda',
                   checkpoint_path: str = None) -> dict:
    """
    Comprehensive model evaluation.
    
    Args:
        model: Transformer model (or path to checkpoint)
        test_loader: Test data loader
        tokenizer: WordTokenizer instance
        device: Device to evaluate on
        checkpoint_path: Path to checkpoint file (optional)
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Load checkpoint if provided (with proper device mapping)
    if checkpoint_path and isinstance(model, str):
        model = torch.load(checkpoint_path, map_location=device)
    
    # Ensure model is on the correct device
    model = model.to(device)
    model.eval()
    
    # Calculate top-1 accuracy
    top1_accuracy = evaluate_top1_accuracy(model, test_loader, tokenizer, device)
    
    # Calculate random baseline (1/vocab_size)
    random_baseline = (1 / len(tokenizer.word2idx)) * 100
    
    results = {
        'top1_accuracy': top1_accuracy,
        'random_baseline': random_baseline,
        'improvement_over_baseline': top1_accuracy - random_baseline,
        'meets_target': top1_accuracy > 15.0
    }
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Inference Engine Example")
    print("=" * 60)
    print("\nUsage:")
    print("  from inference import InferenceEngine")
    print("  from transformer_model import TransformerModel")
    print("  from data_pipeline import create_dataloaders")
    print("  ")
    print("  train_loader, val_loader, tokenizer = create_dataloaders('data/shakespeare.txt')")
    print("  model = TransformerModel(vocab_size=len(tokenizer.word2idx))")
    print("  model.load_state_dict(torch.load('checkpoints/best_model.pt'))")
    print("  ")
    print("  engine = InferenceEngine(model, tokenizer)")
    print("  top_5 = engine.predict_top_k('to be or not')")
    print("  print(top_5)")
    print("  ")
    print("  continuation = engine.predict_text_continuation('to be or not', num_words=10)")
    print("  print(continuation)")
