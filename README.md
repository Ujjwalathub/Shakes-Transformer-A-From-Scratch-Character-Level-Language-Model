# Transformer Model from Scratch - Hackathon Challenge

A complete implementation of a Transformer model trained on Shakespeare text, built from scratch without using `nn.Transformer`.

## Project Structure

```
Project/
├── data/
│   └── shakespeare.txt          # Raw Shakespeare text
├── checkpoints/
│   ├── best_model.pt           # Best trained model weights
│   ├── checkpoint_epoch_*.pt    # Checkpoint after each epoch
│   └── training_history.json    # Training metrics
├── scripts/
│   ├── data_pipeline.py         # Data loading and tokenization
│   ├── transformer_model.py     # Transformer architecture
│   ├── train.py                 # Training loop and optimizer setup
│   └── inference.py             # Inference and evaluation
├── main.py                      # Main training script
└── requirements.txt             # Dependencies
```

## Features Implemented

### 1. Data Pipeline (`data_pipeline.py`)
- **Word-based Tokenizer**: Maps words to integers with vocabulary size ≥ 4000
- **PyTorch Dataset**: Returns sequences (X, Y) where Y is X shifted by 1 position
- **DataLoader**: Batch size 32 with shuffling for training

### 2. Transformer Architecture (`transformer_model.py`)
- **Scaled Dot-Product Attention**: Implements $Attention(Q,K,V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V$
- **Causal Mask**: Look-ahead mask prevents attention to future tokens
- **Multi-Head Attention**: 4 heads with d_model=128 (32 per head)
- **Feed-Forward Network**: Two-layer network with ReLU (hidden dimension 512)
- **Pre-LN Architecture**: LayerNorm before each sublayer with residual connections
- **Full Model**: 4 transformer blocks with positional encoding

### 3. Training (`train.py`)
- **Optimizer**: Adam with lr=0.001, weight_decay=0.0001
- **Learning Rate Scheduler**: Linear warmup (500 steps) followed by cosine annealing decay
- **Loss Function**: CrossEntropyLoss with label smoothing (α=0.1)
- **Gradient Clipping**: Clipped at 1.0 to prevent exploding gradients
- **Training**: 10 epochs with validation

### 4. Inference & Evaluation (`inference.py`)
- **Top-K Prediction**: Returns top 5 most likely next words with probabilities
- **Text Generation**: Generates text continuations token by token
- **Accuracy Evaluation**: Measures Top-1 accuracy on test set
- **Target**: >15% accuracy (vs random baseline of ~0.02%)

## Setup Instructions

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (or CPU, but training will be slower)

### Installation

1. **Clone/navigate to the project directory**
```bash
cd Project
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Prepare Shakespeare text**
   - Download Shakespeare text and save as `data/shakespeare.txt`
   - Or use the provided sample if available

### Running the Training

```bash
python main.py
```

The script will:
1. Load and tokenize the Shakespeare text
2. Create train/validation splits
3. Initialize the Transformer model
4. Train for 10 epochs with detailed progress
5. Save the best model to `checkpoints/best_model.pt`
6. Evaluate on test set and show results
7. Demonstrate text generation capabilities

### Using the Trained Model

```python
import torch
from scripts.data_pipeline import create_dataloaders
from scripts.transformer_model import TransformerModel
from scripts.inference import InferenceEngine

# Load data and tokenizer
train_loader, val_loader, tokenizer = create_dataloaders('data/shakespeare.txt')

# Create and load model
model = TransformerModel(vocab_size=len(tokenizer.word2idx))
model.load_state_dict(torch.load('checkpoints/best_model.pt'))

# Create inference engine
engine = InferenceEngine(model, tokenizer)

# Get top-5 predictions
top_5 = engine.predict_top_k("to be or not to be", k=5)
print(top_5)

# Generate text continuation
continuation = engine.predict_text_continuation("to be or not to be", num_words=10)
print(continuation)
```

## Model Configuration

- **Model Dimension**: 128
- **Number of Heads**: 4 (32 dimensions per head)
- **Feed-Forward Dimension**: 512
- **Number of Layers**: 4 transformer blocks
- **Sequence Length**: 32 tokens
- **Dropout**: 0.1
- **Positional Encoding**: Sinusoidal absolute positional encoding

## Training Configuration

- **Batch Size**: 32
- **Epochs**: 10
- **Optimizer**: Adam (lr=0.001, weight_decay=0.0001)
- **Learning Rate Schedule**: 
  - Warmup: 500 steps (linear)
  - Decay: Cosine annealing after warmup
- **Loss**: CrossEntropyLoss with label smoothing (α=0.1)
- **Gradient Clipping**: 1.0
- **Train/Val Split**: 80/20

## Expected Performance

- **Target Top-1 Accuracy**: >15%
- **Random Baseline**: ~0.025% (1/vocab_size)
- **Expected Improvement**: >600x over random baseline

## Outputs

After training, the `checkpoints/` directory contains:
- `best_model.pt`: Best model weights (lowest validation loss)
- `checkpoint_epoch_*.pt`: Checkpoint after each epoch
- `training_history.json`: Training and validation loss curves

## Technical Details

### Scaled Dot-Product Attention
The attention mechanism computes:
$$Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

With causal masking to prevent the model from attending to future tokens.

### Multi-Head Attention
Input dimension is split into multiple heads:
- d_model = 128
- num_heads = 4
- d_k = 128 / 4 = 32

Each head learns different representations, and outputs are concatenated.

### Pre-LayerNorm Architecture
Instead of Post-LN (standard), uses Pre-LN:
- LayerNorm → Attention → Residual
- LayerNorm → FFN → Residual

This improves training stability and convergence.

## Troubleshooting

**Out of Memory Error**:
- Reduce batch size in `main.py`
- Reduce model dimensions (d_model, d_hidden)
- Use CPU instead of GPU

**Slow Training**:
- Use GPU (CUDA) if available
- Reduce sequence length
- Reduce number of transformer layers

**Missing Data File**:
- Ensure `shakespeare.txt` is in the `data/` directory
- Sample text should be at least 1MB in size

## References

- Attention Is All You Need (Vaswani et al., 2017)
- Pre Norm vs Post Norm (Xiong et al., 2020)
- Label Smoothing (Szegedy et al., 2016)

## License

This implementation is provided for educational purposes as part of a hackathon challenge.
