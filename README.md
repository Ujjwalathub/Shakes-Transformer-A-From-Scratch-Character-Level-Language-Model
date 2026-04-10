# Shakes-Transformer: Next-Word Prediction

A pedagogical implementation of the Transformer architecture built from scratch for next-word prediction on Shakespearean text. [cite_start]This project was developed as part of a Hackathon challenge to demonstrate mastery of self-attention mechanisms and training pipelines[cite: 1, 6].

## 🚀 Overview
[cite_start]The goal of this project is to build a Transformer-based language model that predicts the next word in a sequence using a controlled vocabulary of ~5,000 words[cite: 4, 11]. [cite_start]The model is trained on a ~5MB Shakespeare corpus and achieves measurable improvements over a random baseline [cite: 9, 10, 181-185].

### Key Constraints:
- [cite_start]**No `nn.Transformer`:** Multi-head attention, positional encoding, and transformer blocks are implemented from scratch using basic `nn.Linear` and `nn.LayerNorm` layers [cite: 39, 372-373].
- [cite_start]**Causal Masking:** Strict implementation of look-ahead masks to prevent information leakage during training[cite: 43].
- [cite_start]**GPU Accelerated:** Optimized training loop for CUDA-enabled devices[cite: 381].

## 🏗️ Model Architecture
[cite_start]The architecture follows a standard Transformer-Decoder style with the following configuration [cite: 63-77]:
- **Model Dimension ($d_{model}$):** 128
- **Attention Heads:** 4
- **Layers:** 2
- **Feed-Forward Dimension ($d_{ff}$):** 512
- **Max Sequence Length:** 32 tokens
- **Positional Encoding:** Sinusoidal ($sin/cos$)



[Image of Transformer architecture diagram]


## 📁 Repository Structure
```text
transformer-hackathon/
├── CHECKPOINT_1_Data/          # Tokenizer and PyTorch DataLoaders [cite: 248]
├── CHECKPOINT_2_Model/         # Manual Transformer implementation [cite: 260]
├── CHECKPOINT_3_Training/      # Training loop and best_model.pt [cite: 260]
├── CHECKPOINT_4_Inference/     # Top-K prediction script [cite: 260]
├── CHECKPOINT_5_Evaluation/    # Metrics and results.json [cite: 260]
├── CHECKPOINT_6_Ablation/      # Performance analysis [cite: 260]
├── requirements.txt            # Project dependencies [cite: 247]
└── README.md                   # This file [cite: 246]
