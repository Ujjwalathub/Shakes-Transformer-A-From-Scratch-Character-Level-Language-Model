"""
Transformer Model Implementation from Scratch
Implements scaled dot-product attention, multi-head attention, and transformer blocks
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention mechanism."""
    
    def __init__(self, d_k: int, dropout: float = 0.1):
        """
        Args:
            d_k: Dimension of key/query
            dropout: Dropout probability
        """
        super().__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: Query tensor (batch_size, seq_len, d_k)
            K: Key tensor (batch_size, seq_len, d_k)
            V: Value tensor (batch_size, seq_len, d_v)
            mask: Attention mask (optional)
        
        Returns:
            output: Attention output (batch_size, seq_len, d_v)
            attention_weights: Attention weights (batch_size, seq_len, seq_len)
        """
        # Compute attention scores: Attention(Q,K,V) = softmax(Q*K^T / sqrt(d_k)) * V
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply causal mask (look-ahead mask) if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism."""
    
    def __init__(self, d_model: int = 128, num_heads: int = 4, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension (default 128)
            num_heads: Number of attention heads (default 4)
            dropout: Dropout probability
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 32 for d_model=128, num_heads=4
        
        # Linear transformations for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(self.d_k, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q, K, V: Input tensors (batch_size, seq_len, d_model)
            mask: Attention mask (optional)
        
        Returns:
            output: Multi-head attention output
        """
        batch_size = Q.shape[0]
        
        # Linear transformations
        Q = self.W_q(Q)  # (batch_size, seq_len, d_model)
        K = self.W_k(K)  # (batch_size, seq_len, d_model)
        V = self.W_v(V)  # (batch_size, seq_len, d_model)
        
        # Split into multiple heads
        # Reshape to (batch_size, seq_len, num_heads, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # Now shape: (batch_size, num_heads, seq_len, d_k)
        
        # Apply scaled dot-product attention
        attn_output, _ = self.attention(Q, K, V, mask)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        
        # Final linear transformation
        output = self.W_o(attn_output)
        output = self.dropout(output)
        
        return output


class FeedForwardNetwork(nn.Module):
    """Feed-Forward Network (FFN) with two linear layers."""
    
    def __init__(self, d_model: int = 128, d_hidden: int = 512, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension (default 128)
            d_hidden: Hidden layer dimension (default 512)
            dropout: Dropout probability
        """
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
        
        Returns:
            output: FFN output (batch_size, seq_len, d_model)
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer Block with Pre-LayerNorm architecture."""
    
    def __init__(self, d_model: int = 128, num_heads: int = 4, 
                 d_hidden: int = 512, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension (default 128)
            num_heads: Number of attention heads (default 4)
            d_hidden: Hidden dimension in FFN (default 512)
            dropout: Dropout probability
        """
        super().__init__()
        
        # Pre-LN architecture: LayerNorm -> Sublayer -> Residual
        self.ln1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model, d_hidden, dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Attention mask (optional)
        
        Returns:
            output: Transformer block output
        """
        # Pre-LN Attention with residual connection
        x_norm = self.ln1(x)
        attn_output = self.attention(x_norm, x_norm, x_norm, mask)
        x = x + attn_output
        
        # Pre-LN FFN with residual connection
        x_norm = self.ln2(x)
        ffn_output = self.ffn(x_norm)
        x = x + ffn_output
        
        return x


class TransformerModel(nn.Module):
    """Complete Transformer model for Shakespeare text."""
    
    def __init__(self, vocab_size: int, d_model: int = 128, num_heads: int = 4,
                 d_hidden: int = 512, num_layers: int = 4, seq_length: int = 32,
                 dropout: float = 0.1, max_no_improve: int = 5):
        """
        Args:
            vocab_size: Vocabulary size (>= 4000)
            d_model: Model dimension (default 128)
            num_heads: Number of attention heads (default 4)
            d_hidden: Hidden dimension in FFN (default 512)
            num_layers: Number of transformer blocks (default 4)
            seq_length: Sequence length (default 32)
            dropout: Dropout probability
            max_no_improve: For early stopping
        """
        super().__init__()
        
        self.d_model = d_model
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.positional_encoding = self._create_positional_encoding(seq_length, d_model)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_hidden, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def _create_positional_encoding(self, seq_length: int, d_model: int):
        """Create positional encoding for sequences."""
        pe = torch.zeros(seq_length, d_model)
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('positional_encoding', pe)
        return pe
    
    def _create_causal_mask(self, seq_length: int, device):
        """Create causal mask (look-ahead mask) to prevent attention to future tokens."""
        mask = torch.tril(torch.ones(seq_length, seq_length, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
    
    def forward(self, x):
        """
        Args:
            x: Input token indices (batch_size, seq_length)
        
        Returns:
            logits: Output logits (batch_size, seq_length, vocab_size)
        """
        batch_size, seq_length = x.shape
        
        # Embedding
        x = self.embedding(x) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = x + self.positional_encoding[:, :seq_length, :]
        x = self.dropout(x)
        
        # Create causal mask
        mask = self._create_causal_mask(seq_length, x.device)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        # Output projection
        logits = self.output_layer(x)
        
        return logits


if __name__ == "__main__":
    # Test the model
    batch_size = 32
    seq_length = 32
    vocab_size = 4000
    
    model = TransformerModel(vocab_size=vocab_size, d_model=128, num_heads=4,
                            d_hidden=512, num_layers=4, seq_length=seq_length)
    
    # Test forward pass
    x = torch.randint(0, vocab_size, (batch_size, seq_length))
    logits = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
