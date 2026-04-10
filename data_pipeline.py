"""
Data Pipeline for Shakespeare Text Processing
Implements tokenizer, dataset, and dataloader for training
"""
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re
from typing import List, Tuple


class WordTokenizer:
    """Word-based tokenizer that maps words to integers."""
    
    def __init__(self, vocab_size: int = 4000):
        self.vocab_size = vocab_size
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.vocab_size = vocab_size
    
    def build_vocabulary(self, text: str):
        """Build vocabulary from text with minimum vocab size of 4000."""
        # Tokenize text into words
        words = self._tokenize_text(text)
        
        # Count word frequencies
        word_counts = Counter(words)
        
        # Sort by frequency and take top words
        most_common = word_counts.most_common(self.vocab_size - 2)  # -2 for PAD and UNK
        
        # Build word to index mapping
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        print(f"Vocabulary built with {len(self.word2idx)} tokens (target: {self.vocab_size})")
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Convert text to list of words."""
        # Convert to lowercase and split into words
        text = text.lower()
        # Keep only alphanumeric and spaces
        text = re.sub(r'[^a-z0-9\s]', '', text)
        words = text.split()
        return words
    
    def encode(self, text: str) -> List[int]:
        """Convert text to indices."""
        words = self._tokenize_text(text)
        return [self.word2idx.get(word, self.word2idx['<UNK>']) for word in words]
    
    def decode(self, indices: List[int]) -> str:
        """Convert indices back to text."""
        words = [self.idx2word.get(idx, '<UNK>') for idx in indices]
        return ' '.join(words)


class ShakespeareDataset(Dataset):
    """PyTorch Dataset for Shakespeare text."""
    
    def __init__(self, text: str, tokenizer: WordTokenizer, seq_length: int = 32):
        """
        Args:
            text: Raw text string
            tokenizer: WordTokenizer instance
            seq_length: Sequence length (default 32)
        """
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        # Encode the entire text
        self.tokens = tokenizer.encode(text)
        
        # Create sequences
        self.sequences = []
        for i in range(len(self.tokens) - seq_length):
            # Input: [t_1, t_2, ..., t_32]
            # Target: [t_2, t_3, ..., t_33]
            x = self.tokens[i:i + seq_length]
            y = self.tokens[i + 1:i + seq_length + 1]
            self.sequences.append((x, y))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        x, y = self.sequences[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def create_dataloaders(file_path: str, batch_size: int = 32, 
                       train_split: float = 0.8, seq_length: int = 32):
    """
    Create train and validation dataloaders from Shakespeare text.
    
    Args:
        file_path: Path to shakespeare.txt
        batch_size: Batch size (default 32)
        train_split: Fraction of data for training
        seq_length: Sequence length (default 32)
    
    Returns:
        train_loader, val_loader, tokenizer
    """
    # Read text file
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Create and build tokenizer
    tokenizer = WordTokenizer(vocab_size=4000)
    tokenizer.build_vocabulary(text)
    
    # Create dataset
    dataset = ShakespeareDataset(text, tokenizer, seq_length=seq_length)
    
    # Split into train and validation
    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader, tokenizer


if __name__ == "__main__":
    # Test the pipeline
    import os
    
    # Check if shakespeare.txt exists
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'shakespeare.txt')
    
    if os.path.exists(data_path):
        train_loader, val_loader, tokenizer = create_dataloaders(data_path)
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Vocab size: {len(tokenizer.word2idx)}")
        
        # Test a batch
        for x, y in train_loader:
            print(f"Input shape: {x.shape}, Target shape: {y.shape}")
            break
    else:
        print(f"Please place shakespeare.txt in {data_path}")
