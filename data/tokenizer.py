"""
Tokenizer for ExamHandOCR supporting mixed Chinese characters and LaTeX math.

Handles the unique challenge of recognizing heterogeneous content where Chinese 
prose is interleaved with mathematical expressions delimited by \( ... \).

The tokenizer implements the segmentation strategy required for ESA-CER computation,
where mathematical tokens are differentially weighted to align OCR errors with 
grading consequences (Section 5.1).
"""

import re
import json
from typing import List, Dict, Optional, Tuple, Union
from collections import Counter

import torch
import numpy as np


class ExamHandOCRTokenizer:
    """
    Character-level tokenizer for handwritten text with LaTeX math.
    
    ExamHandOCR contains mixed-modality content: Chinese prose, English text,
    and mathematical expressions. This tokenizer handles all three modalities
    and provides specialized support for ESA-CER evaluation.
    
    Key features:
    1. Character-level tokenization for Chinese (each character is a token)
    2. LaTeX math expression detection and special handling
    3. Special tokens for math mode (MATH_START, MATH_END)
    4. Vocabulary building from training data
    5. Support for ESA-CER computation with math token identification
    
    Example usage:
        >>> tokenizer = ExamHandOCRTokenizer(max_length=512)
        >>> text = "解方程 \\(x^2 + 2x + 1 = 0\\) 得"
        >>> tokens = tokenizer.tokenize(text)
        >>> ids = tokenizer.encode(text)
    
    Args:
        vocab_file: Path to vocabulary JSON file (optional)
        max_length: Maximum sequence length
        special_tokens: Dictionary of special token names to values
    """
    
    # Special token definitions
    PAD_TOKEN = '<pad>'           # Padding for batching
    SOS_TOKEN = '<sos>'           # Start of sequence
    EOS_TOKEN = '<eos>'           # End of sequence
    UNK_TOKEN = '<unk>'           # Unknown character
    MATH_START = '<math>'         # Start of math expression
    MATH_END = '</math>'          # End of math expression
    SPACE_TOKEN = '<space>'       # Explicit space marker
    
    # LaTeX math delimiters
    MATH_DELIMITERS = [
        (r'\\\(', r'\\\)'),      # Inline math: \( ... \)
        (r'\\\[', r'\\\]'),      # Display math: \[ ... \]
        (r'\$\$', r'\$\$'),        # Display math: $$ ... $$
    ]
    
    def __init__(
        self,
        vocab_file: Optional[str] = None,
        max_length: int = 512,
        special_tokens: Optional[Dict[str, str]] = None,
    ):
        self.max_length = max_length
        
        # Initialize special tokens list
        self.special_tokens = [
            self.PAD_TOKEN,
            self.SOS_TOKEN,
            self.EOS_TOKEN,
            self.UNK_TOKEN,
            self.MATH_START,
            self.MATH_END,
            self.SPACE_TOKEN,
        ]
        
        # Load or build vocabulary
        if vocab_file is not None:
            self.vocab = self._load_vocab(vocab_file)
        else:
            self.vocab = self._build_default_vocab()
        
        # Create token to ID and ID to token mappings
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        
        # Cache special token IDs for fast access
        self.pad_id = self.token_to_id[self.PAD_TOKEN]
        self.sos_id = self.token_to_id[self.SOS_TOKEN]
        self.eos_id = self.token_to_id[self.EOS_TOKEN]
        self.unk_id = self.token_to_id[self.UNK_TOKEN]
        self.math_start_id = self.token_to_id[self.MATH_START]
        self.math_end_id = self.token_to_id[self.MATH_END]
        
        # Define math token patterns for ESA-CER
        self._init_math_patterns()
        
        print(f"[Tokenizer] Initialized with vocabulary size: {len(self.vocab)}")
    
    def _init_math_patterns(self):
        """Initialize patterns for identifying math-related tokens."""
        # Mathematical symbols and operators
        self.math_symbols = set([
            # Basic operators
            '+', '-', '*', '/', '=', '≠', '≈', '<', '>', '≤', '≥', '≡', '±',
            # Superscript/subscript
            '^', '_',
            # Greek letters (lowercase)
            'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ',
            'ν', 'ξ', 'ο', 'π', 'ρ', 'σ', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω',
            # Greek letters (uppercase)
            'Γ', 'Δ', 'Θ', 'Λ', 'Ξ', 'Π', 'Σ', 'Φ', 'Ψ', 'Ω',
            # Math notation
            '√', '∫', '∑', '∏', '∂', '·', '×', '÷', '∞', '∇',
            '∈', '∉', '∪', '∩', '⊂', '⊃', '⊆', '⊇',
            # Arrows
            '→', '←', '↑', '↓', '⇒', '⇐', '⇑', '⇓',
            # Delimiters
            '(', ')', '[', ']', '{', '}',
        ])
        
        # LaTeX commands commonly found in handwritten math
        self.latex_commands = [
            'frac', 'sqrt', 'sum', 'int', 'prod', 'lim', 'infty',
            'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'theta',
            'lambda', 'mu', 'pi', 'sigma', 'omega', 'Delta', 'Sigma',
            'left', 'right', 'cdot', 'times', 'div', 'pm', 'mp',
            'leq', 'geq', 'neq', 'approx', 'equiv', 'sim',
            'rightarrow', 'leftarrow', 'Rightarrow', 'Leftarrow',
        ]
    
    def _file_exists(self, path: str) -> bool:
        """Check if file exists."""
        import os
        return os.path.exists(path)
    
    def _build_default_vocab(self) -> List[str]:
        """
        Build default vocabulary covering common cases.
        
        In practice, vocabulary should be built from training data using
        build_vocab_from_data() method.
        """
        vocab = list(self.special_tokens)
        
        # Common Chinese characters (simplified Chinese, high frequency)
        # These are the most frequent characters in educational contexts
        chinese_common = (
            "的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成"
            "会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着"
            "等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把"
            "性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新"
            "线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公"
            "无系军很情最何发些物解几那经者知到完地事性然问路等什观提解式教常入决"
            "解确等许传复全反更解特各他上"
        )
        vocab.extend(list(chinese_common))
        
        # English letters (both cases)
        vocab.extend([chr(i) for i in range(ord('a'), ord('z') + 1)])
        vocab.extend([chr(i) for i in range(ord('A'), ord('Z') + 1)])
        
        # Digits
        vocab.extend([str(i) for i in range(10)])
        
        # Common punctuation
        punctuation = """.,!?;:''""()[]{}+-*/=<>_%^&|~`@#$\\ \t\n"""
        vocab.extend(list(punctuation))
        
        # Common LaTeX math commands (as individual tokens)
        for cmd in self.latex_commands:
            vocab.append(f'\\{cmd}')
        
        # Mathematical symbols
        vocab.extend(list(self.math_symbols))
        
        return vocab
    
    def _load_vocab(self, vocab_file: str) -> List[str]:
        """Load vocabulary from JSON file."""
        with open(vocab_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            if isinstance(data, dict) and 'vocab' in data:
                return data['vocab']
            elif isinstance(data, list):
                return data
            else:
                print(f"[Warning] Invalid vocab format, using default")
                return self._build_default_vocab()
    
    def save_vocab(self, vocab_file: str):
        """Save vocabulary to JSON file."""
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump({
                'vocab': self.vocab,
                'token_to_id': self.token_to_id,
                'special_tokens': {
                    'pad': self.pad_id,
                    'sos': self.sos_id,
                    'eos': self.eos_id,
                    'unk': self.unk_id,
                },
                'vocab_size': len(self.vocab),
            }, f, ensure_ascii=False, indent=2)
        
        print(f"[Tokenizer] Vocabulary saved to {vocab_file}")
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into character-level tokens with LaTeX math handling.
        
        This method identifies LaTeX math expressions (delimited by \( ... \))
        and marks them with special tokens for ESA-CER computation.
        
        Args:
            text: Input text with possible LaTeX math
        
        Returns:
            List of token strings
        """
        tokens = []
        i = 0
        
        while i < len(text):
            # Check for LaTeX math delimiters
            found_math = False
            
            for open_delim, close_delim in self.MATH_DELIMITERS:
                pattern = f"{open_delim}(.*?){close_delim}"
                match = re.match(pattern, text[i:], re.DOTALL)
                
                if match:
                    # Add math start token
                    tokens.append(self.MATH_START)
                    
                    # Tokenize content inside math
                    math_content = match.group(1)
                    math_tokens = self._tokenize_math_content(math_content)
                    tokens.extend(math_tokens)
                    
                    # Add math end token
                    tokens.append(self.MATH_END)
                    
                    i += len(match.group(0))
                    found_math = True
                    break
            
            if found_math:
                continue
            
            # Regular character tokenization
            char = text[i]
            
            # Handle whitespace
            if char.isspace():
                tokens.append(self.SPACE_TOKEN)
            else:
                tokens.append(char)
            
            i += 1
        
        return tokens
    
    def _tokenize_math_content(self, math_text: str) -> List[str]:
        """
        Tokenize content inside LaTeX math expressions.
        
        Math content requires special handling for:
        - LaTeX commands (\frac, \sqrt, etc.)
        - Superscripts and subscripts (^ and _)
        - Grouping braces
        - Mathematical symbols
        
        Args:
            math_text: Text inside \( ... \)
        
        Returns:
            List of math tokens
        """
        tokens = []
        i = 0
        
        while i < len(math_text):
            char = math_text[i]
            
            # Check for LaTeX command
            if char == '\\' and i + 1 < len(math_text):
                # Extract command name
                j = i + 1
                while j < len(math_text) and math_text[j].isalpha():
                    j += 1
                
                if j > i + 1:
                    cmd = math_text[i:j]
                    tokens.append(f'\\{cmd}')
                    i = j
                    continue
            
            # Handle special math characters
            if char in ['^', '_', '{', '}', '&', '\\']:
                tokens.append(char)
            elif char.isspace():
                tokens.append(self.SPACE_TOKEN)
            else:
                tokens.append(char)
            
            i += 1
        
        return tokens
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text string
        
        Returns:
            List of integer token IDs, including SOS and EOS
        """
        tokens = self.tokenize(text)
        
        # Convert tokens to IDs
        ids = []
        for token in tokens:
            token_id = self.token_to_id.get(token, self.unk_id)
            ids.append(token_id)
        
        # Truncate if necessary, reserving space for SOS and EOS
        max_content_length = self.max_length - 2
        if len(ids) > max_content_length:
            ids = ids[:max_content_length]
        
        # Add SOS and EOS
        ids = [self.sos_id] + ids + [self.eos_id]
        
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            ids: List of token IDs
            skip_special_tokens: Whether to remove special tokens from output
        
        Returns:
            Decoded text string
        """
        tokens = []
        
        for id in ids:
            # Skip padding
            if id == self.pad_id:
                continue
            
            token = self.id_to_token.get(id, self.UNK_TOKEN)
            
            if skip_special_tokens:
                # Handle special token conversion
                if token == self.MATH_START:
                    tokens.append('\\(')
                elif token == self.MATH_END:
                    tokens.append('\\)')
                elif token == self.SPACE_TOKEN:
                    tokens.append(' ')
                elif token in self.special_tokens:
                    continue
                else:
                    tokens.append(token)
            else:
                tokens.append(token)
        
        return ''.join(tokens)
    
    def batch_encode(
        self, 
        texts: List[str], 
        padding: bool = True,
        return_tensors: str = 'pt'
    ) -> torch.Tensor:
        """
        Batch encode multiple texts.
        
        Args:
            texts: List of text strings
            padding: Whether to pad to max_length
            return_tensors: Return format ('pt' for PyTorch tensor)
        
        Returns:
            Tensor of shape (batch_size, seq_len)
        """
        # Encode all texts
        batch_ids = [self.encode(text) for text in texts]
        
        if padding:
            # Determine max length in batch
            max_len = max(len(ids) for ids in batch_ids)
            max_len = min(max_len, self.max_length)
            
            # Pad all sequences
            padded_ids = []
            for ids in batch_ids:
                if len(ids) < max_len:
                    # Pad with PAD token
                    ids = ids + [self.pad_id] * (max_len - len(ids))
                else:
                    # Truncate
                    ids = ids[:max_len]
                padded_ids.append(ids)
            
            batch_ids = padded_ids
        
        if return_tensors == 'pt':
            return torch.tensor(batch_ids, dtype=torch.long)
        else:
            return batch_ids
    
    def batch_decode(
        self, 
        batch_ids: Union[torch.Tensor, List[List[int]]],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """
        Batch decode token IDs to texts.
        
        Args:
            batch_ids: Tensor or list of token ID sequences
            skip_special_tokens: Whether to remove special tokens
        
        Returns:
            List of decoded text strings
        """
        if isinstance(batch_ids, torch.Tensor):
            batch_ids = batch_ids.tolist()
        
        texts = []
        for ids in batch_ids:
            text = self.decode(ids, skip_special_tokens)
            texts.append(text)
        
        return texts
    
    def build_vocab_from_data(
        self, 
        texts: List[str], 
        min_freq: int = 2,
        vocab_file: Optional[str] = None,
    ):
        """
        Build vocabulary from training corpus.
        
        This is the recommended way to create vocabulary for ExamHandOCR,
        as it captures the character distribution of the specific domain.
        
        Args:
            texts: List of transcription texts from training set
            min_freq: Minimum frequency for inclusion in vocabulary
            vocab_file: Optional path to save vocabulary
        """
        print(f"[Tokenizer] Building vocabulary from {len(texts)} texts...")
        
        # Count all tokens
        counter = Counter()
        
        for text in texts:
            tokens = self.tokenize(text)
            counter.update(tokens)
        
        print(f"[Tokenizer] Found {len(counter)} unique tokens")
        
        # Build vocabulary starting with special tokens
        vocab = list(self.special_tokens)
        
        # Add tokens meeting frequency threshold
        for token, freq in counter.most_common():
            if freq >= min_freq and token not in self.special_tokens:
                vocab.append(token)
        
        self.vocab = vocab
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        
        # Update special token IDs
        self.pad_id = self.token_to_id[self.PAD_TOKEN]
        self.sos_id = self.token_to_id[self.SOS_TOKEN]
        self.eos_id = self.token_to_id[self.EOS_TOKEN]
        self.unk_id = self.token_to_id[self.UNK_TOKEN]
        self.math_start_id = self.token_to_id[self.MATH_START]
        self.math_end_id = self.token_to_id[self.MATH_END]
        
        print(f"[Tokenizer] Vocabulary built: {len(self.vocab)} tokens")
        print(f"  - Special tokens: {len(self.special_tokens)}")
        print(f"  - Regular tokens: {len(self.vocab) - len(self.special_tokens)}")
        
        if vocab_file:
            self.save_vocab(vocab_file)
    
    def is_math_token(self, token_id: int) -> bool:
        """
        Check if a token ID corresponds to a math-related token.
        
        This is used for ESA-CER computation where math tokens receive
        higher weight (α=3.0) due to their grading consequence (Section 5.1).
        
        Args:
            token_id: Token ID to check
        
        Returns:
            True if token is math-related, False otherwise
        """
        token = self.id_to_token.get(token_id, '')
        
        # Check if it's a math delimiter
        if token in [self.MATH_START, self.MATH_END]:
            return True
        
        # Check if it's a math symbol
        if token in self.math_symbols:
            return True
        
        # Check if it's a LaTeX command
        if token.startswith('\\') and len(token) > 1:
            cmd = token[1:]
            if cmd in self.latex_commands:
                return True
        
        # Check if it's a digit (digits in math mode are consequential)
        if token.isdigit():
            return True
        
        return False
    
    def identify_math_regions(self, token_ids: List[int]) -> List[Tuple[int, int]]:
        """
        Identify regions of math tokens in a sequence.
        
        Returns start and end indices of math regions (between MATH_START
        and MATH_END tokens). Used for visualization and ESA-CER weighting.
        
        Args:
            token_ids: Sequence of token IDs
        
        Returns:
            List of (start_idx, end_idx) tuples
        """
        regions = []
        in_math = False
        start_idx = -1
        
        for i, token_id in enumerate(token_ids):
            if token_id == self.math_start_id:
                in_math = True
                start_idx = i
            elif token_id == self.math_end_id:
                if in_math and start_idx != -1:
                    regions.append((start_idx, i))
                in_math = False
                start_idx = -1
        
        return regions
    
    def compute_sequence_weights(self, token_ids: List[int], alpha: float = 3.0) -> List[float]:
        """
        Compute per-token weights for ESA-CER calculation.
        
        Formula from Section 5.1:
            w_i = α if char_i is a math-token (inside \( ... \))
            w_i = 1 otherwise
        
        Args:
            token_ids: Sequence of token IDs
            alpha: Weight for mathematical tokens (default 3.0)
        
        Returns:
            List of weights (one per token)
        """
        weights = []
        in_math = False
        
        for token_id in token_ids:
            # Track math regions
            if token_id == self.math_start_id:
                in_math = True
                weights.append(alpha)  # Math start token is math-related
            elif token_id == self.math_end_id:
                in_math = False
                weights.append(alpha)  # Math end token is math-related
            elif in_math or self.is_math_token(token_id):
                weights.append(alpha)
            else:
                weights.append(1.0)
        
        return weights
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)
    
    def get_vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)
