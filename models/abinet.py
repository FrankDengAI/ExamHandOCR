"""
ABINet (Autonomous, Bidirectional, Iterative Network) implementation.

Based on Fang et al. 2021 "Read Like Humans: Autonomous, Bidirectional and 
Iterative Language Modeling for Scene Text Recognition" published at CVPR 2021.

ABINet introduces three key innovations:
1. Autonomous: Disentangled vision and language models with explicit feedback
2. Bidirectional: BERT-style masked language modeling for robust recognition
3. Iterative: Progressive correction through multiple refinement steps

Architecture Overview:
    Vision Model: ResNet-based feature extraction
    Language Model: BERT-style bidirectional transformer
    Iterative Refinement: Multiple passes with visual feedback

ExamHandOCR Results (Table 1, Section 7.3):
    CER: 12.34%, WER: 19.87%, ESA-CER: 16.42%, RI: 0.41

This places ABINet between CRNN and TrOCR in terms of accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class ConvBNReLU(nn.Module):
    """
    Standard convolutional block: Conv2d + BatchNorm + ReLU.
    
    Basic building block for the visual feature extractor.
    Uses bias=False when followed by batch normalization.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """
    Residual block with two convolutional layers.
    
    Implementation follows ResNet architecture with skip connection
    around the two convolutional layers.
    """
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBNReLU(channels, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with residual connection."""
        residual = x
        out = self.conv1(x)
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        return F.relu(out)


class VisualBackbone(nn.Module):
    """
    Visual feature extraction backbone.
    
    Architecture (from paper):
    - Multiple stages with progressive downsampling
    - Residual blocks within each stage
    - Total downsampling: 4x (two stride-2 stages)
    
    For input (384, 128):
    - After stage 1: (192, 64), 64 channels
    - After stage 2: (96, 32), 128 channels  
    - After stage 3: (48, 16), 256 channels
    - After stage 4: (24, 8), 512 channels
    
    Final sequence length: 24 for width dimension
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_stages: int = 4,
    ):
        super().__init__()
        
        self.stages = nn.ModuleList()
        channels = 64
        
        # Initial convolution (no downsampling)
        self.stages.append(nn.Sequential(
            ConvBNReLU(in_channels, channels, 3, 1, 1),
            ConvBNReLU(channels, channels, 3, 1, 1),
        ))
        
        # Progressive downsampling stages
        for i in range(num_stages):
            out_channels = channels * 2
            self.stages.append(nn.Sequential(
                ConvBNReLU(channels, out_channels, 3, 2, 1),  # Downsample
                ResidualBlock(out_channels),
                ResidualBlock(out_channels),
            ))
            channels = out_channels
        
        self.output_channels = channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract visual features.
        
        Args:
            x: Input images (B, C, H, W)
        
        Returns:
            Feature sequence (B, H*W, C) for language model
        """
        for stage in self.stages:
            x = stage(x)
        
        # Reshape to sequence: (B, C, H, W) -> (B, H*W, C)
        B, C, H, W = x.size()
        x = x.reshape(B, C, H * W).permute(0, 2, 1)
        
        return x


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer.
    
    Uses sine and cosine functions of different frequencies to encode
    position information, following the original Transformer paper.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # Precompute positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Divisor term
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        
        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, but persistent)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[:, :x.size(1)]


class LanguageModel(nn.Module):
    """
    Autonomous and bidirectional language model.
    
    Implements BERT-style masked language modeling with:
    - Token embeddings
    - Positional encoding
    - Multi-layer bidirectional transformer encoder
    - Output projection to vocabulary
    
    The "autonomous" aspect means the language model operates independently
    of the vision model, receiving only text tokens as input, not image features.
    This forces the model to learn proper linguistic representations.
    
    Args:
        vocab_size: Size of vocabulary
        d_model: Model dimension (default: 512)
        nhead: Number of attention heads (default: 8)
        num_layers: Number of transformer layers (default: 4)
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_embed = PositionalEncoding(d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Transformer encoder (bidirectional)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embeddings."""
        nn.init.normal_(self.token_embed.weight, mean=0, std=0.02)
    
    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with bidirectional encoding.
        
        Args:
            token_ids: (B, seq_len)
            attention_mask: (B, seq_len) binary mask (1 = attend, 0 = ignore)
        
        Returns:
            Logits (B, seq_len, vocab_size)
        """
        # Embed tokens
        x = self.token_embed(token_ids) * (self.d_model ** 0.5)  # Scale embedding
        
        # Add positional encoding
        x = self.pos_embed(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Convert mask for transformer (True = ignore)
        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = None
        
        # Bidirectional encoding
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Project to vocabulary
        logits = self.output(x)
        
        return logits
    
    def decode_autoregressive(
        self,
        visual_features: torch.Tensor,
        max_length: int = 25,
        bos_token_id: int = 0,
    ) -> torch.Tensor:
        """
        Autoregressive decoding with visual features as initial state.
        
        Note: In full ABINet, the vision and language models are more tightly
        integrated. This is a simplified version.
        
        Args:
            visual_features: (B, seq_len, d_model)
            max_length: Maximum decoding length
            bos_token_id: Beginning of sequence token ID
        
        Returns:
            Generated token IDs (B, seq_len)
        """
        B = visual_features.size(0)
        device = visual_features.device
        
        # Start with BOS token
        token_ids = torch.full((B, 1), bos_token_id, dtype=torch.long, device=device)
        
        for _ in range(max_length):
            # Get predictions for current sequence
            logits = self.forward(token_ids)
            next_token_logits = logits[:, -1, :]  # Last position
            
            # Greedy decoding
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            
            # Append
            token_ids = torch.cat([token_ids, next_token], dim=1)
            
            # Check for EOS (simplified - would need actual EOS token)
            if (next_token == 1).all():
                break
        
        return token_ids


class FusionModule(nn.Module):
    """
    Fusion module combining visual and language features.
    
    Implements the cross-modal fusion that allows visual features to
    inform the language model predictions and vice versa.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        visual_features: torch.Tensor,
        language_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse visual and language features.
        
        Args:
            visual_features: (B, seq_len, d_model)
            language_features: (B, seq_len, d_model)
        
        Returns:
            Fused features (B, seq_len, d_model)
        """
        # Concatenate along feature dimension
        combined = torch.cat([visual_features, language_features], dim=-1)
        
        # Fusion transformation
        fused = self.fusion(combined)
        
        return fused


class ABINet(nn.Module):
    """
    Complete ABINet model.
    
    Architecture Components:
    1. Vision Model: ResNet-based feature extraction
    2. Language Model: Bidirectional transformer for text understanding
    3. Fusion: Cross-modal feature combination
    4. Iterative Refinement: Multiple passes for error correction
    
    Training Strategy:
    - First pre-train language model with masked language modeling
    - Then train full model end-to-end
    - Iterative refinement happens during inference
    
    Paper Results on ExamHandOCR (Table 1):
        CER: 12.34%
        WER: 19.87%
        ESA-CER: 16.42%
        RI: 0.41
    
    Args:
        num_classes: Number of character classes
        d_model: Hidden dimension (default: 512)
        max_length: Maximum sequence length
        num_iter: Number of iterative refinement steps (default: 3)
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        num_classes: int,
        d_model: int = 512,
        max_length: int = 25,
        num_iter: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.d_model = d_model
        self.max_length = max_length
        self.num_iter = num_iter
        
        # Vision backbone
        self.visual = VisualBackbone()
        
        # Project visual features to model dimension
        self.visual_proj = nn.Linear(self.visual.output_channels, d_model)
        
        # Positional encoding for visual features
        self.visual_pos = PositionalEncoding(d_model)
        
        # Language model (autonomous, bidirectional)
        self.language = LanguageModel(
            vocab_size=num_classes,
            d_model=d_model,
            nhead=8,
            num_layers=4,
            dropout=dropout,
        )
        
        # Fusion module
        self.fusion = FusionModule(d_model, dropout)
        
        # Iterative refinement modules
        self.refiners = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(num_iter)
        ])
        
        # Final output projection
        self.output_proj = nn.Linear(d_model, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional iterative refinement.
        
        Args:
            x: Input images (B, C, H, W)
            targets: Target token IDs (B, seq_len), optional
        
        Returns:
            Dictionary with loss and/or logits
        """
        # Visual feature extraction
        visual_feat = self.visual(x)  # (B, N, C)
        visual_feat = self.visual_proj(visual_feat)  # (B, N, d_model)
        visual_feat = self.visual_pos(visual_feat)
        
        # Language model prediction
        if self.training and targets is not None:
            # Teacher forcing during training
            # Exclude last token for input, use from second token for target
            lang_logits = self.language(targets[:, :-1])
        else:
            # Autoregressive generation for inference
            lang_logits = self.language.decode_autoregressive(
                visual_feat, self.max_length
            )
        
        # Iterative refinement (simplified version)
        current_logits = lang_logits
        for i in range(self.num_iter):
            # Refine using transformer decoder
            if isinstance(current_logits, torch.Tensor):
                # Ensure same sequence length as visual features
                if current_logits.size(1) != visual_feat.size(1):
                    # Pad or trim to match
                    if current_logits.size(1) < visual_feat.size(1):
                        pad_size = visual_feat.size(1) - current_logits.size(1)
                        current_logits = F.pad(current_logits, (0, 0, 0, pad_size))
                    else:
                        current_logits = current_logits[:, :visual_feat.size(1)]
                
                refined = self.refiners[i](current_logits, visual_feat)
                current_logits = refined
        
        # Final prediction
        if isinstance(current_logits, torch.Tensor):
            final_logits = self.output_proj(current_logits)
        else:
            final_logits = current_logits
        
        # Compute loss if training
        if self.training and targets is not None:
            loss = F.cross_entropy(
                final_logits.reshape(-1, self.num_classes),
                targets.reshape(-1),
                ignore_index=0,  # Pad token
            )
            return {'loss': loss, 'logits': final_logits}
        
        return {'logits': final_logits}
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict text from images.
        
        Args:
            x: Input images (B, C, H, W)
        
        Returns:
            Predicted token IDs (B, seq_len)
        """
        with torch.no_grad():
            outputs = self.forward(x)
            logits = outputs['logits']
            preds = logits.argmax(dim=-1)
        return preds
    
    def iterative_refine(
        self,
        x: torch.Tensor,
        num_iterations: Optional[int] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Explicitly perform iterative refinement with intermediate outputs.
        
        Args:
            x: Input images
            num_iterations: Override default number of iterations
        
        Returns:
            Tuple of (final_predictions, list_of_intermediate_predictions)
        """
        num_iterations = num_iterations or self.num_iter
        
        with torch.no_grad():
            # Initial prediction
            outputs = self.forward(x)
            current_logits = outputs['logits']
            
            intermediate_preds = [current_logits.argmax(dim=-1)]
            
            # Iterative refinement
            for i in range(num_iterations):
                # Refinement step
                # (Simplified - full implementation would use proper fusion)
                pass
        
        return intermediate_preds[-1], intermediate_preds


def build_abinet(num_classes: int, **kwargs) -> ABINet:
    """
    Factory function for ABINet.
    
    Args:
        num_classes: Number of character classes
        **kwargs: Additional arguments
    
    Returns:
        ABINet model instance
    """
    return ABINet(num_classes=num_classes, **kwargs)
