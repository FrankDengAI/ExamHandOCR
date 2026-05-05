"""
ViT-OCR (Vision Transformer for OCR) implementation.

Based on Diaz et al. 2021 "Rethinking Text Line Recognition Models".

This model applies a pure transformer approach to handwriting recognition:
- Vision Transformer (ViT) encoder for image understanding
- Autoregressive decoder for text generation

Key difference from convolutional approaches:
- Global receptive field from first layer via self-attention
- No inductive bias of locality (must learn spatial relationships)
- Better long-range dependency modeling for context-conditioned recognition

ExamHandOCR Results (Table 1, Section 7.3):
    CER: 7.23%, WER: 12.08%, ESA-CER: 9.74%, RI: 0.61

ViT-OCR achieves strong results, second only to TrOCR + SSL variant.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


class PatchEmbedding(nn.Module):
    """
    Patch embedding layer for Vision Transformer.
    
    Divides the input image into non-overlapping patches and projects
    each patch to the embedding dimension. This is the first step of ViT,
    converting from image space to sequence space.
    
    For handwritten text (384 x 128) with patch size 16:
    - Number of patches: (384/16) * (128/16) = 24 * 8 = 192 patches
    - Each patch: 16 x 16 = 256 pixels
    - Patch embedding: 256 -> embed_dim
    
    Args:
        img_size: Input image size (width, height)
        patch_size: Size of each square patch
        in_channels: Number of input channels (1 for grayscale)
        embed_dim: Embedding dimension
    """
    
    def __init__(
        self,
        img_size: Tuple[int, int] = (384, 128),
        patch_size: int = 16,
        in_channels: int = 1,
        embed_dim: int = 768,
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches_w = img_size[0] // patch_size
        self.num_patches_h = img_size[1] // patch_size
        self.num_patches = self.num_patches_w * self.num_patches_h
        
        # Patch projection via convolution
        # Kernel size = patch size, stride = patch size (non-overlapping)
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        
        # Positional embedding (learned, not sinusoidal)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(0.1)
        
        # Initialize
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert image to patch embeddings.
        
        Args:
            x: Input images (B, C, H, W)
        
        Returns:
            Patch embeddings (B, num_patches, embed_dim)
        """
        B, C, H, W = x.shape
        
        # Extract patches via convolution
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        
        # Flatten spatial dimensions
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention module.
    
    Standard transformer attention with multiple heads for parallel
    attention computation at different representation subspaces.
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    
    Args:
        dim: Input dimension
        num_heads: Number of attention heads
        qkv_bias: Whether to use bias in QKV projections
        attn_drop: Attention dropout probability
        proj_drop: Projection dropout probability
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5  # Scale factor for stability
        
        # Combined QKV projection (more efficient than separate)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        # Dropout for regularization
        self.attn_drop = nn.Dropout(attn_drop)
        
        # Output projection
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Multi-head self-attention forward pass.
        
        Args:
            x: Input (B, N, C)
            mask: Optional attention mask
        
        Returns:
            Attention output (B, N, C)
        """
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        
        # Apply mask if provided
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        # Softmax normalization
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class TransformerBlock(nn.Module):
    """
    Standard transformer encoder block.
    
    Consists of:
    1. Multi-head self-attention
    2. Feed-forward network (MLP)
    3. Layer normalization and residual connections
    
    Pre-norm architecture (more stable for deep networks):
    - LayerNorm before attention/FFN
    - Residual connection around each sub-layer
    
    Args:
        dim: Input dimension
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim ratio (dim * mlp_ratio)
        qkv_bias: Whether to use bias in QKV
        drop: Dropout probability
        attn_drop: Attention dropout probability
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        
        # Pre-norm: LayerNorm before attention
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(
            dim, num_heads, qkv_bias, attn_drop, drop,
        )
        
        # Pre-norm: LayerNorm before FFN
        self.norm2 = nn.LayerNorm(dim)
        
        # MLP (Feed-forward network)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),  # GELU activation (standard for transformers)
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward with pre-norm and residuals."""
        # Attention block with residual
        x = x + self.attn(self.norm1(x), mask)
        
        # MLP block with residual
        x = x + self.mlp(self.norm2(x))
        
        return x


class ViTEncoder(nn.Module):
    """
    Vision Transformer encoder.
    
    Standard ViT architecture with patch embedding followed by
    a stack of transformer blocks.
    
    Args:
        img_size: Input image size
        patch_size: Patch size
        in_channels: Number of input channels
        embed_dim: Embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP ratio
        drop_rate: Dropout probability
    """
    
    def __init__(
        self,
        img_size: Tuple[int, int] = (384, 128),
        patch_size: int = 16,
        in_channels: int = 1,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim,
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim, num_heads, mlp_ratio,
                drop=drop_rate, attn_drop=drop_rate,
            )
            for _ in range(depth)
        ])
        
        # Final layer normalization
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode image with Vision Transformer.
        
        Args:
            x: Input images (B, C, H, W)
        
        Returns:
            Encoded features (B, num_patches, embed_dim)
        """
        # Patch embedding
        x = self.patch_embed(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final normalization
        x = self.norm(x)
        
        return x


class AutoregressiveDecoder(nn.Module):
    """
    Autoregressive decoder for text generation.
    
    Uses transformer decoder architecture with:
    - Token embeddings
    - Causal (autoregressive) self-attention
    - Cross-attention to encoder outputs
    - Position-wise FFN
    
    The causal mask ensures the model can only attend to previous tokens,
    maintaining the autoregressive property for generation.
    
    Args:
        vocab_size: Size of output vocabulary
        embed_dim: Embedding dimension
        num_layers: Number of decoder layers
        num_heads: Number of attention heads
        max_length: Maximum sequence length
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        max_length: int = 512,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_length = max_length
        
        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        
        # Learned positional embeddings (different from sinusoidal)
        self.pos_embed = nn.Embedding(max_length, embed_dim)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection to vocabulary
        self.output_proj = nn.Linear(embed_dim, vocab_size)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embeddings."""
        nn.init.normal_(self.token_embed.weight, mean=0, std=0.02)
        nn.init.normal_(self.pos_embed.weight, mean=0, std=0.02)
    
    def forward(
        self,
        memory: torch.Tensor,
        tgt_tokens: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode with cross-attention to encoder memory.
        
        Args:
            memory: Encoder outputs (B, N, embed_dim)
            tgt_tokens: Target tokens for teacher forcing (B, seq_len)
            tgt_mask: Causal mask for autoregressive property
        
        Returns:
            Output logits (B, seq_len, vocab_size)
        """
        B = memory.size(0)
        device = memory.device
        
        # If no target tokens provided, we need to generate
        # For simplicity, we assume tgt_tokens is always provided in training
        if tgt_tokens is None:
            # This would be the inference/generation path
            raise NotImplementedError("Generation without tgt_tokens not implemented")
        
        seq_len = tgt_tokens.size(1)
        
        # Token embeddings
        tgt_embed = self.token_embed(tgt_tokens) * math.sqrt(self.embed_dim)
        
        # Positional embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)
        pos_embed = self.pos_embed(positions)
        
        # Combined embeddings
        tgt = tgt_embed + pos_embed
        
        # Create causal mask if not provided
        if tgt_mask is None:
            tgt_mask = self.generate_causal_mask(seq_len, device)
        
        # Decode with cross-attention to encoder memory
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        
        # Project to vocabulary
        logits = self.output_proj(output)
        
        return logits
    
    def generate_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """
        Generate causal (autoregressive) attention mask.
        
        The mask prevents attending to future positions, ensuring
        the model only uses past information for prediction.
        
        Mask[i, j] = -inf if j > i (future positions)
        Mask[i, j] = 0 if j <= i (past and current positions)
        
        Args:
            size: Sequence length
            device: Device to create mask on
        
        Returns:
            Causal mask (size, size)
        """
        # Upper triangular matrix (excluding diagonal)
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        # Convert to attention mask (0 = attend, -inf = ignore)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def generate(
        self,
        memory: torch.Tensor,
        bos_token_id: int = 0,
        eos_token_id: int = 1,
        max_length: Optional[int] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Autoregressive generation with encoder memory.
        
        Args:
            memory: Encoder outputs (B, N, embed_dim)
            bos_token_id: Beginning of sequence token ID
            eos_token_id: End of sequence token ID
            max_length: Maximum generation length
            temperature: Sampling temperature (1.0 = greedy)
        
        Returns:
            Generated token IDs (B, seq_len)
        """
        max_length = max_length or self.max_length
        B = memory.size(0)
        device = memory.device
        
        # Start with BOS token
        generated = torch.full((B, 1), bos_token_id, dtype=torch.long, device=device)
        
        # Generate one token at a time
        for _ in range(max_length - 1):
            # Get logits for current sequence
            tgt_mask = self.generate_causal_mask(generated.size(1), device)
            
            positions = torch.arange(generated.size(1), device=device).unsqueeze(0).expand(B, -1)
            tgt_embed = self.token_embed(generated) * math.sqrt(self.embed_dim)
            pos_embed = self.pos_embed(positions)
            tgt = tgt_embed + pos_embed
            
            output = self.decoder(tgt, memory, tgt_mask=tgt_mask)
            logits = self.output_proj(output)
            
            # Get next token logits (last position)
            next_logits = logits[:, -1, :] / temperature
            
            # Greedy decoding (could use sampling instead)
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            
            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for EOS
            if (next_token == eos_token_id).all():
                break
        
        return generated


class ViTOCR(nn.Module):
    """
    Complete ViT-OCR model.
    
    Combines Vision Transformer encoder with autoregressive decoder
    for end-to-end handwritten text recognition.
    
    Architecture Summary:
    1. Patch Embedding: Image -> Patches -> Embeddings
    2. ViT Encoder: Self-attention across all patches
    3. Autoregressive Decoder: Cross-attention to encoder, causal self-attention
    4. Output: Character class predictions
    
    Training:
    - Teacher forcing with ground truth tokens
    - Cross-entropy loss on character prediction
    
    Inference:
    - Beam search or greedy decoding
    
    Paper Configuration (Section 7.1):
        Patch size: 16
        Encoder depth: 12 layers
        Decoder depth: 6 layers
        Hidden dimension: 768
        Attention heads: 12
    
    ExamHandOCR Results (Table 1):
        CER: 7.23%
        WER: 12.08%
        ESA-CER: 9.74%
        RI: 0.61
    
    Args:
        num_classes: Number of character classes
        img_size: Input image size (width, height)
        patch_size: Patch size for ViT
        embed_dim: Embedding dimension
        encoder_depth: Number of encoder layers
        decoder_depth: Number of decoder layers
        num_heads: Number of attention heads
        max_length: Maximum sequence length
    """
    
    def __init__(
        self,
        num_classes: int,
        img_size: Tuple[int, int] = (384, 128),
        patch_size: int = 16,
        embed_dim: int = 768,
        encoder_depth: int = 12,
        decoder_depth: int = 6,
        num_heads: int = 12,
        max_length: int = 512,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.max_length = max_length
        
        # Vision Transformer encoder
        self.encoder = ViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=1,  # Grayscale
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=num_heads,
        )
        
        # Autoregressive decoder
        self.decoder = AutoregressiveDecoder(
            vocab_size=num_classes,
            embed_dim=embed_dim,
            num_layers=decoder_depth,
            num_heads=num_heads,
            max_length=max_length,
        )
        
        # Special tokens
        self.bos_token_id = 0
        self.eos_token_id = 1
        self.pad_token_id = 2
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with truncated normal."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training or inference.
        
        Args:
            x: Input images (B, C, H, W)
            targets: Target token IDs (B, seq_len), optional
        
        Returns:
            Dictionary with loss and/or logits
        """
        # Encode image with ViT
        memory = self.encoder(x)  # (B, N, D)
        
        # Decode
        if targets is not None:
            # Teacher forcing during training
            # Input: all tokens except last
            # Target: all tokens except first (shifted right)
            logits = self.decoder(memory, targets[:, :-1])
            
            # Compute cross-entropy loss
            loss = F.cross_entropy(
                logits.reshape(-1, self.num_classes),
                targets[:, 1:].reshape(-1),
                ignore_index=self.pad_token_id,
            )
            
            return {
                'loss': loss,
                'logits': logits,
            }
        else:
            # Inference - generate autoregressively
            predictions = self.decoder.generate(
                memory,
                bos_token_id=self.bos_token_id,
                eos_token_id=self.eos_token_id,
                max_length=self.max_length,
            )
            return {'predictions': predictions}
    
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
            if 'predictions' in outputs:
                return outputs['predictions']
            else:
                return outputs['logits'].argmax(dim=-1)


def build_vit_ocr(
    num_classes: int,
    img_size: Tuple[int, int] = (384, 128),
    **kwargs
) -> ViTOCR:
    """
    Factory function for ViT-OCR.
    
    Args:
        num_classes: Number of character classes
        img_size: Input image size
        **kwargs: Additional arguments
    
    Returns:
        ViT-OCR model instance
    """
    return ViTOCR(num_classes=num_classes, img_size=img_size, **kwargs)
