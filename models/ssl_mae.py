"""
Masked Autoencoder (MAE) for Self-Supervised Learning pre-training.
Based on He et al. 2022 "Masked Autoencoders Are Scalable Vision Learners"
Implements the SSL pre-training described in Section 7.1 of the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PatchEmbedding(nn.Module):
    """Patch embedding for MAE."""
    
    def __init__(self, img_size=(384, 128), patch_size=16, in_channels=1, embed_dim=768):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[1] // patch_size) * (img_size[0] // patch_size)
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class TransformerEncoder(nn.Module):
    """Transformer encoder for MAE."""
    
    def __init__(self, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., dropout=0.1):
        super().__init__()
        
        self.pos_embed = nn.Parameter(torch.zeros(1, 5000, embed_dim))  # Max 5000 patches
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        self.norm = nn.LayerNorm(embed_dim)
        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (B, N, D) where N is number of visible patches
            mask: Optional mask
        """
        # Add positional embedding
        x = x + self.pos_embed[:, :x.size(1)]
        
        # Encode
        x = self.encoder(x, src_key_padding_mask=mask)
        x = self.norm(x)
        
        return x


class TransformerDecoder(nn.Module):
    """Lightweight decoder for MAE reconstruction."""
    
    def __init__(
        self,
        patch_size=16,
        num_patches=192,
        encoder_dim=768,
        decoder_dim=512,
        decoder_depth=8,
        num_heads=16,
        mlp_ratio=4.,
        dropout=0.1,
    ):
        super().__init__()
        
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.decoder_dim = decoder_dim
        
        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_dim))
        
        # Encoder to decoder projection
        self.encoder_to_decoder = nn.Linear(encoder_dim, decoder_dim)
        
        # Decoder transformer
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=num_heads,
            dim_feedforward=int(decoder_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_depth)
        
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        
        # Reconstruction head
        self.recon_head = nn.Linear(decoder_dim, patch_size * patch_size * 1)  # 1 channel
        
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, encoder_output, mask_indices, unmask_indices):
        """
        Args:
            encoder_output: (B, N_visible, encoder_dim)
            mask_indices: (B, N_masked) indices of masked patches
            unmask_indices: (B, N_visible) indices of unmasked patches
        
        Returns:
            recon: (B, N_total, patch_dim) reconstructed patches
        """
        B = encoder_output.size(0)
        N_total = self.num_patches
        N_masked = mask_indices.size(1)
        
        # Project encoder output
        x = self.encoder_to_decoder(encoder_output)  # (B, N_visible, decoder_dim)
        
        # Create mask tokens
        mask_tokens = self.mask_token.expand(B, N_masked, -1)  # (B, N_masked, decoder_dim)
        
        # Combine visible and mask tokens
        x_full = torch.zeros(B, N_total, self.decoder_dim, device=x.device)
        
        # Fill in visible tokens
        for i in range(B):
            x_full[i, unmask_indices[i]] = x[i]
            x_full[i, mask_indices[i]] = mask_tokens[i]
        
        # Add positional embedding
        x_full = x_full + self.pos_embed
        
        # Decode
        x_full = self.decoder(x_full)
        x_full = self.decoder_norm(x_full)
        
        # Reconstruct
        recon = self.recon_head(x_full)
        
        return recon


class MaskedAutoencoder(nn.Module):
    """
    Masked Autoencoder (MAE) for self-supervised pre-training.
    
    Paper configuration (Section 7.1):
    - Patch size: 16
    - Mask ratio: 0.75
    - Encoder: BEiT-base or ViT-base
    - Decoder: Lightweight transformer (512-dim, 8 layers)
    - Pre-training: 100 epochs on 3.15M unannotated images
    
    Args:
        img_size: Input image size (W, H)
        patch_size: Patch size for tokenization
        mask_ratio: Ratio of patches to mask
        embed_dim: Encoder embedding dimension
        encoder_depth: Number of encoder layers
        decoder_dim: Decoder embedding dimension
        decoder_depth: Number of decoder layers
        num_heads: Number of attention heads
    """
    
    def __init__(
        self,
        img_size=(384, 128),
        patch_size=16,
        mask_ratio=0.75,
        embed_dim=768,
        encoder_depth=12,
        decoder_dim=512,
        decoder_depth=8,
        num_heads=12,
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        
        # Calculate number of patches
        self.num_patches_h = img_size[1] // patch_size
        self.num_patches_w = img_size[0] // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, 1, embed_dim)
        
        # Encoder
        self.encoder = TransformerEncoder(
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=num_heads,
        )
        
        # Decoder
        self.decoder = TransformerDecoder(
            patch_size=patch_size,
            num_patches=self.num_patches,
            encoder_dim=embed_dim,
            decoder_dim=decoder_dim,
            decoder_depth=decoder_depth,
            num_heads=num_heads,
        )
        
        self.patch_dim = patch_size * patch_size
    
    def random_masking(self, x, mask_ratio):
        """
        Random masking of patches.
        
        Args:
            x: (B, N, D) patch embeddings
            mask_ratio: Ratio to mask
        
        Returns:
            x_visible: (B, N_visible, D) visible patches
            mask: (B, N) binary mask (1 = masked)
            indices_restore: (B, N) indices to restore original order
        """
        B, N, D = x.shape
        N_masked = int(N * mask_ratio)
        
        # Random permutation
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Masked and unmasked indices
        ids_keep = ids_shuffle[:, :N - N_masked]
        ids_mask = ids_shuffle[:, N - N_masked:]
        
        # Keep visible patches
        x_visible = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        
        # Create binary mask: 0 = keep, 1 = remove
        mask = torch.ones(B, N, device=x.device)
        mask[:, :N - N_masked] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_visible, mask, ids_restore, ids_mask, ids_keep
    
    def forward(self, imgs, mask_ratio=None):
        """
        Args:
            imgs: Input images (B, 1, H, W)
            mask_ratio: Optional override for mask ratio
        
        Returns:
            loss, pred, mask
        """
        mask_ratio = mask_ratio or self.mask_ratio
        
        # Patch embedding
        x = self.patch_embed(imgs)  # (B, N, D)
        
        # Random masking
        x_visible, mask, ids_restore, ids_mask, ids_keep = self.random_masking(x, mask_ratio)
        
        # Encode (only visible patches)
        latent = self.encoder(x_visible)  # (B, N_visible, D)
        
        # Decode (reconstruct all patches)
        pred = self.decoder(latent, ids_mask, ids_keep)  # (B, N, patch_dim)
        
        # Calculate loss only on masked patches
        target = self.patchify(imgs)
        
        loss = self.forward_loss(pred, target, mask)
        
        return loss, pred, mask
    
    def forward_loss(self, pred, target, mask):
        """
        Calculate MSE loss on masked patches only.
        
        Args:
            pred: (B, N, patch_dim)
            target: (B, N, patch_dim)
            mask: (B, N) binary mask (1 = masked)
        """
        # Normalize target (similar to MAE paper)
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target_norm = (target - mean) / (var + 1e-6) ** 0.5
        
        # MSE loss on masked patches
        loss = (pred - target_norm) ** 2
        loss = loss.mean(dim=-1)  # (B, N)
        
        # Only compute loss on masked patches
        loss = (loss * mask).sum() / mask.sum()
        
        return loss
    
    def patchify(self, imgs):
        """
        Convert images to patches.
        
        Args:
            imgs: (B, 1, H, W)
        
        Returns:
            patches: (B, N, patch_size^2)
        """
        B, C, H, W = imgs.shape
        p = self.patch_size
        
        # Ensure dimensions are divisible by patch size
        H = H // p * p
        W = W // p * p
        imgs = imgs[:, :, :H, :W]
        
        # Reshape to patches
        patches = imgs.reshape(B, C, H // p, p, W // p, p)
        patches = patches.permute(0, 2, 4, 1, 3, 5)
        patches = patches.reshape(B, -1, C * p * p)
        
        return patches
    
    def unpatchify(self, patches):
        """
        Convert patches back to image.
        
        Args:
            patches: (B, N, patch_size^2)
        
        Returns:
            imgs: (B, 1, H, W)
        """
        B, N, D = patches.shape
        p = self.patch_size
        h = self.num_patches_h
        w = self.num_patches_w
        
        patches = patches.reshape(B, h, w, 1, p, p)
        patches = patches.permute(0, 3, 1, 4, 2, 5)
        imgs = patches.reshape(B, 1, h * p, w * p)
        
        return imgs
    
    def get_encoder(self):
        """Get encoder for downstream fine-tuning."""
        return self.encoder


def build_mae(**kwargs):
    """Factory function for MAE."""
    return MaskedAutoencoder(**kwargs)
