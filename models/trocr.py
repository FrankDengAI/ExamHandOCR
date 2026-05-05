"""
TrOCR (Transformer-based Optical Character Recognition) implementation.

Based on Li et al. 2023 "TrOCR: Transformer-based Optical Character Recognition 
with Pre-trained Models" published at AAAI 2023.

TrOCR uses a Vision Transformer (BEiT) encoder and a RoBERTa decoder with
cross-attention for text generation. This architecture achieved state-of-the-art
results on IAM dataset and serves as the second-best baseline on ExamHandOCR.

Paper Results on ExamHandOCR (Table 1, Section 7.3):
    TrOCR-Base: CER 8.71%, ESA-CER 11.83%, RI 0.53
    TrOCR + SSL (Ours): CER 5.84%, ESA-CER 7.62%, RI 0.69

The SSL variant uses MAE pre-training on 3.15M unannotated images (Section 7.1),
achieving 14× reduction in annotation requirement.
"""

import torch
import torch.nn as nn
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    BeitModel,
    BeitConfig,
    RobertaTokenizer,
    RobertaForCausalLM,
    RobertaConfig,
)
from typing import Optional, Dict, List


class TrOCRModel(nn.Module):
    """
    TrOCR model for handwritten text recognition.
    
    Architecture:
    - Encoder: BEiT (BERT Pre-Training of Image Transformers) pretrained on 
              large-scale image data
    - Decoder: RoBERTa (Robustly Optimized BERT Approach) with cross-attention
              to encoder outputs
    - Training: Standard cross-entropy loss on token prediction
    
    The model uses beam search for generation during inference.
    
    Args:
        model_name: HuggingFace model identifier
        max_length: Maximum generation length
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/trocr-base-handwritten",
        max_length: int = 512,
    ):
        super().__init__()
        
        # Load pretrained processor and model
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        
        self.max_length = max_length
        
        # Configure generation
        self.model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.model.config.vocab_size = self.model.config.decoder.vocab_size
        
        # Beam search configuration
        self.model.config.eos_token_id = self.processor.tokenizer.sep_token_id
        self.model.config.max_length = max_length
        self.model.config.early_stopping = True
        self.model.config.length_penalty = 2.0
        self.model.config.num_beams = 4
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training or inference.
        
        Args:
            pixel_values: Input images (B, C, H, W)
            labels: Target token IDs (B, seq_len), optional
        
        Returns:
            Dictionary with loss and/or logits
        """
        outputs = self.model(
            pixel_values=pixel_values,
            labels=labels,
        )
        
        return {
            'loss': outputs.loss,
            'logits': outputs.logits,
        }
    
    def generate(
        self,
        pixel_values: torch.Tensor,
        num_beams: int = 4,
        max_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate text from images using beam search.
        
        Args:
            pixel_values: Input images (B, C, H, W)
            num_beams: Number of beams for beam search
            max_length: Maximum generation length
        
        Returns:
            Generated token IDs (B, seq_len)
        """
        max_length = max_length or self.max_length
        
        generated_ids = self.model.generate(
            pixel_values,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
        )
        
        return generated_ids
    
    def decode(self, generated_ids: torch.Tensor) -> List[str]:
        """
        Decode generated token IDs to text strings.
        
        Args:
            generated_ids: Tensor of token IDs (B, seq_len)
        
        Returns:
            List of decoded text strings
        """
        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        return generated_text
    
    def freeze_encoder(self):
        """Freeze encoder weights (for transfer learning experiments)."""
        for param in self.model.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze encoder weights."""
        for param in self.model.encoder.parameters():
            param.requires_grad = True


class MAEDecoder(nn.Module):
    """
    Lightweight decoder for MAE (Masked Autoencoder) reconstruction.
    
    The decoder reconstructs the original image from the encoded latent
    representation. It is much lighter than the encoder, as reconstruction
    is only needed during pre-training, not during fine-tuning.
    
    Architecture (from He et al. 2022):
    - Input: Latent features from encoder + mask tokens for masked positions
    - Processing: Series of transformer blocks
    - Output: Reconstructed patches
    
    Args:
        patch_size: Size of image patches
        num_patches: Total number of patches in image
        encoder_dim: Dimension of encoder output
        decoder_dim: Hidden dimension for decoder
        num_layers: Number of transformer decoder layers
        num_heads: Number of attention heads
    """
    
    def __init__(
        self,
        patch_size: int,
        num_patches: int,
        encoder_dim: int,
        decoder_dim: int,
        num_layers: int,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.decoder_dim = decoder_dim
        
        # Learnable mask token for masked positions
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        
        # Positional embeddings for all patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_dim))
        
        # Projection from encoder to decoder dimension
        self.encoder_to_decoder = nn.Linear(encoder_dim, decoder_dim)
        
        # Transformer decoder blocks
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=num_heads,
            dim_feedforward=int(decoder_dim * mlp_ratio),
            dropout=0.1,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output head: project to pixel values
        self.head = nn.Linear(decoder_dim, patch_size * patch_size * 1)  # Grayscale
        
        # Initialize
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(
        self,
        encoder_output: torch.Tensor,
        mask_indices: torch.Tensor,
        unmask_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode and reconstruct image patches.
        
        Args:
            encoder_output: (B, N_visible, encoder_dim)
            mask_indices: (B, N_masked) indices of masked patches
            unmask_indices: (B, N_visible) indices of visible patches
        
        Returns:
            Reconstructed patches (B, N_total, patch_dim)
        """
        B = encoder_output.size(0)
        N_masked = mask_indices.size(1)
        
        # Project encoder output to decoder dimension
        visible_tokens = self.encoder_to_decoder(encoder_output)
        
        # Expand mask tokens for all masked positions
        mask_tokens = self.mask_token.expand(B, N_masked, -1)
        
        # Combine visible and masked tokens
        # Strategy: maintain original position order for positional encoding
        full_tokens = torch.zeros(B, self.num_patches, self.decoder_dim,
                                 device=encoder_output.device)
        
        for b in range(B):
            # Place visible tokens
            full_tokens[b, unmask_indices[b]] = visible_tokens[b]
            # Place mask tokens
            full_tokens[b, mask_indices[b]] = mask_tokens[b]
        
        # Add positional embeddings
        full_tokens = full_tokens + self.pos_embed
        
        # Apply transformer decoder
        decoded = self.decoder(full_tokens, full_tokens)
        
        # Reconstruct pixel values
        recon = self.head(decoded)
        
        return recon


class TrOCRWithSSL(nn.Module):
    """
    TrOCR with Self-Supervised Learning (SSL) pre-training via MAE.
    
    This is our proposed method achieving the best results on ExamHandOCR.
    
    Training Strategy (Section 7.1):
    1. Pre-training: MAE on 3.15M unannotated images from train-unsup
       - 100 epochs, lr=1.5e-4, warmup=40 epochs
       - Mask ratio: 0.75, Patch size: 16
    2. Fine-tuning: Supervised training on 6,048 annotated images
       - Encoder initialized from MAE, decoder randomly initialized
       - lr=1e-4, cosine decay, 50 epochs
    
    Results (Table 1):
        - CER: 5.84% (33% improvement over supervised-only TrOCR)
        - ESA-CER: 7.62%
        - RI: 0.69
        - Annotation efficiency: 14× reduction (432 images match 6,048)
    
    Args:
        encoder_name: HuggingFace BEiT model identifier
        decoder_vocab_size: Size of output vocabulary
        patch_size: Patch size for MAE
        mask_ratio: Ratio of patches to mask during MAE
        max_length: Maximum generation length
    """
    
    def __init__(
        self,
        encoder_name: str = "microsoft/beit-base-patch16-224",
        decoder_vocab_size: int = 50265,
        patch_size: int = 16,
        mask_ratio: float = 0.75,
        max_length: int = 512,
    ):
        super().__init__()
        
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        
        # Load BEiT encoder
        self.encoder = BeitModel.from_pretrained(encoder_name)
        encoder_config = self.encoder.config
        encoder_dim = encoder_config.hidden_size
        
        # MAE decoder for SSL pre-training
        num_patches = (384 // patch_size) * (128 // patch_size)
        self.mae_decoder = MAEDecoder(
            patch_size=patch_size,
            num_patches=num_patches,
            encoder_dim=encoder_dim,
            decoder_dim=512,
            num_layers=8,
            num_heads=16,
        )
        
        # OCR decoder (RoBERTa-based)
        decoder_config = RobertaConfig(
            vocab_size=decoder_vocab_size,
            hidden_size=encoder_dim,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=encoder_dim * 4,
            max_position_embeddings=max_length,
            is_decoder=True,
            add_cross_attention=True,
        )
        self.ocr_decoder = RobertaForCausalLM(decoder_config)
        
        self.max_length = max_length
        
        # Connector: encoder output to decoder input
        self.encoder_to_decoder = nn.Linear(encoder_dim, encoder_dim)
        
        # Special tokens
        self.sos_token_id = 0
        self.eos_token_id = 1
        self.pad_token_id = 2
    
    def forward_mae(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for MAE pre-training.
        
        Args:
            pixel_values: Input images (B, 1, H, W)
        
        Returns:
            Reconstruction loss
        """
        B = pixel_values.size(0)
        
        # Patchify and create mask
        patches = self.patchify(pixel_values)
        num_patches = patches.size(1)
        
        # Random masking
        num_masked = int(num_patches * self.mask_ratio)
        
        # Shuffle indices
        noise = torch.rand(B, num_patches, device=pixel_values.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Split into masked and unmasked
        ids_keep = ids_shuffle[:, :num_patches - num_masked]
        ids_mask = ids_shuffle[:, num_patches - num_masked:]
        
        # Encode only visible patches using BEiT
        # BEiT expects pixel values, so we need to handle this carefully
        encoder_output = self.encoder(pixel_values).last_hidden_state
        
        # Take only the first num_patches tokens (patch embeddings)
        encoder_output = encoder_output[:, :num_patches]
        
        # Select visible tokens
        visible_output = torch.gather(
            encoder_output, dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, encoder_output.size(-1))
        )
        
        # Decode with MAE decoder
        pred = self.mae_decoder(visible_output, ids_mask, ids_keep)
        
        # Compute reconstruction loss on masked patches only
        target = patches
        loss = self.mae_loss(pred, target, ids_mask)
        
        return loss
    
    def forward_ocr(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for OCR fine-tuning.
        
        Args:
            pixel_values: Input images (B, 1, H, W)
            labels: Target token IDs (B, seq_len)
        
        Returns:
            Dictionary with loss and logits
        """
        # Encode with BEiT
        encoder_output = self.encoder(pixel_values).last_hidden_state
        
        # Project to decoder input
        decoder_inputs = self.encoder_to_decoder(encoder_output)
        
        # Decode with RoBERTa
        if labels is not None:
            outputs = self.ocr_decoder(
                inputs_embeds=decoder_inputs,
                labels=labels,
            )
            return {
                'loss': outputs.loss,
                'logits': outputs.logits,
            }
        else:
            # Inference
            outputs = self.ocr_decoder(inputs_embeds=decoder_inputs)
            return {'logits': outputs.logits}
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mode: str = 'ocr',
    ) -> Dict[str, torch.Tensor]:
        """Unified forward pass."""
        if mode == 'mae':
            loss = self.forward_mae(pixel_values)
            return {'loss': loss}
        else:
            return self.forward_ocr(pixel_values, labels)
    
    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
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
        H = (H // p) * p
        W = (W // p) * p
        imgs = imgs[:, :, :H, :W]
        
        # Reshape to patches
        patches = imgs.reshape(B, C, H // p, p, W // p, p)
        patches = patches.permute(0, 2, 4, 1, 3, 5)
        patches = patches.reshape(B, -1, C * p * p)
        
        return patches
    
    def mae_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute MAE loss on masked patches only.
        
        Normalizes targets and computes MSE only on masked positions.
        """
        B = pred.size(0)
        
        # Normalize target (per-patch normalization)
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target_norm = (target - mean) / (var + 1e-6) ** 0.5
        
        # Compute loss on all patches
        loss = (pred - target_norm) ** 2
        loss = loss.mean(dim=-1)  # (B, N)
        
        # Mask for masked positions
        mask = torch.zeros(B, target.size(1), device=target.device)
        for b in range(B):
            mask[b, mask_indices[b]] = 1
        
        # Weighted average over masked positions
        loss = (loss * mask).sum() / mask.sum()
        
        return loss
    
    def generate(
        self,
        pixel_values: torch.Tensor,
        num_beams: int = 4,
    ) -> torch.Tensor:
        """Generate text from images."""
        # Encode
        encoder_output = self.encoder(pixel_values).last_hidden_state
        decoder_inputs = self.encoder_to_decoder(encoder_output)
        
        # Generate
        outputs = self.ocr_decoder.generate(
            inputs_embeds=decoder_inputs,
            max_length=self.max_length,
            num_beams=num_beams,
        )
        
        return outputs


def build_trocr(model_name: str = "microsoft/trocr-base-handwritten", **kwargs) -> TrOCRModel:
    """Factory function for TrOCR."""
    return TrOCRModel(model_name=model_name, **kwargs)


def build_trocr_ssl(
    encoder_name: str = "microsoft/beit-base-patch16-224",
    **kwargs
) -> TrOCRWithSSL:
    """Factory function for TrOCR with SSL pre-training capability."""
    return TrOCRWithSSL(encoder_name=encoder_name, **kwargs)
