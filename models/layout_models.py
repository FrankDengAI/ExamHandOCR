"""
Layout analysis models for text-line segmentation.
Implements U-Net, Mask R-CNN (placeholder), and DETR-based models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolutional block with batch norm and activation."""
    
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)


class DownBlock(nn.Module):
    """Downsampling block."""
    
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)
    
    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class UpBlock(nn.Module):
    """Upsampling block with skip connection."""
    
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ConvBlock(in_ch, out_ch)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
            self.conv = ConvBlock(in_ch, out_ch)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle size mismatch
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class UNetLayout(nn.Module):
    """
    U-Net for text-line segmentation.
    Baseline model as described in Section 7.1.
    
    Args:
        in_channels: Input channels (1 for grayscale)
        num_classes: Number of output classes (2 for binary: text/background)
        bilinear: Use bilinear upsampling
    """
    
    def __init__(self, in_channels=1, num_classes=2, bilinear=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        
        # Encoder
        self.inc = ConvBlock(in_channels, 64)
        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        self.down3 = DownBlock(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = DownBlock(512, 1024 // factor)
        
        # Decoder
        self.up1 = UpBlock(1024, 512 // factor, bilinear)
        self.up2 = UpBlock(512, 256 // factor, bilinear)
        self.up3 = UpBlock(256, 128 // factor, bilinear)
        self.up4 = UpBlock(128, 64, bilinear)
        
        # Output
        self.outc = nn.Conv2d(64, num_classes, 1)
    
    def forward(self, x):
        """
        Args:
            x: Input images (B, C, H, W)
        
        Returns:
            logits: (B, num_classes, H, W)
        """
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        return logits
    
    def predict(self, x, threshold=0.5):
        """Predict segmentation masks."""
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
        return preds


class DETRLayout(nn.Module):
    """
    DETR-based layout model for text-line detection.
    Uses transformer encoder-decoder architecture.
    
    Note: This is a simplified implementation. Full DETR requires
    more complex components (Hungarian matching, etc.).
    
    Args:
        in_channels: Input channels
        hidden_dim: Hidden dimension
        num_queries: Number of object queries
        num_classes: Number of classes (background + text-line)
    """
    
    def __init__(
        self,
        in_channels=1,
        hidden_dim=256,
        num_queries=100,
        num_classes=2,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
    ):
        super().__init__()
        
        self.num_queries = num_queries
        self.num_classes = num_classes
        
        # Backbone CNN
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        )
        
        # Input projection
        self.input_proj = nn.Conv2d(hidden_dim, hidden_dim, 1)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # Object queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # Output heads
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),  # (x, y, w, h)
        )
        
        # Segmentation head (optional)
        self.mask_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(self, x):
        """
        Args:
            x: Input images (B, C, H, W)
        
        Returns:
            pred_logits: (B, num_queries, num_classes)
            pred_boxes: (B, num_queries, 4)
        """
        B = x.size(0)
        
        # Extract features
        features = self.backbone(x)  # (B, hidden_dim, H', W')
        
        # Project and flatten
        src = self.input_proj(features)
        H, W = src.shape[-2:]
        src = src.flatten(2).permute(0, 2, 1)  # (B, H'*W', hidden_dim)
        
        # Create positional encoding
        pos = self.create_positional_encoding(H, W, src.size(-1), x.device)
        pos = pos.flatten(0, 1).unsqueeze(0).expand(B, -1, -1)
        
        # Encode
        memory = self.encoder(src + pos)
        
        # Decode
        query_embed = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        tgt = torch.zeros_like(query_embed)
        
        hs = self.decoder(tgt, memory, query_pos=query_embed)
        
        # Predict
        pred_logits = self.class_embed(hs)
        pred_boxes = self.bbox_embed(hs).sigmoid()
        
        return {'pred_logits': pred_logits, 'pred_boxes': pred_boxes}
    
    def create_positional_encoding(self, H, W, dim, device):
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(H, W, dim, device=device)
        
        y_pos = torch.arange(H, device=device).unsqueeze(1).repeat(1, W)
        x_pos = torch.arange(W, device=device).unsqueeze(0).repeat(H, 1)
        
        dim_t = torch.arange(dim // 2, device=device)
        dim_t = 10000 ** (2 * (dim_t // 2) / (dim // 2))
        
        pe[:, :, 0::2] = torch.sin(y_pos.unsqueeze(-1) / dim_t)
        pe[:, :, 1::2] = torch.cos(y_pos.unsqueeze(-1) / dim_t)
        
        return pe


def build_unet_layout(**kwargs):
    """Factory function for U-Net layout model."""
    return UNetLayout(**kwargs)


def build_detr_layout(**kwargs):
    """Factory function for DETR layout model."""
    return DETRLayout(**kwargs)
