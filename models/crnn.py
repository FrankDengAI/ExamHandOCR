"""
CRNN (Convolutional Recurrent Neural Network) baseline implementation.

Architecture based on Shi et al. 2016 "An End-to-End Trainable Neural Network 
for Image-based Sequence Recognition and its Application to Scene Text Recognition".

This is the lightweight baseline for ExamHandOCR, using CNN feature extraction
followed by bidirectional LSTM and CTC decoding. While simple, it serves as
an important baseline for evaluating more complex architectures.

Paper Results (Table 1, Section 7.3):
    CER: 18.72%, WER: 27.43%, ESA-CER: 24.31%, RI: 0.34
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet31Block(nn.Module):
    """
    ResNet-31 style residual block.
    
    Used as the building block for the CNN encoder in CRNN. Consists of
    two 3x3 convolutions with batch normalization and ReLU activation,
    with a residual skip connection.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Stride for the first convolution (default: 1)
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        # First convolution
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolution
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection for residual
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Residual connection
        out = F.relu(out)
        return out


class CRNNEncoder(nn.Module):
    """
    CNN encoder with ResNet-31 backbone for feature extraction.
    
    The encoder progressively downsamples the input image while extracting
    hierarchical features. For ExamHandOCR with input size (384, 128):
    - After layer2: H/2, W/2
    - After layer4: H/4, W/4
    
    This provides a good balance between spatial resolution and feature depth
    for handwritten text recognition.
    """
    
    def __init__(self, in_channels: int = 1, hidden_size: int = 512):
        super().__init__()
        
        # Initial convolution (no downsampling)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # ResNet blocks with progressive downsampling
        # Layer 1: 32 -> 32, no downsampling
        self.layer1 = self._make_layer(32, 32, num_blocks=2, stride=1)
        
        # Layer 2: 32 -> 64, downsample by 2
        self.layer2 = self._make_layer(32, 64, num_blocks=2, stride=2)
        
        # Layer 3: 64 -> 128, no downsampling
        self.layer3 = self._make_layer(64, 128, num_blocks=2, stride=1)
        
        # Layer 4: 128 -> 256, downsample by 2
        self.layer4 = self._make_layer(128, 256, num_blocks=2, stride=2)
        
        # Layer 5: 256 -> 512, no downsampling
        self.layer5 = self._make_layer(256, 512, num_blocks=2, stride=1)
        
        # Layer 6: 512 -> hidden_size, no downsampling
        self.layer6 = self._make_layer(512, hidden_size, num_blocks=2, stride=1)
    
    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int):
        """Create a layer with multiple residual blocks."""
        layers = []
        layers.append(ResNet31Block(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResNet31Block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input image.
        
        Args:
            x: Input images (B, C, H, W)
        
        Returns:
            Feature sequence (B, W', C'*H') for RNN processing
        """
        # Initial conv
        x = F.relu(self.bn1(self.conv1(x)))
        
        # ResNet stages
        x = self.layer1(x)
        x = self.layer2(x)  # /2
        x = self.layer3(x)
        x = self.layer4(x)  # /4
        x = self.layer5(x)
        x = self.layer6(x)
        
        # Reshape for RNN: (B, C, H, W) -> (B, W, C*H)
        B, C, H, W = x.size()
        x = x.permute(0, 3, 1, 2)  # (B, W, C, H)
        x = x.reshape(B, W, C * H)  # Each vertical strip becomes a feature vector
        
        return x


class BidirectionalLSTM(nn.Module):
    """
    Bidirectional LSTM for sequence modeling.
    
    The bidirectional structure allows the model to capture both left-to-right
    and right-to-left dependencies, which is crucial for handwritten text where
    context from both directions helps resolve ambiguous characters.
    
    Args:
        input_size: Size of input features
        hidden_size: Size of hidden state
        output_size: Size of output (if None, same as hidden_size)
        dropout: Dropout probability
        num_layers: Number of LSTM layers (default: 2)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int = None,
        dropout: float = 0.1,
        num_layers: int = 2,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # Output projection (2*hidden_size due to bidirectional)
        output_size = output_size or hidden_size
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process feature sequence with bidirectional LSTM.
        
        Args:
            x: Feature sequence (B, T, input_size)
        
        Returns:
            Output sequence (B, T, output_size)
        """
        # Flatten LSTM parameters for potential multi-GPU training
        self.lstm.flatten_parameters()
        
        # LSTM forward
        x, _ = self.lstm(x)
        
        # Dropout for regularization
        x = self.dropout(x)
        
        # Project to output size
        x = self.fc(x)
        
        return x


class CRNN(nn.Module):
    """
    Complete CRNN model for handwriting recognition.
    
    Architecture:
    1. CNN Encoder: ResNet-31 backbone for visual feature extraction
    2. Bidirectional LSTM: Sequence modeling with left-to-right and right-to-left processing
    3. CTC Decoder: Connectionist Temporal Classification for alignment-free training
    
    The model is designed for CTC loss, making it suitable for handwritten text
    where character-level alignment is not available.
    
    Paper results on ExamHandOCR (Table 1):
        CER: 18.72%, WER: 27.43%, ESA-CER: 24.31%, RI: 0.34
    
    Args:
        num_classes: Number of character classes (including CTC blank)
        in_channels: Input image channels (1 for grayscale)
        hidden_size: Hidden dimension for LSTM
        image_height: Expected image height (default: 128)
    """
    
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        hidden_size: int = 512,
        image_height: int = 128,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.image_height = image_height
        
        # CNN encoder
        self.encoder = CRNNEncoder(in_channels, hidden_size)
        
        # Calculate feature dimensions after CNN
        # After 2 stride-2 layers: H -> H/4
        feature_height = image_height // 4
        rnn_input_size = hidden_size * feature_height
        
        # Bidirectional LSTM
        self.rnn = BidirectionalLSTM(
            input_size=rnn_input_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=0.1,
            num_layers=2,
        )
        
        # Output classifier
        self.fc = nn.Linear(hidden_size, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for training with CTC loss.
        
        Args:
            x: Input images (B, C, H, W)
        
        Returns:
            Logits of shape (T, B, num_classes) for CTC loss computation
            Note: T is the sequence length (width after CNN downsampling)
        """
        # CNN feature extraction
        features = self.encoder(x)  # (B, W', C'*H')
        
        # RNN sequence modeling
        output = self.rnn(features)  # (B, W', hidden_size)
        
        # Project to character classes
        logits = self.fc(output)  # (B, W', num_classes)
        
        # Transpose to (T, B, C) format expected by CTC loss
        logits = logits.permute(1, 0, 2)
        
        return logits
    
    def predict(self, x: torch.Tensor) -> List[List[int]]:
        """
        Greedy CTC decoding for inference.
        
        Performs best-path decoding by taking the argmax at each timestep,
        then merging repeated characters and removing blanks.
        
        Args:
            x: Input images (B, C, H, W)
        
        Returns:
            List of decoded sequences (as lists of character IDs)
        """
        with torch.no_grad():
            logits = self.forward(x)
            preds = logits.argmax(dim=-1)  # (T, B)
            
            # CTC decoding: merge repeats and remove blanks
            B = preds.size(1)
            results = []
            
            for b in range(B):
                seq = preds[:, b].tolist()
                decoded = []
                prev = -1
                
                for p in seq:
                    # Remove blanks (class 0) and merge repeats
                    if p != prev and p != 0:
                        decoded.append(p)
                    prev = p
                
                results.append(decoded)
        
        return results
    
    def beam_search_decode(
        self,
        x: torch.Tensor,
        beam_width: int = 5,
    ) -> List[Tuple[List[int], float]]:
        """
        Beam search decoding (optional, more accurate than greedy).
        
        Args:
            x: Input images (B, C, H, W)
            beam_width: Number of beams to maintain
        
        Returns:
            List of (decoded_sequence, log_probability) tuples
        """
        with torch.no_grad():
            logits = self.forward(x)
            log_probs = F.log_softmax(logits, dim=-1)
            
            B = log_probs.size(1)
            results = []
            
            for b in range(B):
                # Simple beam search implementation
                beam = [( [], 0.0 )]  # (sequence, score)
                
                for t in range(logits.size(0)):
                    new_beam = []
                    for seq, score in beam:
                        # Top-k extensions
                        top_k = log_probs[t, b].topk(beam_width)
                        for log_prob, idx in zip(top_k.values, top_k.indices):
                            new_seq = seq + [idx.item()]
                            new_score = score + log_prob.item()
                            new_beam.append((new_seq, new_score))
                    
                    # Keep top beam_width sequences
                    new_beam.sort(key=lambda x: x[1], reverse=True)
                    beam = new_beam[:beam_width]
                
                # CTC merge and return best
                best_seq, best_score = beam[0]
                decoded = []
                prev = -1
                for p in best_seq:
                    if p != prev and p != 0:
                        decoded.append(p)
                    prev = p
                
                results.append((decoded, best_score))
        
        return results


def build_crnn(num_classes: int, **kwargs) -> CRNN:
    """
    Factory function to create CRNN model.
    
    Args:
        num_classes: Number of character classes (including blank)
        **kwargs: Additional arguments passed to CRNN constructor
    
    Returns:
        CRNN model instance
    """
    return CRNN(num_classes=num_classes, **kwargs)
