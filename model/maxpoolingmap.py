import torch
import torch.nn as nn
import torch.nn.functional as F


class MaxPoolingMap(nn.Module):
    """
    High-frequency modulates low-frequency using multi-scale max pooling.
    
    - Uses 3 scales of max pooling (kernel sizes: 3, 5, 7)
    - Combines results with softmax-weighted fusion
    - Generates modulation map to multiply with low-frequency features
    - Final output with residual connection
    """
    
    def __init__(
        self,
        dim: int,
        pool_sizes: tuple = (3, 5, 7),
        reduction: int = 4,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.pool_sizes = pool_sizes
        self.num_scales = len(pool_sizes)
        
        self.poolings = nn.ModuleList([
            nn.MaxPool2d(
                kernel_size=size,
                stride=1,
                padding=size // 2
            ) for size in pool_sizes
        ])
        
        self.scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim // reduction, 1, bias=False),
                nn.BatchNorm2d(dim // reduction),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim // reduction, dim, 1, bias=False),
                nn.BatchNorm2d(dim),
            ) for _ in pool_sizes
        ])
        
        self.attention_conv = nn.Sequential(
            nn.Conv2d(dim * self.num_scales, self.num_scales, 1, bias=False),
            nn.Softmax(dim=1)
        )
        
        self.map_gen = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.Sigmoid()
        )
        
    def forward(self, high_feat: torch.Tensor, low_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            high_feat: [B, C, H, W], high-frequency feature
            low_feat: [B, C, H, W], low-frequency feature
            
        Returns:
            modulated_low: [B, C, H, W]
        """
        if high_feat.shape != low_feat.shape:
            raise ValueError(
                f"Shape mismatch: high_feat {tuple(high_feat.shape)} vs low_feat {tuple(low_feat.shape)}"
            )
        
        b, c, h, w = high_feat.shape
        if c != self.dim:
            raise ValueError(f"Input channel {c} must match dim={self.dim}.")
        
        scale_features = []
        for pooling, conv in zip(self.poolings, self.scale_convs):
            pooled = pooling(high_feat)
            feat = conv(pooled)
            scale_features.append(feat)
        
        concat_feats = torch.cat(scale_features, dim=1)
        
        scale_weights = self.attention_conv(concat_feats)
        
        fused_feat = torch.zeros_like(scale_features[0])
        for i, feat in enumerate(scale_features):
            fused_feat = fused_feat + scale_weights[:, i:i+1, :, :] * feat
        
        modulation_map = self.map_gen(fused_feat)
        
        modulated_low = low_feat * modulation_map + low_feat
        
        return modulated_low
