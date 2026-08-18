import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class SpatialFiLM(nn.Module):
    """
    空间级 FiLM (Feature-wise Linear Modulation)
    在空间维度上进行调制，保留空间细节
    """
    def __init__(self, num_channels, reduction=16):
        """
        参数:
            num_channels: 特征通道数
            reduction: 降维比例
        """
        super().__init__()
        self.num_channels = num_channels
        hidden_dim = max(num_channels // reduction, 64)
        
        # 空间 FiLM 生成器（保留空间分辨率，使用LayerNorm）
        self.film_generator = nn.Sequential(
            nn.Conv2d(num_channels, hidden_dim, 3, padding=1),
            LayerNorm(hidden_dim, eps=1e-6, data_format="channels_first"),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_channels * 2, 1)  # [B, 2C, H, W]
        )
        
        # CBAM空间注意力生成器
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        # Dropout
        self.dropout = nn.Dropout2d(p=0.2)
        
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, condition):
        """
        前向传播：空间感知的自适应FiLM
        
        参数:
            x: [B, C, H, W] 待调制的特征
            condition: [B, C, H, W] 条件特征
            
        返回:
            output: [B, C, H, W] 调制后的特征
        """
        # 1. 生成CBAM空间注意力图
        # 通道池化：Max和Avg
        max_pool = torch.max(condition, dim=1, keepdim=True)[0]  # [B, 1, H, W]
        avg_pool = torch.mean(condition, dim=1, keepdim=True)    # [B, 1, H, W]
        spatial_input = torch.cat([max_pool, avg_pool], dim=1)   # [B, 2, H, W]
        
        # 生成空间注意力图
        spatial_attn = self.spatial_attn(spatial_input)  # [B, 1, H, W]
        
        # 2. 生成FiLM参数
        film_params = self.film_generator(condition)  # [B, 2C, H, W]
        gamma, beta = torch.chunk(film_params, 2, dim=1)  # 各 [B, C, H, W]
        
        # gamma 通过 sigmoid 限制
        gamma = torch.sigmoid(gamma) * 2.0
        
        # 3. 空间注意力调制FiLM参数（自适应强度）
        gamma_adaptive = gamma * spatial_attn  # [B, C, H, W]
        beta_adaptive = beta * spatial_attn    # [B, C, H, W]
        
        # 4. 应用自适应FiLM
        modulated = gamma_adaptive * x + beta_adaptive
        
        # 5. Dropout
        modulated = self.dropout(modulated)
        
        # 6. 残差连接
        output = x + modulated
        
        return output


if __name__ == '__main__':
    # 测试 SpatialFiLM
    print("=" * 50)
    print("测试 SpatialFiLM")
    print("=" * 50)
    
    B, C, H, W = 2, 768, 8, 8
    
    spatial_film = SpatialFiLM(num_channels=C, reduction=16)
    
    x = torch.randn(B, C, H, W)
    condition = torch.randn(B, C, H, W)
    
    output = spatial_film(x, condition)
    
    print(f"输入形状: {x.shape}")
    print(f"条件形状: {condition.shape}")
    print(f"输出形状: {output.shape}")
    print(f"参数量: {sum(p.numel() for p in spatial_film.parameters()) / 1e6:.2f}M")
