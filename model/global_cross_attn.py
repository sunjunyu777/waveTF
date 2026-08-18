import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """CBAM Channel Attention Module"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out


class AdaptiveMultiScaleCrossAttention(nn.Module):
    """
    自适应多尺度交叉注意力
    
    创新点：
    1. 同时在 3 个尺度上做交叉注意力（1x, 2x, 4x 下采样）
    2. 用轻量级网络预测每个尺度的重要性权重（内容自适应）
    3. 动态融合多尺度结果
    
    适用场景：
    - 密集场景（人多）→ 权重倾向 1x/2x（保留细节）
    - 稀疏场景（人少）→ 权重倾向 4x（扩大感受野）
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qk_dim: int = 128,
        scales: tuple = (1, 2, 4),
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if qk_dim % num_heads != 0:
            raise ValueError(f"qk_dim ({qk_dim}) must be divisible by num_heads ({num_heads}).")

        self.dim = dim
        self.num_heads = num_heads
        self.qk_dim = qk_dim
        self.head_dim = qk_dim // num_heads
        self.scales = scales
        self.attn_dropout = attn_dropout

        # 输入归一化
        self.input_norm_high = nn.GroupNorm(1, dim)
        self.input_norm_low = nn.GroupNorm(1, dim)
        
        # Q 投影（所有尺度共享）
        self.q_proj = nn.Conv2d(dim, qk_dim, kernel_size=1, bias=False)
        self.q_norm = nn.GroupNorm(1, qk_dim)
        
        # 为每个尺度创建独立的 K/V 投影
        self.k_projs = nn.ModuleList([
            nn.Conv2d(dim, qk_dim, kernel_size=1, bias=False) for _ in scales
        ])
        self.v_projs = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size=1, bias=False) for _ in scales
        ])
        self.k_norms = nn.ModuleList([nn.GroupNorm(1, qk_dim) for _ in scales])
        self.v_norms = nn.ModuleList([nn.GroupNorm(1, dim) for _ in scales])
        
        # 尺度权重预测网络（根据 high_feat 的内容动态决定用哪个尺度）
        self.scale_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, len(scales), 1, bias=False),
            nn.Softmax(dim=1)  # [B, num_scales, 1, 1]
        )
        
        # 输出投影
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.out_dropout = nn.Dropout(proj_dropout)

    def _to_heads(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """[B, C, H, W] -> [B, heads, HW, head_dim]"""
        b = x.shape[0]
        c = x.shape[1]
        head_dim = c // self.num_heads
        x = x.view(b, self.num_heads, head_dim, h * w)
        return x.transpose(-1, -2).contiguous()  # [B, heads, HW, head_dim]

    def _from_heads(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """[B, heads, HW, head_dim] -> [B, C, H, W]"""
        b = x.shape[0]
        x = x.transpose(-1, -2).contiguous()  # [B, heads, head_dim, HW]
        x = x.view(b, self.dim, h, w)
        return x

    def forward(self, high_feat: torch.Tensor, low_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            high_feat: [B, C, H, W], high-frequency branch feature.
            low_feat:  [B, C, H, W], low-frequency branch feature.

        Returns:
            enhanced_high: [B, C, H, W]
        """
        if high_feat.shape != low_feat.shape:
            raise ValueError(
                f"Shape mismatch: high_feat {tuple(high_feat.shape)} vs low_feat {tuple(low_feat.shape)}"
            )

        b, c, h, w = high_feat.shape
        if c != self.dim:
            raise ValueError(f"Input channel {c} must match dim={self.dim}.")

        # 归一化
        high_feat_norm = self.input_norm_high(high_feat)
        low_feat_norm = self.input_norm_low(low_feat)
        
        # 1. 预测尺度权重（内容自适应）
        scale_weights = self.scale_predictor(high_feat_norm)  # [B, num_scales, 1, 1]
        
        # 2. Q 投影（所有尺度共享）
        q = self.q_norm(self.q_proj(high_feat_norm))  # [B, qk_dim, H, W]
        q = self._to_heads(q, h, w)  # [B, heads, HW, head_dim]
        
        # 3. 多尺度 K/V 投影 + 交叉注意力
        multi_scale_outputs = []
        scale = self.head_dim ** -0.5
        
        for i, (s, k_proj, v_proj, k_norm, v_norm) in enumerate(
            zip(self.scales, self.k_projs, self.v_projs, self.k_norms, self.v_norms)
        ):
            # 下采样
            if s > 1:
                low_ds = F.avg_pool2d(low_feat_norm, kernel_size=s, stride=s)
            else:
                low_ds = low_feat_norm
            
            h_ds, w_ds = low_ds.shape[-2:]
            
            # K/V 投影
            k = k_norm(k_proj(low_ds))  # [B, qk_dim, H/s, W/s]
            v = v_norm(v_proj(low_ds))  # [B, dim, H/s, W/s]
            k = self._to_heads(k, h_ds, w_ds)  # [B, heads, (H/s)*(W/s), head_dim]
            v = self._to_heads(v, h_ds, w_ds)  # [B, heads, (H/s)*(W/s), head_dim]
            
            # 交叉注意力
            attn = torch.matmul(q, k.transpose(-1, -2)) * scale  # [B, heads, HW, (H/s)*(W/s)]
            attn = torch.softmax(attn, dim=-1)
            if self.attn_dropout > 0 and self.training:
                attn = F.dropout(attn, p=self.attn_dropout, training=True)
            
            out = torch.matmul(attn, v)  # [B, heads, HW, head_dim]
            out = self._from_heads(out, h, w)  # [B, C, H, W]
            
            multi_scale_outputs.append(out)
        
        # 4. 加权融合多尺度结果
        fused = sum(
            scale_weights[:, i:i+1, :, :] * out 
            for i, out in enumerate(multi_scale_outputs)
        )
        
        # 5. 输出投影 + 残差
        msg = self.out_proj(fused)
        msg = self.out_dropout(msg)
        
        return high_feat + msg


class AdaptiveHighFreqEnhancement(nn.Module):
    """
    空间自适应的趋势一致性高频增强模块

    设计动机：
    - low_feat 表示局部趋势 / 缓慢变化
    - high_feat 表示快速变化 / 细节 / 边缘 / 噪声
    - 因此不应只让 low_feat 单独预测阈值，而应判断：
      当前高频变化是否与低频趋势在局部方向上保持一致

    核心机制：
    1. 用 low_feat 与 high_feat 的局部梯度计算余弦相似度
    2. 余弦相似度越高，说明该高频更可能是“顺着趋势产生的真实细节”
    3. 余弦相似度越低，说明该高频更可能是噪声或伪高频
    4. 用一致性分数联合高频幅值，动态控制增强系数与软阈值

    数学形式（直观版）：
    - cos_sim = cos(∇L, ∇H)
    - conf = sigmoid( Predictor([low_ctx, |H|, cos_sim]) )
    - H_hat = (1 + tanh(a)) ⊙ H
    - tau = softplus(tau_logits) * (1 + (1 - conf))
    - H_refined = sign(H_hat) * relu(|H_hat| - tau) * sigmoid(k(|H_hat|-tau))
    - out = H + gamma * H_refined

    特点：
    - 接口不变
    - 类名不变
    - 引入“趋势一致性”而不是单纯 low->tau
    - 用可学习残差系数避免把噪声原样加回
    """

    def __init__(self, dim: int, sharpness: float = 10.0):
        super().__init__()
        self.dim = dim
        self.sharpness = nn.Parameter(torch.tensor(sharpness))

        hidden = max(dim // 4, 8)
        fused_dim = dim + dim + 1   # low_feat + |high_feat| + cos_sim = 2*dim + 1

        # 共享编码器（减少冗余计算）
        self.shared_encoder = nn.Sequential(
            nn.Conv2d(fused_dim, hidden, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=hidden),
            nn.GELU()
        )
        
        # 两个独立输出头（去掉 tau_head，阈值直接从 conf 推导）
        self.conf_head = nn.Conv2d(hidden, 1, kernel_size=1, bias=True)
        self.enhance_head = nn.Conv2d(hidden, 1, kernel_size=1, bias=True)

        # 基础阈值（可学习的全局参数）
        self.tau_base = nn.Parameter(torch.tensor(0.1))

    @staticmethod
    def _spatial_gradient(x: torch.Tensor):
        """
        x: [B, C, H, W]
        return:
            gx, gy: [B, C, H, W]
        """
        gx = x[:, :, :, 1:] - x[:, :, :, :-1]
        gy = x[:, :, 1:, :] - x[:, :, :-1, :]

        gx = F.pad(gx, (0, 1, 0, 0), mode='reflect')
        gy = F.pad(gy, (0, 0, 0, 1), mode='reflect')
        return gx, gy

    def _gradient_cosine_similarity(self, low_feat: torch.Tensor, high_feat: torch.Tensor) -> torch.Tensor:
        """
        计算 low/high 局部梯度方向的余弦相似度
        输出: [B, 1, H, W]
        """
        low_gx, low_gy = self._spatial_gradient(low_feat)
        high_gx, high_gy = self._spatial_gradient(high_feat)

        # 沿通道求平均，得到每个空间位置的二维方向向量
        low_vec_x = low_gx.mean(dim=1, keepdim=True)
        low_vec_y = low_gy.mean(dim=1, keepdim=True)

        high_vec_x = high_gx.mean(dim=1, keepdim=True)
        high_vec_y = high_gy.mean(dim=1, keepdim=True)

        dot = low_vec_x * high_vec_x + low_vec_y * high_vec_y
        low_norm = torch.sqrt(low_vec_x.pow(2) + low_vec_y.pow(2) + 1e-6)
        high_norm = torch.sqrt(high_vec_x.pow(2) + high_vec_y.pow(2) + 1e-6)

        cos_sim = dot / (low_norm * high_norm + 1e-6)   # [B,1,H,W], 范围约 [-1,1]
        return cos_sim

    def forward(self, high_feat: torch.Tensor, low_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            high_feat: [B, C, H, W]
            low_feat:  [B, C, H, W]，已编码的低频特征

        Returns:
            out: [B, C, H, W]
        """
        # 1) 低频趋势上下文（直接使用已编码 low_feat）
        low_ctx = low_feat                                  # [B, C, H, W]

        # 2) 高频幅值
        high_amp = high_feat.abs()                          # [B, C, H, W]

        # 3) 趋势一致性（梯度余弦相似度）
        cos_sim = self._gradient_cosine_similarity(low_feat, high_feat)   # [B, 1, H, W]

        # 4) 融合信息
        fused = torch.cat([low_ctx, high_amp, cos_sim], dim=1)            # [B, 2C+1, H, W]

        # 5) 共享编码 + 双头预测
        shared_feat = self.shared_encoder(fused)                          # [B, hidden, H, W]
        conf = torch.sigmoid(self.conf_head(shared_feat))                 # [B, 1, H, W]
        a = self.enhance_head(shared_feat)                                # [B, 1, H, W]

        # 6) 动态增强（去掉 clamp，范围 [0, 2]）
        enhance_factor = 1.0 + conf * torch.tanh(a)                       # [B, 1, H, W]
        H_hat = enhance_factor * high_feat                                # broadcast 到 [B,C,H,W]

        # 7) 软阈值去噪（单层设计，去掉动态调制）
        H_hat_abs = H_hat.abs()
        delta = H_hat_abs - self.tau_base                                 # 使用固定基础阈值
        gate = torch.sigmoid(self.sharpness * delta)
        H_refined = torch.sign(H_hat) * F.relu(delta) * gate

        # 8) 直接残差注入（去掉 res_scale）
        out = high_feat + H_refined
        return out