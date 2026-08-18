import torch
import torch.nn as nn
import torch.nn.functional as F


class LowFiLM(nn.Module):
    """
    GAP + STD 通道统计 FiLM 调制模块

    核心思想：
    - 对 low_freq_temporal 做 avg / std 两种全局统计
    - 拼接后通过共享 MLP 生成逐通道 (γ_res, β)，形状 [B, C, 1, 1]
    - output = (1 + γ_fused) * x + β_fused，broadcast [B, C, 1, 1] → [B, C, H, W]
    - MLP 最后一层零初始化 → 启动时恒等映射

    统计量语义：
        avg pool → 通道平均能量（整体密度水平）
        std pool → 通道活跃度（纹理/边缘丰富程度）
    """
    def __init__(self, num_channels, reduction=8):
        super().__init__()
        self.num_channels = num_channels
        hidden_dim = max(num_channels // reduction, 16)

        self.mlp = nn.Sequential(
            nn.Conv2d(num_channels * 2, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, num_channels * 2, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # 最后一层零初始化 → 启动时 γ_res=0, β=0 → 恒等
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x, low_freq_temporal):
        """
        GAP + STD 通道统计 FiLM：用 low_freq_temporal 生成逐通道 (γ, β) 调制 x

        参数:
            x:                [B, C, H, W] 当前帧特征
            low_freq_temporal:[B, C, H, W] 时序低频特征
        返回:
            output:           [B, C, H, W] 调制后的特征
        """
        # 两种全局统计 → [B, C, 1, 1]
        avg_p = low_freq_temporal.mean(dim=[2, 3], keepdim=True)
        std_p = low_freq_temporal.std(dim=[2, 3], keepdim=True)

        # 共享 MLP → [B, 2C, 1, 1] → split → (γ_res, β) 各 [B, C, 1, 1]
        gamma_res, beta = self.mlp(torch.cat([avg_p, std_p], dim=1)).chunk(2, dim=1)

        return (1.0 + gamma_res) * x + beta


if __name__ == '__main__':
    # 测试 LowFiLM
    print("=" * 50)
    print("测试 LowFiLM")
    print("=" * 50)
    
    B, C, H, W = 2, 768, 8, 8
    
    low_film = LowFiLM(num_channels=C, reduction=16)
    
    # 当前帧特征
    current_frame = torch.randn(B, C, H, W)
    # 时序低频特征
    low_freq_temporal = torch.randn(B, C, H, W)
    
    output = low_film(current_frame, low_freq_temporal)
    
    print(f"当前帧形状: {current_frame.shape}")
    print(f"时序低频形状: {low_freq_temporal.shape}")
    print(f"输出形状: {output.shape}")
    print(f"参数量: {sum(p.numel() for p in low_film.parameters()) / 1e6:.4f}M")
    
    # 验证残差连接
    with torch.no_grad():
        print(f"\n输入均值: {current_frame.mean().item():.6f}")
        print(f"输出均值: {output.mean().item():.6f}")
        print(f"差异: {(output - current_frame).abs().mean().item():.6f}")
