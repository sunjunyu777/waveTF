import torch
import torch.nn as nn


class LowFiLM(nn.Module):
    """
    ECA 式双流通道注意力（GAP 稳态分支 + GMP 瞬态分支 + 软路由融合）

    参考：Wang et al., "ECA-Net: Efficient Channel Attention for Deep
    Convolutional Neural Networks", CVPR 2020.

    设计思想：
    - 两种描述子互补：
        GAP —— 通道平均响应，刻画"稳态/背景趋势"，对噪声鲁棒
        GMP —— 通道最大响应，刻画"是否存在显著峰值"，对稀疏激活敏感
    
    - 稳态分支（看相似性）：
        输入 [gap_x, gap_t, gap_x·gap_t]
        → 点积捕捉 x 和 t 的对齐程度
        → 高相似 = 稳定场景
    
    - 瞬态分支（看差异）：
        输入 [gmp_x, gmp_t, gmp_x-gmp_t]
        → 差值捕捉突变
        → 大差异 = 突变场景
    
    - 自适应融合：
        两分支各自输出 logits → softmax 归一化 → 加权融合
        → 每个通道自动决定「信稳态 vs 信瞬态」
    
    - 沿通道维做 1D 卷积（kernel=k），局部跨通道交互，参数量与 C 无关
    - 末层零初始化：启动 score=0 → gate=0.5 → 输出 = 1.5·x（温和残差）

    参数量（k=7）: (3·7+1)·2 = 44，极轻。

    Args:
        num_channels: C
        k_size: 通道维 1D 卷积核大小；建议按 ECA 自适应公式
                k = odd(|log2(C)/2 + 1|)：C=384→5, C=768→7
    """
    def __init__(self, num_channels, k_size=7):
        super().__init__()
        self.num_channels = num_channels
        assert k_size % 2 == 1, "k_size must be odd"
        pad = k_size // 2

        # 稳态分支：[GAP(x), GAP(t), GAP(x)·GAP(t)] → logits_avg
        self.conv_avg = nn.Conv1d(3, 1, kernel_size=k_size, padding=pad, bias=True)
        # 瞬态分支：[GMP(x), GMP(t), GMP(x)-GMP(t)] → logits_max
        self.conv_max = nn.Conv1d(3, 1, kernel_size=k_size, padding=pad, bias=True)

        self._init_weights()

    def _init_weights(self):
        # 全部零初始化 → 启动 logits=0 → softmax 均匀 → gate=0.5 → enhanced=1.5·x
        for m in (self.conv_avg, self.conv_max):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    @staticmethod
    def _gap(z):
        # z: [B, C, H, W] -> [B, C]
        return z.mean(dim=(2, 3))

    @staticmethod
    def _gmp(z):
        # z: [B, C, H, W] -> [B, C]
        return z.amax(dim=(2, 3))

    def forward(self, x, low_freq_temporal):
        """
        Args:
            x:                 [B, C, H, W] 当前帧特征（待增强）
            low_freq_temporal: [B, C, H, W] 时序低频（全局趋势）
        Returns:
            enhanced:          [B, C, H, W]
        """
        # 1. 双描述子
        gap_x = self._gap(x)                       # [B, C]
        gap_t = self._gap(low_freq_temporal)       # [B, C]
        gmp_x = self._gmp(x)                       # [B, C]
        gmp_t = self._gmp(low_freq_temporal)       # [B, C]

        # 2. 稳态分支（看相似性）
        # 输入：[gap_x, gap_t, gap_x·gap_t (逐元素点积)]
        # - 点积高 → 同向 → 稳定
        z_avg = torch.stack([
            gap_x,
            gap_t,
            gap_x * gap_t,  # 逐元素点积（相似性）
        ], dim=1)  # [B, 3, C]
        logits_avg = self.conv_avg(z_avg).squeeze(1)  # [B, C]

        # 3. 瞬态分支（看差异）
        # 输入：[gmp_x, gmp_t, gmp_x-gmp_t (有符号差)]
        # - 差值大 → 突变
        z_max = torch.stack([
            gmp_x,
            gmp_t,
            gmp_x - gmp_t,  # 有符号差异
        ], dim=1)  # [B, 3, C]
        logits_max = self.conv_max(z_max).squeeze(1)  # [B, C]

        # 4. Softmax 归一化融合
        # 两分支 logits 做 softmax → 每个通道自动决定「信稳态 vs 信瞬态」
        logits = torch.stack([logits_avg, logits_max], dim=1)  # [B, 2, C]
        weights = torch.softmax(logits, dim=1)  # [B, 2, C] 归一化权重
        
        gate_avg = torch.sigmoid(logits_avg)  # [B, C]
        gate_max = torch.sigmoid(logits_max)  # [B, C]
        
        gate = weights[:, 0] * gate_avg + weights[:, 1] * gate_max  # [B, C] 加权融合
        gate = gate.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]

        # 6. 残差式通道加权
        return x + x * gate


if __name__ == '__main__':
    # 测试 LowFiLM
    print("=" * 50)
    print("测试 LowFiLM")
    print("=" * 50)
    
    B, C, H, W = 2, 768, 8, 8
    
    low_film = LowFiLM(num_channels=C, k_size=7)
    
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
