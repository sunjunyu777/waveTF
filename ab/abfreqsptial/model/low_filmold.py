import torch
import torch.nn as nn
import torch.nn.functional as F


class LowFiLM(nn.Module):
    """
    Temporal-Routed Channel Prototypes (TRCP)

    设计思想 (Mixture-of-Experts for Channel Attention):
    - 维护 K 个可学"通道门控原型"，每个 ∈ R^C（sigmoid 前的 gate logits）
      每个原型代表一种"场景模式"下的通道重要性分布
    - Router 根据 (x, temporal) 的全局特征生成 K 个原型的 softmax 权重
      → 判断"当前帧属于哪种场景"，soft-path 到对应专家
    - 最终 gate = sigmoid(原型加权组合)，残差式通道加权

    创新点：
    1. 不同于 SE/CBAM 每次从零算 gate，TRCP 预学 K 个原型复用
    2. K 个原型可独立可视化 → 对应"稠密/稀疏/静态"等场景模式
    3. 样本自适应：不同帧激活不同原型组合
    4. 参数极轻量：prototypes (K·C) + 小 router

    Args:
        num_channels: 特征通道数
        num_prototypes: 原型 (专家) 数量 K，默认 8
        reduction: router 隐层降维比例
        temperature: softmax 温度，>1 更平均，<1 更尖锐
    """
    def __init__(self, num_channels, num_prototypes=8, reduction=16, temperature=1.0):
        super().__init__()
        self.num_channels = num_channels
        self.num_prototypes = num_prototypes
        self.temperature = temperature
        hidden = max(num_channels // reduction, 32)

        # K 个通道门控原型 (sigmoid 前的 logits)，形状 [K, C]
        # 零初始化 → 启动时所有原型贡献 gate=0.5 → enhanced = 1.5·x
        self.prototypes = nn.Parameter(torch.zeros(num_prototypes, num_channels))

        # Router: [x_pool | t_pool] → K 个原型路由权重
        self.router = nn.Sequential(
            nn.Conv2d(2 * num_channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, num_prototypes, 1),
        )

        self._init_weights()

    def _init_weights(self):
        # Router 隐层 Kaiming，末层零 → 启动时 route_logits=0 → 各原型权重均匀 (1/K)
        convs = [m for m in self.router.modules() if isinstance(m, nn.Conv2d)]
        nn.init.kaiming_normal_(convs[0].weight, nonlinearity='relu')
        nn.init.zeros_(convs[0].bias)
        nn.init.zeros_(convs[-1].weight)
        nn.init.zeros_(convs[-1].bias)

    def forward(self, x, low_freq_temporal):
        """
        Args:
            x: [B, C, H, W] 当前帧特征（待增强）
            low_freq_temporal: [B, C, H, W] 时序低频（全局趋势）
        Returns:
            enhanced: [B, C, H, W]
        """
        B, C = x.size(0), x.size(1)

        # 1. 全局池化作为路由信号
        x_pool = F.adaptive_avg_pool2d(x, 1)                          # [B, C, 1, 1]
        t_pool = F.adaptive_avg_pool2d(low_freq_temporal, 1)          # [B, C, 1, 1]
        router_in = torch.cat([x_pool, t_pool], dim=1)                # [B, 2C, 1, 1]

        # 2. Router → K 个原型路由权重 (softmax)
        route_logits = self.router(router_in).view(B, self.num_prototypes)  # [B, K]
        weights = F.softmax(route_logits / self.temperature, dim=-1)         # [B, K]

        # 3. 原型加权组合 → per-sample 通道 gate logits
        gate_logits = weights @ self.prototypes                       # [B, K] @ [K, C] = [B, C]
        gate = torch.sigmoid(gate_logits).view(B, C, 1, 1)            # [B, C, 1, 1] ∈ (0, 1)

        # 4. 残差式通道注意力: gate=0 不动, gate=1 放大 2 倍
        enhanced = x + x * gate

        return enhanced

    @torch.no_grad()
    def get_prototype_gates(self):
        """可视化用：返回 K 个原型各自的 sigmoid gate，形状 [K, C]。"""
        return torch.sigmoid(self.prototypes)

    @torch.no_grad()
    def get_routing_weights(self, x, low_freq_temporal):
        """可视化用：返回当前 batch 的路由权重分布 [B, K]。"""
        x_pool = F.adaptive_avg_pool2d(x, 1)
        t_pool = F.adaptive_avg_pool2d(low_freq_temporal, 1)
        router_in = torch.cat([x_pool, t_pool], dim=1)
        B = x.size(0)
        route_logits = self.router(router_in).view(B, self.num_prototypes)
        return F.softmax(route_logits / self.temperature, dim=-1)


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
