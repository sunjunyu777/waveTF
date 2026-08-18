"""
连续自适应ESHA: 动态通道分配的交叉注意力模块

核心创新:
- ESHA的4分支设计 (Q,K,V,U): 结合全局注意力和局部卷积
- 连续自适应r值: 根据输入特征动态调整V/U通道比例
- Tanh约束: 限制r的变化范围在 ±0.1，保证稳定性
- 补0策略: 处理动态通道数，避免梯度问题
- 输入归一化: 训练推理完全一致

设计细节:
1. 密度估计器: 预测归一化偏移量 ∈ [-1, 1]
2. 动态r值: r = r_base + tanh(x) * r_delta ∈ [0.115, 0.315]
3. 通道分配: 
   - Q, K: 固定16通道（轻量级查询键）
   - V: 动态通道（全局注意力）
   - U: 动态通道（局部卷积）
4. 补0到最大通道: V→Cv_max, U→Cu_max
5. 投影融合: 1x1卷积 + GroupNorm + 残差连接
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class DeformableESHACrossAttention(nn.Module):
    """
    连续自适应ESHA交叉注意力模块
    
    核心机制:
    - ESHA的4分支设计 (Q,K,V,U)
    - 连续自适应r值: 根据输入动态调整V/U通道比例
    - Tanh约束: 限制r的变化范围在 ±0.1
    - 补0策略: 处理动态通道数
    
    参数:
        feat_dim: 特征维度, 默认256
        r_base: ESHA基准通道比例, 默认0.215
        r_delta: r的最大偏移量, 默认0.1
        dropout: Dropout比例
    """
    
    def __init__(self, feat_dim=256, r_base=0.215, r_delta=0.2, dropout=0.1):
        super().__init__()
        self.feat_dim = feat_dim
        self.r_base = r_base
        self.r_delta = r_delta  # r的最大偏移量
        
        # ESHA设计: Q,K固定为16维
        self.Cq = 16
        self.Ck = 16
        
        # r的范围: [r_base - r_delta, r_base + r_delta]
        # 默认: [0.115, 0.315]
        self.r_min = r_base - r_delta
        self.r_max = r_base + r_delta
        
        print(f"[连续自适应ESHA] 初始化:")
        print(f"  特征维度: {feat_dim}")
        print(f"  Q通道: {self.Cq}, K通道: {self.Ck}")
        print(f"  r范围: [{self.r_min:.3f}, {self.r_max:.3f}]")
        print(f"  r_base: {r_base:.3f}, r_delta: {r_delta:.3f}")
        
        # 输入归一化
        self.input_norm1 = nn.GroupNorm(num_groups=1, num_channels=feat_dim)  # LayerNorm风格
        self.input_norm2 = nn.GroupNorm(num_groups=1, num_channels=feat_dim)
        
        # 密度估计器: 预测归一化偏移量 ∈ [-1, 1]
        self.density_estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feat_dim, 64, 1),
            nn.GELU(),
            nn.Conv2d(64, 1, 1),
            nn.Tanh()  # 输出 ∈ [-1, 1]
        )
        
        # 通道重要性评分器: 为每个通道打分，软加权选出信息量最多的通道
        # 输入特征 → 每通道一个重要性分数 ∈ [0, 1]
        self.channel_scorer1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),          # [B, C, H, W] → [B, C, 1, 1]
            nn.Conv2d(feat_dim, feat_dim, 1),  # 通道间交互
            nn.Sigmoid()                       # 分数 ∈ [0, 1]
        )
        self.channel_scorer2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feat_dim, feat_dim, 1),
            nn.Sigmoid()
        )
        # 软加权后的投影层: 加权特征 [B, C, H, W] → [B, Cq/Ck, H, W]
        self.q_proj = nn.Conv2d(feat_dim, self.Cq, 1, bias=False)
        self.k_proj = nn.Conv2d(feat_dim, self.Ck, 1, bias=False)
        
        # 融合投影层：V+U拼接后(固定 feat_dim-Cq-Ck 通道) → feat_dim
        self.fusion_proj = nn.Conv2d(feat_dim - self.Cq - self.Ck, feat_dim, 1, bias=False)
        
        # 输出归一化
        self.out_norm = nn.GroupNorm(num_groups=min(32, feat_dim), num_channels=feat_dim)
        
        # 注意力缩放因子
        self.scale = self.Ck ** -0.5
        
        self.dropout = nn.Dropout(dropout)
    
    def _compute_r(self, offset_normalized):
        """
        计算动态r值
        
        参数:
            offset_normalized: [B, 1, 1, 1] ∈ [-1, 1]
        
        返回:
            r: [B, 1, 1, 1] ∈ [r_min, r_max]
            offset: [B, 1, 1, 1] 实际偏移量
        """
        # offset_normalized ∈ [-1, 1] → offset ∈ [-r_delta, +r_delta]
        offset = offset_normalized * self.r_delta
        r = self.r_base + offset
        return r, offset
        
    def forward(self, feat1, feat2):
        """
        前向传播 - 连续自适应版本
        
        参数:
            feat1: [B, C, H, W] 查询特征
            feat2: [B, C, H, W] 键值特征
            
        返回:
            output: [B, C, H, W] 增强后的特征
        """
        B, C, H, W = feat1.shape
        
        # ============ 步骤1: 输入归一化 ============
        feat1 = self.input_norm1(feat1)
        feat2 = self.input_norm2(feat2)
        
        # ============ 步骤2: 密度自适应通道分配 ============
        # 密度估计器预测归一化偏移量
        offset_normalized = self.density_estimator(feat2)  # [B, 1, 1, 1] ∈ [-1, 1]
        
        # 计算动态r值
        r, offset = self._compute_r(offset_normalized)  # [B, 1, 1, 1]
        
        # 计算当前batch的平均通道分配
        r_mean = r.mean().item()
        Cv = int(self.feat_dim * r_mean)
        Cu = self.feat_dim - self.Cq - self.Ck - Cv
        
        # ============ 步骤3: ESHA的4分支分离 ============
        # 分离4个分支: Q, K, V, U（直接从归一化后的feat1/feat2出发）
        
        # scorer1: 评估feat1各通道重要性 (Q/U复用)
        # scorer2: 评估feat2各通道重要性 (K/V复用)
        scores1 = self.channel_scorer1(feat1).squeeze(-1).squeeze(-1)  # [B, C]
        scores2 = self.channel_scorer2(feat2).squeeze(-1).squeeze(-1)  # [B, C]
        
        # Q: feat1软加权 + 投影到Cq
        Q = self.q_proj(feat1 * scores1.unsqueeze(-1).unsqueeze(-1))  # [B, Cq, H, W]
        
        # K: feat2软加权 + 投影到Ck
        K = self.k_proj(feat2 * scores2.unsqueeze(-1).unsqueeze(-1))  # [B, Ck, H, W]
        
        # V: 从归一化后feat2 Top-K，选Cv个重要性最高的通道
        v_idx = torch.topk(scores2, Cv, dim=1).indices  # [B, Cv]
        V = torch.gather(feat2, 1, v_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W))  # [B, Cv, H, W]
        
        # U: 从归一化后feat1 Top-K，选Cu个重要性最高的通道
        u_idx = torch.topk(scores1, Cu, dim=1).indices  # [B, Cu]
        U = torch.gather(feat1, 1, u_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W))  # [B, Cu, H, W]
        
        # ============ 步骤4: 全局注意力 ============
        Q_flat = Q.flatten(2).transpose(1, 2)  # [B, HW, 16]
        K_flat = K.flatten(2)  # [B, 16, HW]
        V_flat = V.flatten(2).transpose(1, 2)  # [B, HW, Cv]
        
        attn = torch.bmm(Q_flat, K_flat) * self.scale  # [B, HW, HW]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        attn_output = torch.bmm(attn, V_flat)  # [B, HW, Cv]
        attn_output = attn_output.transpose(1, 2).reshape(B, Cv, H, W)
        
        # ============ 步骤5: 拼接V和U ============
        U_activated = F.gelu(U)
        concat = torch.cat([attn_output, U_activated], dim=1)  # [B, Cv+Cu, H, W]
        
        # ============ 步骤6: 融合投影 ============
        output = self.fusion_proj(concat)  # [B, max_channels, H, W] → [B, feat_dim, H, W]
        output = self.out_norm(output)
        
        # 残差连接
        output = output + feat1
        
        return output
    
    def get_r_value(self, feat2):
        """
        获取当前的r值 (用于可视化和调试)
        
        参数:
            feat2: [B, C, H, W] 输入特征
            
        返回:
            dict: 包含r值、offset等信息
        """
        with torch.no_grad():
            offset_normalized = self.density_estimator(feat2)
            r, offset = self._compute_r(offset_normalized)
            
            return {
                'r': r.squeeze().cpu().numpy(),
                'offset': offset.squeeze().cpu().numpy(),
                'offset_normalized': offset_normalized.squeeze().cpu().numpy(),
                'r_min': self.r_min,
                'r_max': self.r_max,
                'r_base': self.r_base
            }


# 测试代码
if __name__ == '__main__':
    print("=" * 60)
    print("测试 连续自适应ESHA (DeformableESHACrossAttention)")
    print("=" * 60)
    
    # 创建模块
    model = DeformableESHACrossAttention(
        feat_dim=256,
        r_base=0.215,
        r_delta=0.1,
        dropout=0.1
    )
    
    # 创建测试数据
    B, C, H, W = 2, 256, 32, 32
    feat1 = torch.randn(B, C, H, W)  # 空间特征
    feat2 = torch.randn(B, C, H, W)  # 频域特征
    
    print(f"\n输入形状:")
    print(f"  feat1 (空间): {feat1.shape}")
    print(f"  feat2 (频域): {feat2.shape}")
    
    # 前向传播
    print(f"\n执行前向传播...")
    output = model(feat1, feat2)
    
    print(f"\n输出形状: {output.shape}")
    
    # 检查档位选择
    print(f"\n档位选择情况:")
    selection = model.get_level_selection(feat2)
    print(f"  选中档位: {selection['selected_level']}")
    print(f"  概率分布: {selection['probs'][0]}")
    for i, (r, Cv, Cu) in enumerate(zip(selection['r_levels'], 
                                         selection['Cv_levels'], 
                                         selection['Cu_levels'])):
        print(f"  档位{i}: r={r:.3f}, Cv={Cv}, Cu={Cu}, prob={selection['probs'][0][i]:.3f}")
    
    # 测试不同场景的档位选择
    print(f"\n" + "=" * 60)
    print("测试不同场景的档位选择")
    print("=" * 60)
    
    # 稀疏场景
    feat2_sparse = torch.randn(B, C, H, W) * 0.1
    selection_sparse = model.get_level_selection(feat2_sparse)
    print(f"\n稀疏场景:")
    print(f"  选中档位: {selection_sparse['selected_level'][0]}")
    print(f"  概率: {selection_sparse['probs'][0]}")
    
    # 密集场景
    feat2_dense = torch.randn(B, C, H, W) * 2.0
    selection_dense = model.get_level_selection(feat2_dense)
    print(f"\n密集场景:")
    print(f"  选中档位: {selection_dense['selected_level'][0]}")
    print(f"  概率: {selection_dense['probs'][0]}")
    
    # 验证初始化: 应该偏向中档(index=1)
    print(f"\n验证初始化 (应该偏向中档):")
    print(f"  选中档位: {selection['selected_level'][0]}")
    print(f"  中档概率: {selection['probs'][0][1]:.4f}")
    print(f"  ✓ 初始化成功! 偏向中档" if selection['selected_level'][0] == 1 else f"  ⚠ 选中档位{selection['selected_level'][0]}, 可能需要检查初始化")
    
    print(f"\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
