import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入可变形卷积
from torchvision.ops import DeformConv2d


# ============ CBAM 2D 定义（根据官方论文）============

class ChannelGate(nn.Module):
    """通道注意力门控"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
    
    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return torch.sigmoid(avg_out + max_out)


class SpatialGate(nn.Module):
    """空间注意力门控"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        return torch.sigmoid(self.conv(out))


class CBAM(nn.Module):
    """
    CBAM: Convolutional Block Attention Module
    论文: "CBAM: Convolutional Block Attention Module" (ECCV 2018)
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_gate = ChannelGate(channels, reduction)
        self.spatial_gate = SpatialGate()
    
    def forward(self, x):
        x = x * self.channel_gate(x)  # 通道注意力
        x = x * self.spatial_gate(x)  # 空间注意力
        return x


# ============ 3D 版本的注意力模块 ============

class ChannelAttention3D(nn.Module):
    """
    3D通道注意力模块
    
    对时空特征的通道维度进行注意力加权
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        # 共享MLP
        self.mlp = nn.Sequential(
            nn.Conv3d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction, channels, 1, bias=False)
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, C, T, H, W]
        Returns:
            channel_att: [B, C, 1, 1, 1]
        """
        # 平均池化和最大池化
        avg_out = self.mlp(self.avg_pool(x))  # [B, C, 1, 1, 1]
        max_out = self.mlp(self.max_pool(x))  # [B, C, 1, 1, 1]
        
        # 融合并激活
        channel_att = torch.sigmoid(avg_out + max_out)
        return channel_att


class SpatialAttention3D(nn.Module):
    """
    3D空间注意力模块
    
    对时空特征的空间维度进行注意力加权
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        
        # 3D卷积生成空间注意力
        self.conv = nn.Conv3d(2, 1, kernel_size=(1, kernel_size, kernel_size), 
                             padding=(0, padding, padding), bias=False)
        
    def forward(self, x):
        """
        Args:
            x: [B, C, T, H, W]
        Returns:
            spatial_att: [B, 1, T, H, W]
        """
        # 在通道维度上做平均池化和最大池化
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, T, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, T, H, W]
        
        # 拼接
        concat = torch.cat([avg_out, max_out], dim=1)  # [B, 2, T, H, W]
        
        # 卷积生成空间注意力
        spatial_att = torch.sigmoid(self.conv(concat))  # [B, 1, T, H, W]
        return spatial_att


class CBAM3D(nn.Module):
    """
    3D CBAM: 通道注意力 + 空间注意力
    
    先通道后空间的顺序
    """
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention3D(channels, reduction)
        self.spatial_att = SpatialAttention3D(kernel_size)
        
    def forward(self, x):
        """
        Args:
            x: [B, C, T, H, W]
        Returns:
            out: [B, C, T, H, W] 注意力增强后的特征
        """
        # 通道注意力
        x = x * self.channel_att(x)  # [B, C, T, H, W]
        
        # 空间注意力
        x = x * self.spatial_att(x)  # [B, C, T, H, W]
        
        return x


class Temporal3DModulation(nn.Module):
    """
    DCN + 双层 FiLM 时序调制模块
    
    设计思路：
    1. DCN 对齐前一帧到当前帧（处理空间错位）
    2. 通道 FiLM：全局特征调制（哪些通道重要）
    3. 空间 FiLM：局部特征调制（哪些位置重要）
    4. 残差连接
    
    创新点：
    - 三层调制机制：DCN（对齐）+ 通道FiLM（全局）+ 空间FiLM（局部）
    - 轻量高效：FiLM 参数量小，计算快
    - 可解释性：γ/β 可视化，分析调制模式
    
    优势：
    - 分工明确：DCN 处理几何变换，FiLM 处理特征调制
    - 适合两帧：不需要长期记忆，专注于两帧关系
    - 鲁棒性强：DCN 处理空间错位，FiLM 处理语义调制
    """
    def __init__(self, in_channels, prev_channels=3, hidden_channels=None, reduction=16):
        """
        Args:
            in_channels: 当前帧通道数
            prev_channels: 前一帧通道数（默认3，原始RGB图像）
            hidden_channels: 未使用（保持接口兼容）
            reduction: 通道 FiLM 的缩减比例
        """
        super().__init__()
        self.in_channels = in_channels
        self.prev_channels = prev_channels
        self.reduction = reduction
        
        # 1. 前一帧对齐层：prev_channels -> in_channels
        # 多层渐进式对齐，增强学习能力（特别是 3 -> 768 的大跨度）
        mid_channels = (prev_channels + in_channels) // 2
        self.prev_align = nn.Sequential(
            nn.Conv2d(prev_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(32, mid_channels), num_channels=mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(32, in_channels), num_channels=in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=min(32, in_channels), num_channels=in_channels)
        )
        
        # 2. offset学习网络：学习可变形卷积的偏移量
        self.offset_conv = nn.Conv2d(
            in_channels, 18,  # 3x3卷积核有9个点，每个点2维(x,y)偏移 = 18
            kernel_size=3, padding=1
        )
        
        # 3. 可变形卷积对齐：使用学到的offset对齐前一帧
        self.dcn_align = DeformConv2d(
            in_channels, in_channels,
            kernel_size=3, stride=1, padding=1
        )
        
        # === 双层 FiLM 调制 ===
        # 4. 通道 FiLM 生成器：全局特征调制
        self.channel_film = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局池化 [B, C, H, W] → [B, C, 1, 1]
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels * 2, 1)  # γ_c 和 β_c
        )
        
        # 5. 空间 FiLM 生成器：局部特征调制
        self.spatial_film = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(32, in_channels // 2), num_channels=in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels * 2, kernel_size=1)  # γ_s 和 β_s
        )
        
    def forward(self, frame_prev, frame_current):
        """
        Args:
            frame_prev: 前一帧 [B, prev_C, H_prev, W_prev] (原始图像，通道数可能是3)
            frame_current: 当前帧（后一帧）[B, C, H, W] (特征图)
        
        Returns:
            output: 时空调制后的特征 [B, C, H, W]
        """
        B, C, H, W = frame_current.shape
        
        # 步骤1: 前一帧对齐（空间+通道）
        import torch.nn.functional as F
        frame_prev_resized = F.interpolate(frame_prev, size=(H, W), mode='bilinear', align_corners=False)
        frame_prev_aligned = self.prev_align(frame_prev_resized)  # [B, C, H, W]
        
        # 步骤2: 学习offset并用可变形卷积对齐前一帧
        offset = self.offset_conv(frame_prev_aligned)  # [B, 18, H, W]
        aligned_prev = self.dcn_align(frame_prev_aligned, offset)  # [B, C, H, W]
        
        # === 双层 FiLM 调制 ===
        # 步骤3: 通道 FiLM 调制（全局特征调制）
        channel_params = self.channel_film(aligned_prev)  # [B, 2C, 1, 1]
        gamma_c, beta_c = torch.chunk(channel_params, 2, dim=1)  # 各 [B, C, 1, 1]
        
        # 通道调制：γ_c 控制通道重要性，β_c 控制通道偏置
        feat_c = gamma_c * frame_current + beta_c  # [B, C, H, W]
        
        # 步骤4: 空间 FiLM 调制（局部特征调制，作用于通道调制后的特征）
        spatial_params = self.spatial_film(aligned_prev)  # [B, 2C, H, W]
        gamma_s, beta_s = torch.chunk(spatial_params, 2, dim=1)  # 各 [B, C, H, W]
        
        # 空间调制：γ_s 控制空间位置重要性，β_s 控制空间偏置
        feat_s = gamma_s * feat_c + beta_s  # [B, C, H, W]
        
        # 步骤5: 残差连接（将调制后的特征与原始特征融合）
        output = feat_s + frame_current  # [B, C, H, W]
        
        return output


# 测试代码
if __name__ == '__main__':
    print("=== DCN + 双层 FiLM 时序调制测试 ===\n")
    
    # 创建模块
    model = Temporal3DModulation(in_channels=256, prev_channels=3, reduction=16)
    
    # 测试数据
    frame_prev = torch.randn(2, 3, 512, 512)      # RGB 原图（前一帧）
    frame_current = torch.randn(2, 256, 64, 64)   # 特征图（当前帧）
    
    # 前向传播
    output = model(frame_prev, frame_current)
    
    print(f"输入前一帧（RGB）: {frame_prev.shape}")
    print(f"输入当前帧（特征）: {frame_current.shape}")
    print(f"输出调制后的特征: {output.shape}")
    
    # 验证输出形状
    assert output.shape == frame_current.shape, "输出形状应与当前帧相同"
    
    # 验证调制效果（输出不应该等于输入）
    assert not torch.allclose(output, frame_current), "调制应该改变特征"
    
    print("\n✓ 测试通过！")
    print("\n调制机制：")
    print("1. DCN 对齐：处理前一帧的空间错位")
    print("2. 通道 FiLM：全局调制（哪些通道重要）")
    print("3. 空间 FiLM：局部调制（哪些位置重要）")
    print("4. 残差连接：保留原始信息")
