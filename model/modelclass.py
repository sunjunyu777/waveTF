import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.ops import DeformConv2d
import re
from timm.models.layers import trunc_normal_, DropPath
from einops import rearrange
import os
import urllib.request
from tqdm import tqdm
import math
from typing import Optional
from pytorch_wavelets import DWTForward
from .wave import TwoFrameDWT
from .convnext import convnext_tiny
from .temporal_3d_modulation import Temporal3DModulation
from .deformable_esha import DeformableESHACrossAttention
from .film_modulation import SpatialFiLM
from .global_cross_attn import AdaptiveHighFreqEnhancement
from .maxpoolingmap import MaxPoolingMap
from .dual_branch_modulation import DualBranchCompetitiveModulation
from .low_film import LowFiLM
from .spatial_prompt import SpatialPromptCrossAttn


class WaveletHighFreqPoolingFusion(nn.Module):
    """
    简化的小波低频提取模块
    
    只保留低频分量，去掉高频调制
    
    流程:
    1. 小波分解得到低频(LL)
    2. 上采样到原分辨率
    3. 直接返回低频特征
    """
    def __init__(self, in_chans=3, wavelet='haar', mode='symmetric'):
        super().__init__()
        
        self.in_chans = in_chans
        
        # 小波分解
        self.dwt = DWTForward(J=1, wave=wavelet, mode=mode)
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] 输入视频帧
        Returns:
            out: [B, C, H, W] 低频特征（上采样到原分辨率）
            high_freq_mask: None（保持接口兼容）
        """
        B, C, H, W = x.shape
        
        # 1. 小波分解，只保留低频
        yl, _ = self.dwt(x)  # yl: [B, C, H/2, W/2]
        
        # 2. 上采样低频到原分辨率
        out = F.interpolate(yl, size=(H, W), mode='bilinear', align_corners=False)  # [B, C, H, W]
        
        # 3. 返回低频特征，high_freq_mask设为None（保持接口兼容）
        return out, None



class TemporalDFGF(nn.Module):
    """
    Temporal Dynamic Guided Filter 基于 DCNv2 的实现。

    - 目标：利用时序高频特征在空间上提供可变形采样位置和通道权重，动态重组低频图像；
    - 结果：输出与输入低频同形状，可直接作为后续主干的低频特征。
    """

    def __init__(
        self,
        in_channels: int,
        guide_channels: Optional[int] = None,
        kernel_size: int = 3,
        hidden_channels: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.guide_channels = guide_channels or in_channels
        self.kernel_size = kernel_size

        if hidden_channels is None:
            hidden_channels = max(1, in_channels // 2)
        self.hidden_channels = hidden_channels

        padding = kernel_size // 2
        # 降维层（加激活函数）
        self.reduce_pw = nn.Sequential(
            nn.Conv2d(in_channels, self.hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=self.hidden_channels),  # LayerNorm
            nn.GELU()
        )
        # 升维层（加激活函数）
        self.expand_pw = nn.Sequential(
            nn.Conv2d(self.hidden_channels, in_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=in_channels),  # LayerNorm
            nn.GELU()
        )

        # 标准可变形卷积（去掉 Depthwise）
        self.deform_conv = DeformConv2d(
            self.hidden_channels,
            self.hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=1,  # 标准卷积，允许通道间交互
            bias=False,
        )

        offset_channels = 2 * kernel_size * kernel_size
        mask_channels = kernel_size * kernel_size
        self.offset_proj = nn.Conv2d(self.guide_channels, offset_channels, kernel_size=3, padding=1, bias=True)
        self.mask_proj = nn.Conv2d(self.guide_channels, mask_channels, kernel_size=3, padding=1, bias=True)
        nn.init.constant_(self.offset_proj.weight, 0.0)
        nn.init.constant_(self.offset_proj.bias, 0.0)
        nn.init.constant_(self.mask_proj.weight, 0.0)
        nn.init.constant_(self.mask_proj.bias, 0.0)

    def forward(self, low_freq: torch.Tensor, high_freq_guidance: torch.Tensor) -> torch.Tensor:
        """
        Args:
            low_freq: 低频特征 [B, C, H, W]
            high_freq_guidance: 高频引导特征 [B, C, H, W]
        Returns:
            out: 动态滤波后的特征 [B, C, H, W]
        """
        offsets = self.offset_proj(high_freq_guidance)
        masks = torch.sigmoid(self.mask_proj(high_freq_guidance))
        low_mid = self.reduce_pw(low_freq)
        filtered_mid = self.deform_conv(low_mid, offsets, masks)
        out = self.expand_pw(filtered_mid)
        # 残差连接：保留原始低频信息
        out = out + low_freq
        return out


class LowFreqAdapter(nn.Module):
    """
    局部窗口Transformer交叉注意力模块
    
    对比文献[35]的余弦相似度:
    - 文献: Wacos = softmax(a·cosine(Ft-1, Ft_nbr) + b)
    - 我们: Attn = softmax(Q·K^T / √d)
    
    功能: 用kv_feat的邻域信息增强query_feat
    """
    def __init__(self, num_channels=3, num_heads=4, window_size=3):
        """
        Args:
            num_channels: 输入通道数
            num_heads: 多头注意力头数
            window_size: 局部窗口大小 (K×K邻域)
        """
        super().__init__()
        self.num_channels = num_channels
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = num_channels // num_heads
        assert num_channels % num_heads == 0, "num_channels must be divisible by num_heads"
        
        # Q/K/V投影
        self.q_proj = nn.Conv2d(num_channels, num_channels, 1)
        self.k_proj = nn.Conv2d(num_channels, num_channels, 1)
        self.v_proj = nn.Conv2d(num_channels, num_channels, 1)
        
        # 输出投影
        self.out_proj = nn.Conv2d(num_channels, num_channels, 1)
        
        # 缩放因子
        self.scale = self.head_dim ** -0.5
        
        # 相对位置编码：为 K×K 窗口定义可学习的位置偏置
        # 对于 3×3 窗口，有 9 个位置，每个位置可以关注 9 个邻域
        self.relative_pos_bias = nn.Parameter(
            torch.zeros(window_size * window_size, window_size * window_size)
        )
        # 初始化：小的随机值，避免初期影响过大
        nn.init.trunc_normal_(self.relative_pos_bias, std=0.02)
    
    def forward(self, query_feat, kv_feat):
        """
        用kv_feat增强query_feat
        
        Args:
            query_feat: [B, C, H, W] 被增强的特征
            kv_feat: [B, C, H, W] 提供信息的特征
        
        Returns:
            enhanced: [B, C, H, W] 增强后的query_feat
        """
        B, C, H, W = query_feat.shape
        K = self.window_size
        
        # Q投影
        Q = self.q_proj(query_feat)  # [B, C, H, W]
        Q = Q.view(B, self.num_heads, self.head_dim, H, W)  # [B, num_heads, head_dim, H, W]
        Q = Q.flatten(3)  # [B, num_heads, head_dim, H*W]
        Q = Q.transpose(2, 3)  # [B, num_heads, H*W, head_dim]
        
        # K/V投影并unfold获取邻域
        K_feat = self.k_proj(kv_feat)  # [B, C, H, W]
        V_feat = self.v_proj(kv_feat)  # [B, C, H, W]
        
        # Unfold: 提取K×K邻域
        K_unfold = F.unfold(K_feat, kernel_size=K, padding=K//2)  # [B, C*K*K, H*W]
        V_unfold = F.unfold(V_feat, kernel_size=K, padding=K//2)  # [B, C*K*K, H*W]
        
        # Reshape: [B, C*K*K, H*W] -> [B, C, K*K, H*W] -> [B, num_heads, head_dim, K*K, H*W]
        # unfold输出格式: 先遍历通道C，再遍历窗口K*K
        K_unfold = K_unfold.view(B, C, K*K, H*W)  # [B, C, K*K, H*W]
        V_unfold = V_unfold.view(B, C, K*K, H*W)  # [B, C, K*K, H*W]
        # 再分多头
        K_unfold = K_unfold.view(B, self.num_heads, self.head_dim, K*K, H*W)
        V_unfold = V_unfold.view(B, self.num_heads, self.head_dim, K*K, H*W)
        
        # Transpose: [B, num_heads, H*W, K*K, head_dim]
        K_unfold = K_unfold.permute(0, 1, 4, 3, 2)  # [B, num_heads, H*W, K*K, head_dim]
        V_unfold = V_unfold.permute(0, 1, 4, 3, 2)  # [B, num_heads, H*W, K*K, head_dim]
        
        # 计算注意力（内容相似度）
        # Q: [B, num_heads, H*W, head_dim]
        # K: [B, num_heads, H*W, K*K, head_dim]
        attn = torch.einsum('bhnq,bhnkq->bhnk', Q, K_unfold) * self.scale  # [B, num_heads, H*W, K*K]
        
        # 加入相对位置编码（广播到所有 batch 和 head）
        # relative_pos_bias: [K*K, K*K] -> [1, 1, K*K, K*K]
        # attn: [B, num_heads, H*W, K*K]
        # 注意：这里假设每个位置的中心都是第 4 个位置（K=3 时）
        # 所以只需要取 relative_pos_bias 的中心行
        center_idx = K * K // 2  # 中心位置索引（对于 3x3 是 4）
        pos_bias = self.relative_pos_bias[center_idx:center_idx+1, :]  # [1, K*K]
        attn = attn + pos_bias.unsqueeze(0).unsqueeze(0)  # [B, num_heads, H*W, K*K]
        
        # Softmax
        attn = F.softmax(attn, dim=-1)  # [B, num_heads, H*W, K*K]
        
        # 加权聚合
        out = torch.einsum('bhnk,bhnkq->bhnq', attn, V_unfold)  # [B, num_heads, H*W, head_dim]
        
        # 合并多头
        out = out.transpose(2, 3).contiguous()  # [B, num_heads, head_dim, H*W]
        out = out.view(B, C, H, W)  # [B, C, H, W]
        out = self.out_proj(out)  # [B, C, H, W]
        
        # 残差连接: 在原特征基础上增强
        out = out + query_feat
        
        return out



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



class AdvancedTwoPathFusion(nn.Module):
    """
    高级两路融合模块（时频分支 + 频空分支）
    
    创新点：
    1. **双向FiLM调制**：两个分支互相生成仿射变换参数
       - z_tf → (γ_fs, β_fs) 调制 z_fs
       - z_fs → (γ_tf, β_tf) 调制 z_tf
       - 通道级别的动态调制，比空间注意力更灵活
    
    2. **轻量级参数生成器**：
       - 全局平均池化 → MLP → 生成 (scale, shift)
       - 参数量小，计算高效
       - 类似于SENet的通道注意力
    
    3. **对称式交叉调制**：
       - 两个分支地位平等，互相增强
       - 调制后拼接降维，融合互补信息
    
    理论依据：
    - FiLM (Feature-wise Linear Modulation): 条件特征调制
    - 交叉注意力：用一个分支的全局信息指导另一个分支
    - 双向信息流：避免单向依赖，增强鲁棒性
    """
    
    def __init__(self, num_channels, reduction=16):
        """
        Args:
            num_channels: 输入特征的通道数
            reduction: FiLM参数生成器的降维比例
        """
        super().__init__()
        self.num_channels = num_channels
        
        # 独立卷积处理
        self.conv_tf = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.conv_fs = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        
        # FiLM参数生成器
        hidden = max(num_channels // reduction, 8)
        self.film_gen_for_fs = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B, C, 1, 1]
            nn.Conv2d(num_channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, 2 * num_channels, 1)  # 生成 (γ, β)
        )
        self.film_gen_for_tf = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B, C, 1, 1]
            nn.Conv2d(num_channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, 2 * num_channels, 1)  # 生成 (γ, β)
        )
        
        # 最终融合（2C -> C）
        self.conv_concat = nn.Conv2d(num_channels * 2, num_channels, 1)
    
    def forward(self, z_tf, z_fs):
        """
        Args:
            z_tf: [B, C, H, W] 时频分支 (temporal_s4)
            z_fs: [B, C, H, W] 频空分支 (spatial_s4_fused)
        
        Returns:
            y: [B, C, H, W] 融合后的特征
        """
        # 1. 独立卷积处理
        z_tf_proc = self.conv_tf(z_tf)  # [B, C, H, W]
        z_fs_proc = self.conv_fs(z_fs)  # [B, C, H, W]
        
        # 2. 生成FiLM参数
        # 2.1) 用z_tf生成调制z_fs的参数
        film_params_fs = self.film_gen_for_fs(z_tf_proc)  # [B, 2C, 1, 1]
        gamma_fs, beta_fs = torch.chunk(film_params_fs, 2, dim=1)  # 各 [B, C, 1, 1]
        
        # 2.2) 用z_fs生成调制z_tf的参数
        film_params_tf = self.film_gen_for_tf(z_fs_proc)  # [B, 2C, 1, 1]
        gamma_tf, beta_tf = torch.chunk(film_params_tf, 2, dim=1)  # 各 [B, C, 1, 1]
        
        # 3. FiLM调制：y = γ * x + β
        z_tf_modulated = gamma_tf * z_tf_proc + beta_tf  # [B, C, H, W]
        z_fs_modulated = gamma_fs * z_fs_proc + beta_fs  # [B, C, H, W]
        
        # 4. 拼接调制后的特征并降维
        concat = torch.cat([z_tf_modulated, z_fs_modulated], dim=1)  # [B, 2C, H, W]
        y = self.conv_concat(concat)  # [B, C, H, W]
        
        return y


class LoRAAdapter(nn.Module):
    """
    轻量低秩残差 adapter (LoRA-style)：
        out = x + Up(Down(x))
    
    用途：吸收"冻结 backbone 输出"和"任务所需特征"之间的分布差异。
    零初始化 Up 层 → 启动时恒等 (out = x)，可无痛插入已训模型。
    
    参数量极少：2 · dim · rank
    """
    def __init__(self, dim, rank=16):
        super().__init__()
        self.down = nn.Conv2d(dim, rank, 1, bias=False)
        self.up   = nn.Conv2d(rank, dim, 1, bias=False)
        nn.init.kaiming_normal_(self.down.weight)
        nn.init.zeros_(self.up.weight)  # 零初始化 → 启动 out = x

    def forward(self, x):
        return x + self.up(self.down(x))


class VCC(nn.Module):
    def __init__(self, in_chans=3, out_chans=1, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., layer_scale_init_value=1e-6, head_init_scale=1.,
                 load_pretrained=True, hidden_dim=192, num_heads=8, save_wave_images=False, window_len=4,
                 convnext_in_22k=False, debug_high_freq=False):
        super().__init__()
        
        # 两帧时序 DWT（时间维）
        self.two_frame_dwt = TwoFrameDWT(
            wavelet='haar', 
            mode='symmetric',
            save_wave_images=save_wave_images
        )
        
        # 空间小波分解 + 高频多尺度池化融合
        self.wavelet_fusion = WaveletHighFreqPoolingFusion(
            in_chans=in_chans,
            wavelet='haar',
            mode='symmetric'
        )

        # 共享主干网络：ConvNeXt-Tiny（原图与频域图共用同一份权重）
        # 标准配置 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]，与 ImageNet-1k/22k 预训练权重匹配
        self.backbone = convnext_tiny(
            pretrained=load_pretrained,
            in_22k=convnext_in_22k,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
        )
        dims = self.backbone.dims  # [96, 192, 384, 768]
        if load_pretrained:
            tag = "ImageNet-22k" if convnext_in_22k else "ImageNet-1k"
            n_params = sum(p.numel() for p in self.backbone.parameters()) / 1e6
            print(f"[VCC] ConvNeXt-Tiny 预训练权重加载完成 | 来源: {tag} | 参数量: {n_params:.2f}M | dims: {dims}")
        else:
            print(f"[VCC] ConvNeXt-Tiny 随机初始化 | dims: {dims}")

        # ===== 频域分支共享 backbone 的 Stage3+4（no_grad 模式，不更新主干权重）=====
        # freq_stem: 代替主干的 Stage1+2，把 3ch 频域图下采样到 dims[1]×H/8×W/8
        # 参数量：~78K（4×4 stride-4 + 2×2 stride-2）
        self.freq_stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2),
            LayerNorm(dims[1], eps=1e-6, data_format="channels_first"),
            nn.GELU(),
        )
        # Adapter：吸收 RGB↔频域分布差（LoRA-style，零初始化启动恒等）
        self.freq_adapter_s3 = LoRAAdapter(dim=dims[2], rank=16)  # 384ch
        self.freq_adapter_s4 = LoRAAdapter(dim=dims[3], rank=16)  # 768ch
        
        # TemporalDFGF: 高频引导 frame2_s4
        self.temporal_dfgf_s4 = TemporalDFGF(
            in_channels=dims[3],
            guide_channels=dims[3],
            kernel_size=3
        )
        
        # Stage4: ECA 式双流通道注意力（GAP+GMP+软路由）
        self.low_film_s4 = LowFiLM(
            num_channels=dims[3],  # 768
            k_size=7,              # ECA 自适应：log2(768)/2+1 ≈ 5.78 → odd 7
        )
        
        # Stage4: 密度引导的空间交叉注意力模块
        self.spatial_density_s4 = SpatialPromptCrossAttn(
            num_channels=dims[3]  # 768
        )

        # Stage4: 高级两路融合模块
        self.advanced_fusion_s4 = AdvancedTwoPathFusion(num_channels=dims[3])  # 768
        
        # Stage4: 降维层（残差连接：先相加再降维）
        self.reduce_s4 = nn.Sequential(
            nn.Conv2d(dims[3], dims[2], kernel_size=1),      # 768 -> 384
            nn.GroupNorm(num_groups=1, num_channels=dims[2]),
            nn.GELU()
        )
        
        
        # TemporalDFGF: 高频引导 s4_to_s3
        self.temporal_dfgf_s3 = TemporalDFGF(
            in_channels=dims[2],
            guide_channels=dims[2],
            kernel_size=3
        )
        
        # Stage3: ECA 式双流通道注意力（GAP+GMP+软路由）
        self.low_film_s3 = LowFiLM(
            num_channels=dims[2],  # 384
            k_size=5,              # ECA 自适应：log2(384)/2+1 ≈ 5.29 → odd 5
        )
        
        # Stage3: 密度引导的空间交叉注意力模块
        self.spatial_density_s3 = SpatialPromptCrossAttn(
            num_channels=dims[2]  # 384
        )
        
        # Stage3: 高级两路融合模块
        self.advanced_fusion_s3 = AdvancedTwoPathFusion(num_channels=dims[2])  # 384
        
        # Stage3: 降维层（残差连接：先相加再降维）
        self.reduce_s3 = nn.Sequential(
            nn.Conv2d(dims[2], dims[2], kernel_size=1),      # 384 -> 384
            LayerNorm(dims[2], eps=1e-6, data_format="channels_first"),
            nn.GELU()
        )
        
        # 密度头输入: 融合后的特征 (dims[2] = 384)
        self.regression_head = nn.Sequential(
            nn.Conv2d(dims[2], 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_chans, kernel_size=1)
        )
    
    def extract_features(self, x, backbone, gates=None):
        """使用指定backbone提取四阶段特征（不再在阶段入口做门控）。"""
        if hasattr(backbone, "forward_features_list"):
            return backbone.forward_features_list(x)

        features = []

        # Stage1（1/4）
        x = backbone.downsample_layers[0](x)
        x = backbone.stages[0](x)
        features.append(x)

        # Stage2（1/8）
        x = backbone.downsample_layers[1](x)
        x = backbone.stages[1](x)
        features.append(x)

        # Stage3（1/16）
        x = backbone.downsample_layers[2](x)
        x = backbone.stages[2](x)
        features.append(x)

        # Stage4（1/32）
        x = backbone.downsample_layers[3](x)
        x = backbone.stages[3](x)
        features.append(x)

        return features 

    def forward(self, frames):
        """
        两帧输入 -> 时序DWT -> 空间DWT引导 HighAttention -> 共享主干 -> 分级 DFGF 残差 -> 单帧密度
        
        Args:
            frames: [2, C, H, W]
        Returns:
            density: Tensor [1, H/8, W/8] 仅输出第二帧的密度图
        """
        assert frames.dim() == 4 and frames.shape[0] == 2, "frames 必须为 [2, C, H, W]"
        _, C, H, W = frames.shape

        # 取两帧（batch 视作 1）
        frame1 = frames[0:1]  # [1, C, H, W]
        frame2 = frames[1:2]  # [1, C, H, W]

        # 1) 时序 DWT（两帧）得到单图低频与高频（同分辨率）
        low_t, high_t = self.two_frame_dwt(frame1, frame2)  # [1, 3, H, W]
        
        # 1.5) 对 frame2 进行空间小波分解 + 高频池化融合
        frame2_freq, high_freq_mask = self.wavelet_fusion(frame2)  # [1, 3, H, W], [1, 1, H, W]
        
        # 2) 分别编码 frame2 和 frame2_freq
        frame2_features = self.extract_features(frame2, self.backbone)
        frame2_stage1, frame2_stage2, frame2_stage3, frame2_stage4 = frame2_features
        
        frame2_freq_features = self.extract_features(frame2_freq, self.backbone)
        frame2_freq_s1, frame2_freq_s2, frame2_freq_s3, frame2_freq_s4 = frame2_freq_features

        # ===== 频域对齐：共享 backbone Stage3+4 (no_grad) + LoRA adapter =====
        # 将 low_t 和 high_t 拼 batch 一次性过 backbone，节省一次前向
        freq_in = torch.cat([low_t, high_t], dim=0)                       # [2, 3, H, W]
        freq_stem_out = self.freq_stem(freq_in)                           # [2, 192, H/8, W/8]
        with torch.no_grad():
            # 借用主干 Stage3 权重 (不更新)
            freq_s3_raw = self.backbone.downsample_layers[2](freq_stem_out)
            freq_s3_raw = self.backbone.stages[2](freq_s3_raw)            # [2, 384, H/16, W/16]
            # 借用主干 Stage4 权重 (不更新)
            freq_s4_raw = self.backbone.downsample_layers[3](freq_s3_raw)
            freq_s4_raw = self.backbone.stages[3](freq_s4_raw)            # [2, 768, H/32, W/32]
        # Adapter 吸收 RGB↔freq 分布差 (detach 切断梯度回到主干)
        freq_s3_adapted = self.freq_adapter_s3(freq_s3_raw.detach())
        freq_s4_adapted = self.freq_adapter_s4(freq_s4_raw.detach())
        # 拆回低频/高频
        low_t_s3, high_t_s3 = freq_s3_adapted.chunk(2, dim=0)             # each [1, 384, H/16, W/16]
        low_t_s4, high_t_s4 = freq_s4_adapted.chunk(2, dim=0)             # each [1, 768, H/32, W/32]

        # ===== Stage4: FiLM调制 + TemporalDFGF + 双分支融合 =====
        
        # 3.2) 用低频相似度引导调制 frame2_freq 的第四阶段
        low_s4_fused = self.low_film_s4(
            x=frame2_freq_s4,           # 单图频域特征的第四阶段
            low_freq_temporal=low_t_s4  # 时序低频
        )  # [1, 768, H/32, W/32]
        
        # 3.3) 高频分支：TemporalDFGF
        temporal_s4 = self.temporal_dfgf_s4(low_s4_fused, high_t_s4)  # [1, 768, H/32, W/32]
        
        # 3.4) 双尺度密度图引导
        spatial_s4_fused, density_s4 = self.spatial_density_s4(
            freq_feat=frame2_freq_s4,   # 频域特征（待增强）
            spatial_feat=frame2_stage4  # 空间特征（提供密度先验）
        )  # [1, 768, H/32, W/32], [1, 1, H/32, W/32]
        
        # 3.5) 高级两路融合：时频分支 + 频空分支
        frequency_s4 = self.advanced_fusion_s4(
            z_tf=temporal_s4,           # 时频特征
            z_fs=spatial_s4_fused       # 频空特征
        )  # [B, 768, H/32, W/32]

        # 3.6) 降维并上采样到 Stage3
        fused_s4 = self.reduce_s4(frequency_s4)  # [1, 384, H/32, W/32] 降维
        s4_to_s3 = F.interpolate(fused_s4, size=frame2_stage3.shape[-2:], mode='bilinear', align_corners=False)  # [1, 384, H/16, W/16]
        
        # ===== Stage3: FiLM调制 + 双分支融合 =====
        # (low_t_s3, high_t_s3 已在前面通过 backbone 共享路径生成)

        # 4.2) 用低频相似度引导调制 s4_to_s3
        low_s3_fused = self.low_film_s3(
            x=s4_to_s3,                 # 来自Stage4的特征
            low_freq_temporal=low_t_s3  # 时序低频
        )  # [1, 384, H/16, W/16]
        
        # 4.3) 高频分支：TemporalDFGF
        temporal_s3 = self.temporal_dfgf_s3(low_s3_fused, high_t_s3)  # [1, 384, H/16, W/16]
        
        # 4.4) 双尺度密度图引导
        spatial_s3_fused, density_s3 = self.spatial_density_s3(
            freq_feat=s4_to_s3,         # 频域特征（待增强）
            spatial_feat=frame2_stage3  # 空间特征（提供密度先验）
        )  # [1, 384, H/16, W/16], [1, 1, H/16, W/16]
        
        # 4.5) 高级两路融合：时频分支 + 频空分支
        frequency_s3 = self.advanced_fusion_s3(
            z_tf=temporal_s3,           # 时频特征
            z_fs=spatial_s3_fused       # 频空特征
        )  # [B, 384, H/16, W/16]

        # 4.6) 降维并上采样到输出分辨率
        fused_s3 = self.reduce_s3(frequency_s3)  # [1, 384, H/16, W/16] 降维
        final_feat = F.interpolate(fused_s3, scale_factor=2, mode='bilinear', align_corners=False)  # [1, 384, H/8, W/8]
        
        # 4.7) 密度头输出
        density = self.regression_head(final_feat).squeeze(0)
        
        # 返回最终密度图和中间密度图（用于辅助监督）
        return density, (density_s4, density_s3)




if __name__ == '__main__':
    model = VCC(
        in_chans=3,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        load_pretrained=False,
        save_wave_images=False
    )

    T = 2
    channels = 3
    height = 128
    width = 128

    # 两帧输入: [2, C, H, W]
    x = torch.rand(T, channels, height, width)

    print(f"输入形状: {x.shape}")
    y = model(x)
    print(f"输出单帧密度图形状: {y.shape}")
