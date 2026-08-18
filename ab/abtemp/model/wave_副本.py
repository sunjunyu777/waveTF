import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np

try:
    import pywt
    import pytorch_wavelets.dtcwt as dtcwt
    from pytorch_wavelets import DWT1DForward, DWT1DInverse
    PYTORCH_WAVELETS_AVAILABLE = True
except ImportError:
    print("Warning: pytorch_wavelets not available, falling back to simple implementation")
    PYTORCH_WAVELETS_AVAILABLE = False


def save_tensor_as_image(tensor, save_path_prefix, title_prefix=""):
    """
    将张量保存为图片进行可视化，保存所有时间步
    
    Args:
        tensor: [T, C, H, W] 张量
        save_path_prefix: 保存路径前缀（不包含扩展名）
        title_prefix: 图片标题前缀
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path_prefix), exist_ok=True)
    
    # 转换为numpy并处理维度
    if tensor.dim() == 4:  # [T, C, H, W]
        T, C, H, W = tensor.shape
        # 保存每个时间步的第一个通道
        for t in range(T):
            img = tensor[t, 0].detach().cpu().numpy()
            
            # 归一化到0-250范围
            img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 250
            
            # 保存图片
            plt.figure(figsize=(8, 6))
            plt.imshow(img, cmap='jet')  # 使用jet颜色映射：蓝→绿→黄→红
            plt.colorbar()
            plt.title(f"{title_prefix} - Time Step {t}")
            plt.axis('off')
            plt.savefig(f"{save_path_prefix}_t{t}.png", dpi=150, bbox_inches='tight')
            plt.close()
            
    elif tensor.dim() == 3:  # [C, H, W] 
        img = tensor[0].detach().cpu().numpy()
        # 归一化到0-250范围
        img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 250
        
        # 保存图片
        plt.figure(figsize=(8, 6))
        plt.imshow(img, cmap='jet')
        plt.colorbar()
        plt.title(title_prefix)
        plt.axis('off')
        plt.savefig(f"{save_path_prefix}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
    else:  # [H, W]
        img = tensor.detach().cpu().numpy()
        # 归一化到0-250范围
        img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 250
        
        # 保存图片
        plt.figure(figsize=(8, 6))
        plt.imshow(img, cmap='jet')
        plt.colorbar()
        plt.title(title_prefix)
        plt.axis('off')
        plt.savefig(f"{save_path_prefix}.png", dpi=150, bbox_inches='tight')
        plt.close()


class TemporalDWT(nn.Module):
    """
    时间维度的真正离散小波变换
    对每个像素的完整时序信号进行DWT分析，而不是简单的配对处理
    """

    def __init__(self, wavelet='haar', mode='symmetric', levels=1, save_wave_images=False):
        """
        初始化时序DWT
        
        Args:
            wavelet: 小波基 ('haar', 'db4', 'db8', 'coif2' 等)
            mode: 边界处理模式 ('symmetric', 'periodization', 'zero', 'constant')
            levels: 分解层数
        """
        super().__init__()
        self.wavelet = wavelet
        self.mode = mode
        self.levels = levels
        self.save_wave_images = save_wave_images
        
        # 只保留pytorch_wavelets实现
        if not PYTORCH_WAVELETS_AVAILABLE:
            raise ImportError("pytorch_wavelets is required for this implementation")
        
        self.dwt_forward = DWT1DForward(wave=wavelet, mode=mode)
        
    def dwt_temporal_pytorch_wavelets(self, x):
        """
        使用pytorch_wavelets对时序信号进行真正的DWT分析
        对每个像素的完整时序信号进行分解，只返回低频分量
        
        Args:
            x: 输入时序特征 [T, C, H, W]
            
        Returns:
            low_freq: 低频分量
        """
        T, C, H, W = x.shape
        
        # 重塑：将每个像素位置的C通道时序信号提取出来
        # [T, C, H, W] → [H*W, C, T]
        x_reshaped = x.permute(2, 3, 1, 0).contiguous().view(H*W, C, T)  # [H*W, C, T]
        
        # 使用pytorch_wavelets对每个像素的时序信号进行DWT
        # pytorch_wavelets 期望 3D 输入 (N, C, L)，格式正好匹配
        low_freq, _ = self.dwt_forward(x_reshaped)  # [H*W, C, T]
        
        # low_freq: [H*W, C, T//2] 或 [H*W, C, T_approx]
        
        # 重塑回空间格式
        T_low = low_freq.shape[2]   # 现在时间维度在第2个位置 [H*W, C, T_low]
        
        # [H*W, C, T_low] → [H, W, C, T_low] → [T_low, C, H, W]
        low_freq_spatial = low_freq.view(H, W, C, T_low).permute(3, 2, 0, 1).contiguous()
        
        # 仅在需要时保存可视化图片
        if self.save_wave_images:
            os.makedirs('waveresults', exist_ok=True)
            save_tensor_as_image(
                low_freq_spatial, 
                f'waveresults/low_freq_visualization_T{T}_C{C}_H{H}_W{W}',
                f'Low Frequency - T:{T} C:{C} H:{H} W:{W}'
            )
        
        return low_freq_spatial
    
    
    def forward(self, x):
        """
        在时间维度上进行真正的DWT分解
        每个像素的时序信号都基于整段帧间信息进行分解
        
        Args:  
            x: [T, C, H, W] 输入时序特征
        
        Returns: 
            low: 低频分量 [T_low, C, H, W]
        """
        # 使用真正的DWT实现，只返回低频
        low = self.dwt_temporal_pytorch_wavelets(x)
        
        return low


class MultiScaleHighFreq(nn.Module):
    """
    多尺度高频提取：以当前帧为中心向两边发散
    """
    
    def __init__(self, wavelet='haar', mode='symmetric', save_wave_images=False, window_len=4):
        super().__init__()
        self.wavelet = wavelet
        self.mode = mode
        self.save_wave_images = save_wave_images
        # 滑动窗口长度 t：
        # - 若 center_idx <= T-t：使用右向累进 [i:i+2], [i:i+3], ... [i:i+t]
        # - 若 center_idx >  T-t：固定使用最后一段 [T-t:T]，并在该段内用旧逻辑适配最后几帧
        self.window_len = window_len
        
        if not PYTORCH_WAVELETS_AVAILABLE:
            raise ImportError("pytorch_wavelets is required for this implementation")
        
        self.dwt_forward = DWT1DForward(wave=wavelet, mode=mode)
    
    def extract_temporal_windows(self, frames, center_idx):
        """
        以center_idx为当前帧，向左右扩展生成连续窗口
        
        例如 T=4, center_idx=1 (第2帧): [2,1], [2,3], [2,3,4]
        例如 T=4, center_idx=2 (第3帧): [3,2], [3,2,1], [3,4]
        
        Args:
            frames: [T, C, H, W] 完整帧序列
            center_idx: 当前帧索引
            
        Returns:
            windows: 多个连续时间窗口的列表 [List[Tensor[T_window, C, H, W]]]
        """
        T, C, H, W = frames.shape
        windows = []
        
        t = max(2, int(self.window_len)) if self.window_len is not None else 4
        if T < 2:
            return windows

        if center_idx <= T - t:
            # 右向累进：从 i 开始，长度 2..t
            for L in range(2, t + 1):
                start = center_idx
                end = center_idx + L  # 不含
                if end <= T:
                    indices = list(range(start, end))
                    window = torch.stack([frames[i] for i in indices], dim=0)
                    windows.append(window)
        else:
            # 固定最后一段 [T-t, T)，在该段内用旧逻辑适配最后几帧
            fixed_start = max(0, T - t)
            sub_T = T - fixed_start
            new_center = center_idx - fixed_start
            # 左扩（最少2）
            for left_len in range(2, new_center + 2):
                start = new_center - left_len + 1
                if start < 0:
                    continue
                indices = [fixed_start + i for i in range(start, new_center + 1)]
                window = torch.stack([frames[i] for i in indices], dim=0)
                windows.append(window)
            # 右扩
            for right_len in range(1, sub_T - new_center):
                indices = [fixed_start + i for i in range(new_center, new_center + right_len + 1)]
                window = torch.stack([frames[i] for i in indices], dim=0)
                windows.append(window)
        
        return windows
    
    def compute_high_freq_for_window(self, window):
        """
        对单个时间窗口计算高频
        
        Args:
            window: [T_window, C, H, W]
            
        Returns:
            high_freq: [T_high, C, H, W] 高频分量
        """
        T_window, C, H, W = window.shape
        
        # 重塑为时序信号格式：每个像素位置的C通道时序信号
        # [T_window, C, H, W] → [H*W, C, T_window]
        x_reshaped = window.permute(2, 3, 1, 0).contiguous().view(H*W, C, T_window)  # [H*W, C, T_window]
        
        # DWT分解 - 格式直接匹配pytorch_wavelets期望
        low_freq, high_freq_list = self.dwt_forward(x_reshaped)
        
        # 取第一级高频
        high_freq = high_freq_list[0]  # [H*W, C, T_high]
        
        # 重塑回空间格式
        T_high = high_freq.shape[2]  # 时间维度现在在第2个位置
        # [H*W, C, T_high] → [H, W, C, T_high] → [T_high, C, H, W]
        high_freq_spatial = high_freq.view(H, W, C, T_high).permute(3, 2, 0, 1).contiguous()

        # 若原窗口长度为奇数（对称补齐导致最后一个时间步易受边界影响），且 T_high≥2 时删除最后一个高频时间步
        if (T_window % 2) == 1 and high_freq_spatial.shape[0] >= 2:
            high_freq_spatial = high_freq_spatial[:-1]
        
        # 仅在需要时保存可视化图片
        if self.save_wave_images:
            os.makedirs('waveresults', exist_ok=True)
            save_tensor_as_image(
                high_freq_spatial,
                f'waveresults/high_freq_visualization_Tw{T_window}_C{C}_H{H}_W{W}',
                f'High Frequency - T_window:{T_window} C:{C} H:{H} W:{W}'
            )
        
        return high_freq_spatial
    
    def forward(self, frames, center_idx):
        """
        为指定的中心帧计算高频特征，并在T维度拼接
        
        Args:
            frames: [T, C, H, W] 完整帧序列
            center_idx: 中心帧索引
            
        Returns:
            concatenated_high_freq: 拼接后的高频特征 [Total_T, C, H, W]
        """
        # 提取多个时间窗口
        windows = self.extract_temporal_windows(frames, center_idx)
        
        # 对每个窗口计算高频
        high_freq_list = []
        for window in windows:
            high_freq = self.compute_high_freq_for_window(window)  # [T_window//2, C, H, W]
            high_freq_list.append(high_freq)
        
        # 在T维度拼接所有高频结果
        if len(high_freq_list) > 0:
            concatenated_high_freq = torch.cat(high_freq_list, dim=0)  # [Total_T, C, H, W]
        else:
            # 如果没有窗口，返回空张量
            _, C, H, W = frames.shape
            concatenated_high_freq = torch.empty(0, C, H, W, device=frames.device, dtype=frames.dtype)
        
        return concatenated_high_freq


class TwoFrameDWT(nn.Module):
    """
    两帧时序DWT：专门处理两帧输入的高低频分解
    直接在时间维度上对两帧进行DWT，分离出低频（趋势）和高频（变化）
    """
    
    def __init__(self, wavelet='haar', mode='symmetric', save_wave_images=False):
        """
        初始化两帧DWT
        
        Args:
            wavelet: 小波基 ('haar', 'db4', 'db8', 'coif2' 等)
            mode: 边界处理模式 ('symmetric', 'periodization', 'zero', 'constant')
            save_wave_images: 是否保存可视化图片
        """
        super().__init__()
        self.wavelet = wavelet
        self.mode = mode
        self.save_wave_images = save_wave_images
        
        if not PYTORCH_WAVELETS_AVAILABLE:
            raise ImportError("pytorch_wavelets is required for this implementation")
        
        self.dwt_forward = DWT1DForward(wave=wavelet, mode=mode)
    
    def forward(self, frame1, frame2):
        """
        对两帧进行时序DWT分解
        
        Args:
            frame1: 第一帧 [B, C, H, W]
            frame2: 第二帧 [B, C, H, W]
            
        Returns:
            low_freq: 低频分量（时序趋势） [B, C, H, W]
            high_freq: 高频分量（时序变化） [B, C, H, W]
        """
        B, C, H, W = frame1.shape
        assert frame2.shape == frame1.shape, "两帧的形状必须相同"
        
        # 将两帧堆叠成时序 [B, C, H, W, 2]
        frames_stacked = torch.stack([frame1, frame2], dim=-1)  # [B, C, H, W, 2]
        
        # 重塑为 [B*H*W, C, 2]，保持通道维度
        # 这样DWT会对每个通道独立地进行时序分解
        frames_reshaped = frames_stacked.permute(0, 2, 3, 1, 4)  # [B, H, W, C, 2]
        frames_reshaped = frames_reshaped.reshape(B * H * W, C, 2)  # [B*H*W, C, 2]
        
        # 使用DWT1DForward对时序信号进行分解
        low_freq_flat, high_freq_list = self.dwt_forward(frames_reshaped)
        high_freq_flat = high_freq_list[0]  # 取第一级高频
        
        # 重塑回空间格式 [B*H*W, C, 1] -> [B, C, H, W]
        low_freq = low_freq_flat.squeeze(-1).reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        high_freq = high_freq_flat.squeeze(-1).reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # 仅在需要时保存可视化图片
        if self.save_wave_images:
            os.makedirs('waveresults', exist_ok=True)
            # 保存第一个batch的第一个通道
            save_tensor_as_image(
                low_freq[0:1],  # [1, C, H, W]
                f'waveresults/two_frame_low_freq_B{B}_C{C}_H{H}_W{W}',
                f'Two-Frame Low Frequency - B:{B} C:{C} H:{H} W:{W}'
            )
            save_tensor_as_image(
                high_freq[0:1],  # [1, C, H, W]
                f'waveresults/two_frame_high_freq_B{B}_C{C}_H{H}_W{W}',
                f'Two-Frame High Frequency - B:{B} C:{C} H:{H} W:{W}'
            )
        
        return low_freq, high_freq


class Wave(nn.Module):
    """
    Wave类：处理光流结果并进行时序DWT变换
    低频：全局的，整个序列只计算一次
    高频：多尺度的，每帧都有自己的多时间窗口组合
    """
    
    def __init__(self, save_wave_images=False, window_len=4):
        super().__init__()
        self.save_wave_images = save_wave_images
        self.temporal_dwt = TemporalDWT(save_wave_images=save_wave_images)
        self.multi_scale_high_freq = MultiScaleHighFreq(save_wave_images=save_wave_images, window_len=window_len)
    
    def forward(self, frames):
        """
        直接对帧序列进行时序DWT处理
        
        Args:
            frames: 输入帧序列 [T, C, H, W]
            
        Returns:
            processed_results: 每个帧对应的(low_freq, multi_scale_high_freq)元组列表
        """
        T, C, H, W = frames.shape
        processed_results = []
        
        # 低频：对完整序列计算一次（全局趋势）
        low_freq = self.temporal_dwt(frames)
        
        # 对每一帧计算其高频特征
        for frame_idx in range(T):
            # 以当前帧为中心，计算高频并在T维度拼接
            concatenated_high_freq = self.multi_scale_high_freq(frames, frame_idx)
            # concatenated_high_freq 是 [Total_T, C, H, W]
            
            processed_results.append((low_freq, concatenated_high_freq))
        
        return processed_results