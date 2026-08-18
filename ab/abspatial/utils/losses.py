import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union

from model.wave import TwoFrameDWT


class TemporalWaveLoss(nn.Module):
    """
    Temporal wavelet supervision between two frames' density maps.
    - Decompose predictions and ground truths into temporal low/high via TwoFrameDWT
    - Compute MSE/L1/SmoothL1 on low and high components respectively
    
    Args:
        w_low: weight for low-frequency loss
        w_high: weight for high-frequency loss
        resize_to: 'pred' | 'gt' | (H, W). Controls common size before DWT
        loss_type: 'mse' | 'l1' | 'smoothl1'
        wavelet: wavelet name for TwoFrameDWT
        mode: boundary mode for TwoFrameDWT
    """
    def __init__(
        self,
        w_low: float = 1.0,
        w_high: float = 1.0,
        resize_to: Union[str, tuple] = 'pred',
        loss_type: str = 'mse',
        wavelet: str = 'haar',
        mode: str = 'symmetric',
    ) -> None:
        super().__init__()
        self.w_low = w_low
        self.w_high = w_high
        self.resize_to = resize_to
        self.two_frame_dwt = TwoFrameDWT(wavelet=wavelet, mode=mode)
        self._low_as_rmse = False
        if loss_type == 'mse':
            self.crit_low = nn.MSELoss(reduction='mean')
        elif loss_type == 'rmse':
            self.crit_low = nn.MSELoss(reduction='mean')
            self._low_as_rmse = True
        elif loss_type == 'l1':
            self.crit_low = nn.L1Loss(reduction='mean')
        elif loss_type == 'smoothl1':
            self.crit_low = nn.SmoothL1Loss(reduction='mean')
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")
        # High-frequency uses robust L1 by default
        self._rmse_eps = 1e-08
        self.crit_high = nn.L1Loss(reduction='mean')

    @staticmethod
    def _ensure_4d(x: torch.Tensor) -> torch.Tensor:
        # Ensure shape [B, 1, H, W]
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(0)
        return x

    @staticmethod
    def _resize_to(x: torch.Tensor, size: tuple) -> torch.Tensor:
        if x.shape[-2:] == size:
            return x
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def forward(
        self,
        pred_prev: torch.Tensor,
        pred_curr: torch.Tensor,
        gt_prev: torch.Tensor,
        gt_curr: torch.Tensor,
    ) -> dict:
        # Shapes to [B, 1, H, W]
        pred_prev = self._ensure_4d(pred_prev)
        pred_curr = self._ensure_4d(pred_curr)
        gt_prev = self._ensure_4d(gt_prev)
        gt_curr = self._ensure_4d(gt_curr)

        # Match sizes
        if isinstance(self.resize_to, tuple):
            size = self.resize_to
            pred_prev = self._resize_to(pred_prev, size)
            pred_curr = self._resize_to(pred_curr, size)
            gt_prev = self._resize_to(gt_prev, size)
            gt_curr = self._resize_to(gt_curr, size)
        elif self.resize_to == 'pred':
            size = pred_curr.shape[-2:]
            gt_prev = self._resize_to(gt_prev, size)
            gt_curr = self._resize_to(gt_curr, size)
            pred_prev = self._resize_to(pred_prev, size)
        elif self.resize_to == 'gt':
            size = gt_curr.shape[-2:]
            pred_prev = self._resize_to(pred_prev, size)
            pred_curr = self._resize_to(pred_curr, size)
            gt_prev = self._resize_to(gt_prev, size)
        else:
            raise ValueError("resize_to must be 'pred', 'gt', or (H, W)")

        # Two-frame temporal DWT on predictions and ground truths
        low_pred, high_pred = self.two_frame_dwt(pred_prev, pred_curr)  # [B, 1, H, W]
        low_gt, high_gt = self.two_frame_dwt(gt_prev, gt_curr)

        # Align shapes (TwoFrameDWT keeps same H/W, but keep safe)
        size = low_pred.shape[-2:]
        low_gt = self._resize_to(low_gt, size)

        # Low-frequency loss (configurable by loss_type)
        loss_low = self.crit_low(low_pred, low_gt)
        if self._low_as_rmse:
            loss_low = torch.sqrt(loss_low + self._rmse_eps)
        # Align and compute high-frequency loss as well
        high_gt = self._resize_to(high_gt, size)
        loss_high = self.crit_high(high_pred, high_gt)
        loss = self.w_low * loss_low + self.w_high * loss_high

        return {
            'loss': loss,
            'loss_low': loss_low,
            'loss_high': loss_high,
            'low_pred': low_pred.detach(),
            'high_pred': high_pred.detach(),
        }


def temporal_wave_sequence_loss(
    preds: Union[List[torch.Tensor], torch.Tensor],
    gts: Union[List[torch.Tensor], torch.Tensor],
    twloss: TemporalWaveLoss,
) -> dict:
    """
    Vectorized temporal wave loss over a sequence.
    - For t=0, use (0,0) self-pair; for t>0, use (t-1, t).
    - Batch all pairs and compute once for efficiency.

    Args:
        preds: Tensor [T, 1, H, W] or list of [1, H, W]
        gts:   same as preds
        twloss: TemporalWaveLoss instance
    Returns:
        dict with 'loss', 'loss_low_sum', 'loss_high_sum' (averaged over T)
    """
    def to_T1HW(seq):
        if isinstance(seq, torch.Tensor):
            if seq.dim() == 3:  # [T, H, W]
                return seq.unsqueeze(1)
            elif seq.dim() == 4:  # [T, 1, H, W]
                return seq
            else:
                raise ValueError("preds/gts tensor must be [T,H,W] or [T,1,H,W]")
        # list of tensors
        items = []
        for x in seq:
            x4 = TemporalWaveLoss._ensure_4d(x)  # [1,1,H,W]
            assert x4.shape[0] == 1, "each list item must have batch=1"
            items.append(x4[0])  # [1,H,W]
        return torch.stack(items, dim=0)  # [T,1,H,W]

    preds_T = to_T1HW(preds)
    gts_T = to_T1HW(gts)
    assert preds_T.shape[0] == gts_T.shape[0], "preds and gts length mismatch"
    T = preds_T.shape[0]

    device = preds_T.device
    idx = torch.arange(T, device=device)
    prev_idx = torch.clamp(idx - 1, min=0)

    pred_prev = preds_T[prev_idx]  # [T,1,H,W]
    pred_curr = preds_T           # [T,1,H,W]
    gt_prev = gts_T[prev_idx]
    gt_curr = gts_T

    out = twloss(pred_prev, pred_curr, gt_prev, gt_curr)

    return {
        'loss': out['loss'],
        'loss_low_sum': out['loss_low'],
        'loss_high_sum': out['loss_high'],
    }


class DensityDistributionLoss(nn.Module):
    """
    密度图分布匹配损失（Soft引导）
    
    核心思想：
    - 不要求中间密度图的数值完全准确
    - 只要求密度分布（相对关系）相似
    - 使用 KL 散度衡量分布差异
    
    优势：
    - 比 MSE 更宽松，避免过拟合到像素级细节
    - 关注"哪里密度高/低"的相对关系
    - 适合作为辅助监督，权重可以设很小
    
    Args:
        eps: 数值稳定性常数
        reduction: 'mean' | 'sum'
    """
    def __init__(self, eps: float = 1e-8, reduction: str = 'mean') -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction
    
    def forward(self, pred_density: torch.Tensor, gt_density: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_density: 预测的密度图 [B, 1, H, W] or [B, H, W] or [1, H, W]
            gt_density: GT 密度图（会下采样到pred的分辨率）
        Returns:
            loss: L1 损失（Soft引导，比MSE更鲁棒）
        """
        # 确保形状一致
        if pred_density.dim() == 3:
            pred_density = pred_density.unsqueeze(1)  # [B, 1, H, W]
        if gt_density.dim() == 3:
            gt_density = gt_density.unsqueeze(1)
        
        # 下采样 GT 到预测的分辨率
        if gt_density.shape[-2:] != pred_density.shape[-2:]:
            gt_density = F.interpolate(
                gt_density, 
                size=pred_density.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # 使用 L1 损失（比 MSE 更鲁棒，对异常值不敏感）
        loss = F.l1_loss(pred_density, gt_density, reduction=self.reduction)
        
        return loss


class TemporalSmoothLoss(nn.Module):
    """
    密度图时序平滑损失：约束相邻帧密度图的逐像素变化
    
    核心思想：预测的密度变化应该匹配 GT 的密度变化
    pred_diff = pred_curr - pred_prev
    gt_diff = gt_curr - gt_prev
    L = |pred_diff - gt_diff|
    
    优势：
    - 比 count-delta 更细粒度（逐像素而非全局）
    - 保留空间信息（哪里变化了）
    - 直接约束密度图的时序一致性
    """
    
    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        assert reduction in ('mean', 'sum'), "reduction must be 'mean' or 'sum'"
        self.reduction = reduction
    
    @staticmethod
    def _ensure_4d(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            return x.unsqueeze(0)
        return x
    
    def forward(
        self,
        pred_prev: torch.Tensor,
        pred_curr: torch.Tensor,
        gt_prev: torch.Tensor,
        gt_curr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_prev: 预测的上一帧密度图 [B, C, H, W] or [B, H, W] or [H, W]
            pred_curr: 预测的当前帧密度图
            gt_prev: GT 上一帧密度图
            gt_curr: GT 当前帧密度图
        Returns:
            loss: 时序平滑损失
        """
        p0 = self._ensure_4d(pred_prev)
        p1 = self._ensure_4d(pred_curr)
        g0 = self._ensure_4d(gt_prev)
        g1 = self._ensure_4d(gt_curr)
        
        # 预测的时序变化（逐像素）
        pred_diff = p1 - p0  # [B, C, H, W]
        
        # GT 的时序变化（逐像素）
        gt_diff = g1 - g0  # [B, C, H, W]
        
        # L1 损失：预测变化应该匹配 GT 变化
        diff = torch.abs(pred_diff - gt_diff)
        
        if self.reduction == 'sum':
            return diff.sum()
        else:
            return diff.mean()
