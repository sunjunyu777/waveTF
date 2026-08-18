import torch
import torch.nn.functional as F
import os
import numpy as np
from dataset.dataset import Crowd
from model.model import VCC
import argparse
from glob import glob
import cv2
from torch.utils.data import DataLoader
import h5py
from tqdm import tqdm

args = None

def custom_collate_fn(batch):
    """与训练一致：DataLoader返回时直接给出时序张量[T, C, H, W]等，不引入batch维。"""
    if len(batch) == 1:
        return batch[0]
    raise ValueError("Batch size must be 1 for sliding/jump window testing")


def enable_debug_vis(model):
    """启用模型的调试可视化模式"""
    count = 0
    for name, module in model.named_modules():
        if hasattr(module, 'return_debug'):
            module.return_debug = True
            print(f"已启用 {name} 的调试模式")
            count += 1
    if count == 0:
        print("[WARNING] 没有找到任何支持调试的模块！")
    else:
        print(f"[INFO] 共启用了 {count} 个模块的调试模式")


def save_debug_visualizations(imgs, debug_info, save_path, step_idx):
    """保存unique_mask和density_map等调试可视化
    Args:
        imgs: [2, C, H, W] 输入图像对
        debug_info: dict，包含unique_mask, shared_mask, density_s4, density_s3等
        save_path: 保存路径
        step_idx: 当前步骤索引
    """
    import matplotlib.pyplot as plt
    
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    
    # 转换为numpy并归一化
    img1 = imgs[0].cpu().permute(1, 2, 0).numpy()
    img2 = imgs[1].cpu().permute(1, 2, 0).numpy()
    img1 = (img1 - img1.min()) / (img1.max() - img1.min() + 1e-8)
    img2 = (img2 - img2.min()) / (img2.max() - img2.min() + 1e-8)
    
    # 提取mask并上采样到原图尺寸
    unique_mask = debug_info['unique_mask'][0, 0].cpu().numpy()  # [H, W]
    shared_mask = debug_info['shared_mask'][0, 0].cpu().numpy()
    similarity = debug_info['similarity'][0, 0].cpu().numpy()
    
    # 提取密度图
    density_s4 = debug_info['density_s4'][0, 0].cpu().numpy()  # [H/32, W/32]
    density_s3 = debug_info['density_s3'][0, 0].cpu().numpy()  # [H/16, W/16]
    
    H, W = img1.shape[:2]
    unique_mask_up = cv2.resize(unique_mask, (W, H), interpolation=cv2.INTER_LINEAR)
    shared_mask_up = cv2.resize(shared_mask, (W, H), interpolation=cv2.INTER_LINEAR)
    similarity_up = cv2.resize(similarity, (W, H), interpolation=cv2.INTER_LINEAR)
    density_s4_up = cv2.resize(density_s4, (W, H), interpolation=cv2.INTER_LINEAR)
    density_s3_up = cv2.resize(density_s3, (W, H), interpolation=cv2.INTER_LINEAR)
    
    # 创建可视化 (3x3布局)
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    
    # 第一行：输入图像和相似度
    axes[0, 0].imshow(img1)
    axes[0, 0].set_title('Frame t-1', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img2)
    axes[0, 1].set_title('Frame t', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    im_sim = axes[0, 2].imshow(similarity_up, cmap='RdYlGn', vmin=-1, vmax=1)
    axes[0, 2].set_title('Cosine Similarity', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im_sim, ax=axes[0, 2], fraction=0.046)
    
    # 第二行：共享/独特mask和叠加
    im_shared = axes[1, 0].imshow(shared_mask_up, cmap='hot', vmin=0, vmax=1)
    axes[1, 0].set_title('Shared Mask', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im_shared, ax=axes[1, 0], fraction=0.046)
    
    im_unique = axes[1, 1].imshow(unique_mask_up, cmap='viridis', vmin=0, vmax=1)
    axes[1, 1].set_title('Unique Mask', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im_unique, ax=axes[1, 1], fraction=0.046)
    
    # 叠加unique_mask到图像上
    overlay = img2.copy()
    overlay[:, :, 0] = np.clip(overlay[:, :, 0] + unique_mask_up * 0.5, 0, 1)
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title('Unique Mask Overlay', fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')
    
    # 第三行：密度图可视化
    im_d4 = axes[2, 0].imshow(density_s4_up, cmap='jet')
    axes[2, 0].set_title(f'Density S4 (max={density_s4.max():.2f})', fontsize=14, fontweight='bold')
    axes[2, 0].axis('off')
    plt.colorbar(im_d4, ax=axes[2, 0], fraction=0.046)
    
    im_d3 = axes[2, 1].imshow(density_s3_up, cmap='jet')
    axes[2, 1].set_title(f'Density S3 (max={density_s3.max():.2f})', fontsize=14, fontweight='bold')
    axes[2, 1].axis('off')
    plt.colorbar(im_d3, ax=axes[2, 1], fraction=0.046)
    
    # 密度图叠加到原图
    overlay_density = img2.copy()
    density_norm = density_s3_up / (density_s3_up.max() + 1e-8)
    overlay_density = overlay_density * 0.5 + plt.cm.jet(density_norm)[:, :, :3] * 0.5
    axes[2, 2].imshow(overlay_density)
    axes[2, 2].set_title('Density S3 Overlay', fontsize=14, fontweight='bold')
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'debug_step_{step_idx:04d}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"已保存可视化: {os.path.join(save_path, f'debug_step_{step_idx:04d}.png')}")


def denormalize_img(img):
    img = img.detach().cpu().permute(1, 2, 0).numpy()
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = img * std + mean
    return np.clip(img, 0, 1)


def to_density_np(density):
    density = density.detach().cpu()
    while density.dim() > 2:
        density = density.squeeze(0)
    return density.numpy()


def collect_vis_sample(vis_samples, img, gt_density, pred_density, gt_count, pred_count):
    if len(vis_samples) >= 4:
        return
    vis_samples.append({
        'img': denormalize_img(img),
        'gt_density': to_density_np(gt_density),
        'pred_density': to_density_np(pred_density),
        'gt_count': float(gt_count),
        'pred_count': float(pred_count),
    })


def collect_and_save_density_vis(vis_samples, img, gt_density, pred_density, gt_count, pred_count,
                                 save_path, already_saved, name='density_summary.png'):
    collect_vis_sample(vis_samples, img, gt_density, pred_density, gt_count, pred_count)
    if len(vis_samples) >= 4 and not already_saved:
        save_count_visualization(vis_samples, save_path, name=name)
        return True
    return already_saved


def save_count_visualization(vis_samples, save_path, name='density_summary.png'):
    if not vis_samples:
        print("未保存密度图谱: 没有收集到可视化样本")
        return
    import matplotlib.pyplot as plt

    os.makedirs(save_path, exist_ok=True)
    cols = 4
    rows = 3
    fig, axes = plt.subplots(rows, cols, figsize=(16, 8.2))

    for col in range(cols):
        if col >= len(vis_samples):
            for row in range(rows):
                axes[row, col].axis('off')
            continue

        sample = vis_samples[col]
        axes[0, col].imshow(sample['img'])
        axes[1, col].imshow(sample['gt_density'], cmap='jet')
        axes[2, col].imshow(sample['pred_density'], cmap='jet')

        text_kwargs = dict(
            ha='right',
            va='bottom',
            fontsize=12,
            color='white',
            bbox=dict(facecolor='black', alpha=0.65, edgecolor='none', pad=3),
        )
        axes[1, col].text(0.98, 0.02, f"gt: {sample['gt_count']:.2f}",
                          transform=axes[1, col].transAxes, **text_kwargs)
        axes[2, col].text(0.98, 0.02, f"our method: {sample['pred_count']:.2f}",
                          transform=axes[2, col].transAxes, **text_kwargs)

        for row in range(rows):
            axes[row, col].axis('off')

    plt.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005, wspace=0.015, hspace=0.005)
    out_path = os.path.join(save_path, name)
    plt.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close()
    print(f"已保存计数可视化: {out_path}")


def unpack_val_batch(batch):
    if len(batch) == 4:
        return batch
    imgs, keypoints, masks = batch
    return imgs, keypoints, masks, None


def parse_args():
    parser = argparse.ArgumentParser(description='Test with visualization support')
    parser.add_argument('--data-dir', default='',
                        help='training data directory')
    parser.add_argument('--save-dir', default='',
                        help='model directory')
    parser.add_argument('--roi-path', default='',
                        help='roi path')
    parser.add_argument('--frame-number', type=int, default=2,
                        help='the number of input frames')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--is-gray', type=bool, default=False,
                        help='whether the input image is gray')
    parser.add_argument('--visualize-masks', action='store_true',
                        help='enable unique_mask visualization')
    parser.add_argument('--no-density-vis', dest='save_density_vis', action='store_false',
                        help='disable default 3x4 density visualization')
    parser.set_defaults(save_density_vis=True)
    parser.add_argument('--vis-save-dir', default='./visualizations',
                        help='directory to save visualizations')
    parser.add_argument('--vis-max-samples', type=int, default=10,
                        help='maximum number of samples to visualize')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu··
    model = VCC(in_chans=3, load_pretrained=False)
    device = torch.device('cuda')
    model.to(device)
    model.eval()
    # 直接加载模型权重
    checkpoint = torch.load(args.save_dir, device)
    model.load_state_dict(checkpoint, strict=False)
    print("成功加载模型权重")
    
    # 启用可视化模式
    if args.save_density_vis:
        print(f"密度图谱默认保存到: {args.vis_save_dir}")
    if args.visualize_masks:
        enable_debug_vis(model)
        print(f"debug mask 可视化已启用，将保存到: {args.vis_save_dir}")
        vis_counter = 0

    if 'fdst' in args.data_dir or 'ucsd' in args.data_dir or 'dronecrowd' in args.data_dir:
        sum_res = []
        datasets = [Crowd(args.data_dir+'/'+'test'+'/'+file, is_gray=args.is_gray, method='val',
                          frame_number=args.frame_number, roi_path=args.roi_path,
                          return_density=args.save_density_vis)
                     for file in tqdm(sorted(os.listdir(os.path.join(args.data_dir, 'test')), key=int), desc="创建数据集")]
        dataloader = [DataLoader(datasets[file], batch_size=1, shuffle=False, num_workers=8, pin_memory=False,
                                 collate_fn=custom_collate_fn)
                      for file in range(len(os.listdir(os.path.join(args.data_dir, 'test'))))]
        file_list = sorted(os.listdir(os.path.join(args.data_dir, 'test')), key=int)
        for file in tqdm(range(len(file_list)), desc="处理测试文件"):
            epoch_res = []
            if args.save_density_vis:
                count_vis_samples = []
                density_vis_saved = False
                density_vis_dir = os.path.join(args.vis_save_dir, file_list[file])
            for step, batch in enumerate(tqdm(dataloader[file], desc=f"处理文件 {file_list[file]}", leave=False)):
                imgs, keypoints, masks, gt_densities = unpack_val_batch(batch)
                T, C, H, W = imgs.shape  # [T, C, H, W]
                imgs = imgs.to(device)
                if gt_densities is not None:
                    gt_densities = gt_densities.to(device)

                # 处理mask，确保形状为 [T, H, W] 且与当前序列长度一致
                if isinstance(masks, torch.Tensor) and masks.dim() > 2:
                    if masks.dim() == 5:  # [B, T, 1, H, W]
                        masks = masks.squeeze(0).squeeze(1)  # [T, H, W]
                    elif masks.dim() == 4:  # [B, T, H, W]
                        masks = masks.squeeze(0)  # [T, H, W]
                    # 其余 dim==3 保持不变
                    masks = masks.to(device)
                    if masks.dim() == 3 and masks.shape[0] != T:
                        if masks.shape[0] == 1:
                            masks = masks.repeat(T, 1, 1)
                        else:
                            masks = masks[:T]
                else:
                    masks = masks.to(device)
                    if masks.dim() == 2:
                        masks = masks.unsqueeze(0).repeat(T, 1, 1)

                with torch.no_grad():
                    outputs = []
                    debug_infos = []
                    for t in range(T):
                        prev_idx = max(0, t - 1)
                        pair = torch.stack([imgs[prev_idx], imgs[t]], dim=0)
                        out_t = model(pair)
                        
                        # 如果启用可视化且未超过最大样本数
                        if args.visualize_masks and vis_counter < args.vis_max_samples:
                            # 检查是否有调试信息（模型可能返回tuple）
                            if isinstance(out_t, tuple):
                                outputs.append(out_t[0])
                                debug_infos.append(out_t[1])
                            else:
                                outputs.append(out_t)
                                debug_infos.append(None)
                        else:
                            if isinstance(out_t, tuple):
                                outputs.append(out_t[0])
                            else:
                                outputs.append(out_t)
                    for t in range(T):
                        frame_output = outputs[t].squeeze(0)  # [H/8, W/8]
                        frame_mask = masks[t]  # [H/8, W/8]
                        
                        # 保存可视化
                        if args.visualize_masks and vis_counter < args.vis_max_samples and t < len(debug_infos) and debug_infos[t] is not None:
                            prev_idx = max(0, t - 1)
                            pair = torch.stack([imgs[prev_idx], imgs[t]], dim=0)
                            save_debug_visualizations(pair, debug_infos[t], args.vis_save_dir, vis_counter)
                            vis_counter += 1
                        
                        # 对齐输出到mask尺寸
                        Hm, Wm = frame_mask.shape[-2], frame_mask.shape[-1]
                        if frame_output.shape[-2:] != (Hm, Wm):
                            frame_output = torch.nn.functional.interpolate(
                                frame_output.unsqueeze(0).unsqueeze(0), size=(Hm, Wm),
                                mode='bilinear', align_corners=False
                            ).squeeze(0).squeeze(0)
                        
                        # 应用mask
                        frame_output_masked = frame_output * frame_mask
                        pred_count = float(torch.sum(frame_output_masked))
                        gt_count = float(keypoints[t]) if keypoints.dim() == 1 else float(keypoints[0, t])
                        epoch_res.append(gt_count - pred_count)
                        if args.save_density_vis and gt_densities is not None and len(count_vis_samples) < 4:
                            density_vis_saved = collect_and_save_density_vis(
                                count_vis_samples, imgs[t], gt_densities[t], frame_output_masked,
                                gt_count, pred_count, density_vis_dir, density_vis_saved)
            
            epoch_res = np.array(epoch_res)
            if 'fdst' in args.data_dir or 'ucsd' in args.data_dir:
                test_img_list = sorted(glob(os.path.join(args.data_dir+'/'+'test'+'/'+file_list[file], '*.jpg')),
                                      key=lambda x: int(x.split('/')[-1].split('.')[0]))
            else:
                test_img_list = sorted(glob(os.path.join(args.data_dir+'/'+'test'+'/'+file_list[file], '*.jpg')),
                                      key=lambda x: int(x.split('_')[-1].split('.')[0]))
            
            # 不再按 N - t + 1 截断，保持滑窗补齐后的完整覆盖
            
            print(f"正在处理测试文件 {file_list[file]}:")
            print(f"图像总数: {len(test_img_list)}, 预测结果数量: {len(epoch_res)}")
            
            valid_img_count = min(len(test_img_list), len(epoch_res))
            for j in tqdm(range(valid_img_count), desc="计算测试结果"):
                k = test_img_list[j]
                h5_path = k.replace('jpg', 'h5')
                h5_file = h5py.File(h5_path, mode='r')
                h5_map = np.asarray(h5_file['density'])
                count = np.sum(h5_map)
                
                img_name = os.path.basename(k)
                print(f"{img_name}: 预测误差={epoch_res[j]:.2f}, 真实计数={count:.2f}, 预测计数={count-epoch_res[j]:.2f}")
                
            for e in epoch_res:
                sum_res.append(e)
            if args.save_density_vis and not density_vis_saved:
                save_count_visualization(count_vis_samples, density_vis_dir)
        
        sum_res = np.array(sum_res)
        rmse = np.sqrt(np.mean(np.square(sum_res)))
        mae = np.mean(np.abs(sum_res))
        log_str = f'Final Test: mae {mae:.4f}, rmse {rmse:.4f}, 总样本数: {len(sum_res)}'
        print(log_str)

    elif 'venice' in args.data_dir:
        sum_res = []
        datasets = [Crowd(args.data_dir+'/'+'test'+'/'+file, is_gray=args.is_gray, method='val',
                          frame_number=args.frame_number, roi_path=args.roi_path,
                          return_density=args.save_density_vis)
                     for file in tqdm(sorted(os.listdir(os.path.join(args.data_dir, 'test')), key=int), desc="创建数据集")]
        dataloader = [DataLoader(datasets[file], 1, shuffle=False, num_workers=8, pin_memory=False)
                       for file in range(len(os.listdir(os.path.join(args.data_dir, 'test'))))]
        file_list = sorted(os.listdir(os.path.join(args.data_dir, 'test')), key=int)
        for file in tqdm(range(len(file_list)), desc="处理测试文件"):
            epoch_res = []  
            if args.save_density_vis:
                count_vis_samples = []
                density_vis_saved = False
                density_vis_dir = os.path.join(args.vis_save_dir, file_list[file])
            for step, batch in enumerate(tqdm(dataloader[file], desc=f"处理子目录 {file_list[file]}", leave=False)):
                imgs, keypoints, masks, gt_densities = unpack_val_batch(batch)
                b, f, c, h, w = imgs.shape
                assert b == 1, 'the batch size should equal to 1 in validation mode'
                
                # 去掉batch维度：[B, T, C, H, W] -> [T, C, H, W]
                imgs = imgs.squeeze(0).to(device)  # [T, C, H, W]
                if gt_densities is not None:
                    gt_densities = gt_densities.squeeze(0).to(device)
                
                # 处理masks
                if masks.dim() > 3:  
                    masks = masks.squeeze(0).to(device)  # [T, H, W]
                else:
                    masks = masks.to(device)  # [T, H, W] 或 [H, W]
                    if masks.dim() == 2:  # [H, W]
                        masks = masks.unsqueeze(0).repeat(f, 1, 1)  # [T, H, W]
                
                with torch.set_grad_enabled(False):
                    if step == 0:
                        # 第一个窗口：预测所有帧
                        out0 = model(torch.stack([imgs[0], imgs[0]], dim=0))
                        out0 = out0[0] if isinstance(out0, tuple) else out0
                        frame_output = out0.squeeze(0)
                        frame_mask = masks[0]
                        Hm, Wm = frame_mask.shape[-2], frame_mask.shape[-1]
                        if frame_output.shape[-2:] != (Hm, Wm):
                            frame_output = torch.nn.functional.interpolate(
                                frame_output.unsqueeze(0).unsqueeze(0), size=(Hm, Wm),
                                mode='bilinear', align_corners=False
                            ).squeeze(0).squeeze(0)
                        frame_output_masked = frame_output * frame_mask
                        pred_count = torch.sum(frame_output_masked).detach().cpu().numpy()
                        gt_count = keypoints[0].float().cpu().numpy() if keypoints.dim() == 1 else keypoints[0, 0].float().cpu().numpy()
                        epoch_res.append(gt_count - pred_count)
                        if args.save_density_vis and gt_densities is not None and len(count_vis_samples) < 4:
                            density_vis_saved = collect_and_save_density_vis(
                                count_vis_samples, imgs[0], gt_densities[0], frame_output_masked,
                                gt_count, pred_count, density_vis_dir, density_vis_saved)
                        
                        if f > 1:
                            out1 = model(torch.stack([imgs[0], imgs[1]], dim=0))
                            out1 = out1[0] if isinstance(out1, tuple) else out1
                            frame_output = out1.squeeze(0)
                            frame_mask = masks[1]
                            Hm, Wm = frame_mask.shape[-2], frame_mask.shape[-1]
                            if frame_output.shape[-2:] != (Hm, Wm):
                                frame_output = torch.nn.functional.interpolate(
                                    frame_output.unsqueeze(0).unsqueeze(0), size=(Hm, Wm),
                                    mode='bilinear', align_corners=False
                                ).squeeze(0).squeeze(0)
                            frame_output_masked = frame_output * frame_mask
                            pred_count = torch.sum(frame_output_masked).detach().cpu().numpy()
                            gt_count = keypoints[1].float().cpu().numpy() if keypoints.dim() == 1 else keypoints[0, 1].float().cpu().numpy()
                            epoch_res.append(gt_count - pred_count)
                            if args.save_density_vis and gt_densities is not None and len(count_vis_samples) < 4:
                                density_vis_saved = collect_and_save_density_vis(
                                    count_vis_samples, imgs[1], gt_densities[1], frame_output_masked,
                                    gt_count, pred_count, density_vis_dir, density_vis_saved)
                    else:
                        # 后续窗口：只预测最后一帧
                        out_cur = model(torch.stack([imgs[f-2], imgs[f-1]], dim=0))
                        out_cur = out_cur[0] if isinstance(out_cur, tuple) else out_cur
                        frame_output = out_cur.squeeze(0)
                        frame_mask = masks[f-1]
                        Hm, Wm = frame_mask.shape[-2], frame_mask.shape[-1]
                        if frame_output.shape[-2:] != (Hm, Wm):
                            frame_output = torch.nn.functional.interpolate(
                                frame_output.unsqueeze(0).unsqueeze(0), size=(Hm, Wm),
                                mode='bilinear', align_corners=False
                            ).squeeze(0).squeeze(0)
                        frame_output_masked = frame_output * frame_mask
                        pred_count = torch.sum(frame_output_masked).detach().cpu().numpy()
                        gt_count = keypoints[f-1].float().cpu().numpy() if keypoints.dim() == 1 else keypoints[0, f-1].float().cpu().numpy()
                        epoch_res.append(gt_count - pred_count)
                        if args.save_density_vis and gt_densities is not None and len(count_vis_samples) < 4:
                            density_vis_saved = collect_and_save_density_vis(
                                count_vis_samples, imgs[f-1], gt_densities[f-1], frame_output_masked,
                                gt_count, pred_count, density_vis_dir, density_vis_saved)
            
            
            epoch_res = np.array(epoch_res)
            test_img_list = sorted(glob(os.path.join(args.data_dir+'/'+'test'+'/'+file_list[file], '*.jpg')),
                                   key=lambda x: int(x.split('_')[-1].split('.')[0]))
            
            print(f"正在处理Venice子目录 {file_list[file]}:")
            print(f"图像总数: {len(test_img_list)}, 预测结果数量: {len(epoch_res)}")
            
        
            # 不再按 N - t + 1 截断
            
            valid_img_count = min(len(test_img_list), len(epoch_res))
            for j in tqdm(range(valid_img_count), desc="计算测试结果"):
                k = test_img_list[j]
                h5_path = k.replace('jpg', 'h5')
                h5_file = h5py.File(h5_path, mode='r')
                h5_map = np.asarray(h5_file['density'])
                
                mat_path = k.replace('jpg', 'mat')
                mask = None
                try:
                    from scipy.io import loadmat
                    mask = loadmat(mat_path)['roi']
                except Exception as e:
                    print(f"读取ROI掩码失败: {e}")
                    if args.roi_path:
                        mask = np.load(args.roi_path)
                
                if mask is not None:
                    h5_map = h5_map * mask
                
                count = np.sum(h5_map)
                
                img_name = os.path.basename(k)
                print(f"{img_name}: 预测误差={epoch_res[j]:.2f}, 真实计数={count:.2f}, 预测计数={count-epoch_res[j]:.2f}")
                
            for e in epoch_res:
                sum_res.append(e)
            if args.save_density_vis and not density_vis_saved:
                save_count_visualization(count_vis_samples, density_vis_dir)
        
        sum_res = np.array(sum_res)
        rmse = np.sqrt(np.mean(np.square(sum_res)))
        mae = np.mean(np.abs(sum_res))
        log_str = f'Final Test on Venice: mae {mae:.4f}, rmse {rmse:.4f}, 总样本数: {len(sum_res)}'
        print(log_str)

    else:
        if args.save_density_vis:
            count_vis_samples = []
            density_vis_saved = False
            density_vis_dir = args.vis_save_dir
        datasets = Crowd(os.path.join(args.data_dir, 'test'), is_gray=args.is_gray, method='val',
                         frame_number=args.frame_number, roi_path=args.roi_path,
                         return_density=args.save_density_vis)
        dataloader = torch.utils.data.DataLoader(datasets, batch_size=1, shuffle=False, num_workers=8, pin_memory=False,
                                                 collate_fn=custom_collate_fn)
        epoch_res = []
        
        for step, batch in enumerate(tqdm(dataloader, desc="处理测试数据")):
            imgs, keypoints, masks, gt_densities = unpack_val_batch(batch)
            T, C, H, W = imgs.shape
            imgs = imgs.to(device)
            if gt_densities is not None:
                gt_densities = gt_densities.to(device)
            
            # 处理mask，确保形状为 [T, H, W] 且与当前序列长度一致
            if isinstance(masks, torch.Tensor) and masks.dim() > 2:
                if masks.dim() == 5:  # [B, T, 1, H, W]
                    masks = masks.squeeze(0).squeeze(1)  # [T, H, W]
                elif masks.dim() == 4:  # [B, T, H, W]
                    masks = masks.squeeze(0)  # [T, H, W]
                # 其余 dim==3 保持不变
                masks = masks.to(device)
                if masks.dim() == 3 and masks.shape[0] != T:
                    if masks.shape[0] == 1:
                        masks = masks.repeat(T, 1, 1)
                    else:
                        masks = masks[:T]
            else:
                masks = masks.to(device)
                if masks.dim() == 2:
                    masks = masks.unsqueeze(0).repeat(T, 1, 1)
            # 若时间长度与当前序列不一致，进行对齐
            if masks.dim() == 3 and masks.shape[0] != T:
                if masks.shape[0] == 1:
                    masks = masks.repeat(T, 1, 1)
                else:
                    masks = masks[:T]
                
            with torch.no_grad():
                if step == 0:
                    # 调试：检查return_debug状态
                    if args.visualize_masks and vis_counter < args.vis_max_samples:
                        for name, module in model.named_modules():
                            if hasattr(module, 'return_debug'):
                                print(f"[DEBUG] {name}.return_debug = {module.return_debug}")
                    
                    # 第一个窗口：预测所有帧
                    pair0 = torch.stack([imgs[0], imgs[0]], dim=0)
                    out0 = model(pair0)
                    
                    # 可视化支持
                    if args.visualize_masks and vis_counter < args.vis_max_samples:
                        print(f"[DEBUG] out0 类型: {type(out0)}")
                        if isinstance(out0, tuple):
                            print(f"[DEBUG] out0 长度: {len(out0)}, 第二个元素类型: {type(out0[1])}")
                            if len(out0) == 2 and isinstance(out0[1], tuple):
                                print(f"[DEBUG] out0[1] 是元组，长度: {len(out0[1])}")
                        if isinstance(out0, tuple) and len(out0) == 2 and isinstance(out0[1], dict):
                            save_debug_visualizations(pair0, out0[1], args.vis_save_dir, vis_counter)
                            vis_counter += 1
                            print(f"[DEBUG] 已保存第 {vis_counter} 个可视化")
                    
                    out0 = out0[0] if isinstance(out0, tuple) else out0
                    frame_output = out0.squeeze(0)
                    frame_mask = masks[0]
                    Hm, Wm = frame_mask.shape[-2], frame_mask.shape[-1]
                    if frame_output.shape[-2:] != (Hm, Wm):
                        frame_output = torch.nn.functional.interpolate(
                            frame_output.unsqueeze(0).unsqueeze(0), size=(Hm, Wm),
                            mode='bilinear', align_corners=False
                        ).squeeze(0).squeeze(0)
                    frame_output_masked = frame_output * frame_mask
                    pred_count = torch.sum(frame_output_masked).detach().cpu().numpy()
                    gt_count = keypoints[0].float().cpu().numpy() if keypoints.dim() == 1 else keypoints[0, 0].float().cpu().numpy()
                    epoch_res.append(gt_count - pred_count)
                    if args.save_density_vis and gt_densities is not None and len(count_vis_samples) < 4:
                        density_vis_saved = collect_and_save_density_vis(
                            count_vis_samples, imgs[0], gt_densities[0], frame_output_masked,
                            gt_count, pred_count, density_vis_dir, density_vis_saved)
                    
                    if T > 1:
                        pair1 = torch.stack([imgs[0], imgs[1]], dim=0)
                        out1 = model(pair1)
                        
                        # 可视化支持
                        if args.visualize_masks and vis_counter < args.vis_max_samples:
                            if isinstance(out1, tuple) and len(out1) == 2 and isinstance(out1[1], dict):
                                save_debug_visualizations(pair1, out1[1], args.vis_save_dir, vis_counter)
                                vis_counter += 1
                        
                        out1 = out1[0] if isinstance(out1, tuple) else out1
                        frame_output = out1.squeeze(0)
                        frame_mask = masks[1]
                        Hm, Wm = frame_mask.shape[-2], frame_mask.shape[-1]
                        if frame_output.shape[-2:] != (Hm, Wm):
                            frame_output = torch.nn.functional.interpolate(
                                frame_output.unsqueeze(0).unsqueeze(0), size=(Hm, Wm),
                                mode='bilinear', align_corners=False
                            ).squeeze(0).squeeze(0)
                        frame_output_masked = frame_output * frame_mask
                        pred_count = torch.sum(frame_output_masked).detach().cpu().numpy()
                        gt_count = keypoints[1].float().cpu().numpy() if keypoints.dim() == 1 else keypoints[0, 1].float().cpu().numpy()
                        epoch_res.append(gt_count - pred_count)
                        if args.save_density_vis and gt_densities is not None and len(count_vis_samples) < 4:
                            density_vis_saved = collect_and_save_density_vis(
                                count_vis_samples, imgs[1], gt_densities[1], frame_output_masked,
                                gt_count, pred_count, density_vis_dir, density_vis_saved)
                else:
                    # 后续窗口：只预测最后一帧
                    pair_cur = torch.stack([imgs[T-2], imgs[T-1]], dim=0)
                    out_cur = model(pair_cur)
                    
                    # 可视化支持
                    if args.visualize_masks and vis_counter < args.vis_max_samples:
                        if isinstance(out_cur, tuple) and len(out_cur) == 2 and isinstance(out_cur[1], dict):
                            save_debug_visualizations(pair_cur, out_cur[1], args.vis_save_dir, vis_counter)
                            vis_counter += 1
                    
                    out_cur = out_cur[0] if isinstance(out_cur, tuple) else out_cur
                    frame_output = out_cur.squeeze(0)
                    frame_mask = masks[T-1]
                    Hm, Wm = frame_mask.shape[-2], frame_mask.shape[-1]
                    if frame_output.shape[-2:] != (Hm, Wm):
                        frame_output = torch.nn.functional.interpolate(
                            frame_output.unsqueeze(0).unsqueeze(0), size=(Hm, Wm),
                            mode='bilinear', align_corners=False
                        ).squeeze(0).squeeze(0)
                    frame_output_masked = frame_output * frame_mask
                    pred_count = torch.sum(frame_output_masked).detach().cpu().numpy()
                    gt_count = keypoints[T-1].float().cpu().numpy() if keypoints.dim() == 1 else keypoints[0, T-1].float().cpu().numpy()
                    epoch_res.append(gt_count - pred_count)
                    if args.save_density_vis and gt_densities is not None and len(count_vis_samples) < 4:
                        density_vis_saved = collect_and_save_density_vis(
                            count_vis_samples, imgs[T-1], gt_densities[T-1], frame_output_masked,
                            gt_count, pred_count, density_vis_dir, density_vis_saved)
        
        
        epoch_res = np.array(epoch_res)
        
        test_img_list = sorted(glob(os.path.join(os.path.join(args.data_dir, 'test'), '*.jpg')),
                               key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        print(f"测试图像总数: {len(test_img_list)}, 预测结果数量: {len(epoch_res)}")
        
        # 不再按 N - t + 1 截断
        
        valid_img_count = min(len(test_img_list), len(epoch_res))
        for j in tqdm(range(valid_img_count), desc="计算测试结果"):
            k = test_img_list[j]
            h5_path = k.replace('jpg', 'h5')
            h5_file = h5py.File(h5_path, mode='r')
            h5_map = np.asarray(h5_file['density'])
            if args.roi_path:
                mask = np.load(args.roi_path)
                h5_map = h5_map * mask
            count = np.sum(h5_map)
            
            img_name = os.path.basename(k)
            print(f"{img_name}: 预测误差={epoch_res[j]:.2f}, 真实计数={count:.2f}, 预测计数={count-epoch_res[j]:.2f}")
        
        valid_errors = epoch_res[:valid_img_count]
        rmse = np.sqrt(np.mean(np.square(valid_errors)))
        mae = np.mean(np.abs(valid_errors))
        log_str = f'Final Test: mae {mae:.4f}, rmse {rmse:.4f}, 有效样本数: {valid_img_count}'
        print(log_str)
        if args.save_density_vis and not density_vis_saved:
            save_count_visualization(count_vis_samples, density_vis_dir)
