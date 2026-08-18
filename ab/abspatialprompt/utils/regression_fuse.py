from utils.trainer import Trainer
from utils.losses import TemporalSmoothLoss, DensityDistributionLoss
from utils.helper import Save_Handle, AverageMeter
import os
import sys
import time
import torch
from torch import optim
from torch.utils.data import DataLoader
import logging
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model.model import VCC
from dataset.dataset import Crowd
from glob import glob
import cv2
import random
from tqdm import tqdm


class RegTrainer(Trainer):
    def setup(self):
        """initial the datasets, model, loss and optimizer"""
        args = self.args
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            # for code conciseness, we release the single gpu version
            assert self.device_count == 1
            logging.info('using {} gpus'.format(self.device_count))
        else:
            raise Exception("gpu is not available")

        if 'fdst' in args.data_dir or 'ucsd' in args.data_dir or 'venice' in args.data_dir or 'dronecrowd' in args.data_dir:
            self.datasets = {x: [Crowd(args.data_dir+'/'+x+'/'+file, args.is_gray, x, args.frame_number,
                                       args.crop_height, args.crop_width, args.roi_path)
                                 for file in sorted(os.listdir(os.path.join(args.data_dir, x)), key=int)]
                             for x in ['train', 'val']}
            self.dataloaders = {x: [DataLoader(self.datasets[x][file],
                                               batch_size=(args.batch_size
                                               if x == 'train' else 1),
                                               shuffle=False,
                                               num_workers=args.num_workers * self.device_count,
                                               pin_memory=(True if x == 'train' else False))
                                    for file in range(len(os.listdir(os.path.join(args.data_dir, x))))]
                                for x in ['train', 'val']}
        else:
            self.datasets = {x: Crowd(os.path.join(args.data_dir, x), args.is_gray, x, args.frame_number, args.crop_height,
                                      args.crop_width, args.roi_path) for x in ['train', 'val']}
            self.dataloaders = {x: DataLoader(self.datasets[x],
                                              batch_size=(args.batch_size
                                              if x == 'train' else 1),
                                              shuffle=False,
                                              num_workers=args.num_workers*self.device_count,
                                              pin_memory=(True if x == 'train' else False))
                                for x in ['train', 'val']}
        # 初始化模型，传入 save_wave_images 参数
        save_wave = getattr(args, 'save_wave_images', True)
        self.model = VCC(in_chans=3, load_pretrained=True, save_wave_images=save_wave)
        self.model.to(self.device)
        # Temporal smooth loss for pixel-wise temporal consistency
        self.tsloss = TemporalSmoothLoss().to(self.device)
        # Density distribution loss for auxiliary supervision
        self.aux_loss = DensityDistributionLoss().to(self.device)
        # Weights
        self.wf = 1.0  # frame loss
        self.wts = 1.0 # temporal smooth loss weight
        self.waux = 1.0 # auxiliary density loss weight (Soft引导)
        # Optimizer: 统一 LR，只过滤冻结参数
        all_params = [p for p in self.model.parameters() if p.requires_grad]

        n_total = sum(p.numel() for p in all_params) / 1e6
        logging.info(f'[Optim] trainable params: {n_total:.2f}M (lr={args.lr:.2e})')

        self.optimizer = optim.AdamW(
            all_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        self.start_epoch = 0
        if args.resume:
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suf == 'pth':
                self.model.load_state_dict(torch.load(args.resume, self.device))

        self.save_list = Save_Handle(max_num=args.max_model_num)
        self.best_mae = np.inf
        self.best_mse = np.inf
        self.best_count = 0
        self.criterion = torch.nn.MSELoss(reduction='sum').to(self.device)
        self.save_all = args.save_all
        self.num = -1

    def train(self):
        """training process"""
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
            self.epoch = epoch
            self.train_eopch()
            if epoch % 50 == 0:
                self.num += 1
                self.best_mae = np.inf
                self.best_mse = np.inf
            elif epoch % args.val_epoch == 0:
                self.val_epoch()

    def train_eopch(self):
        args = self.args
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_start = time.time()
        self.model.train()  

        if 'fdst' in args.data_dir or 'ucsd' in args.data_dir or 'venice' in args.data_dir or 'dronecrowd' in args.data_dir:
            file_list = list(range(len(os.listdir(os.path.join(args.data_dir, 'train')))))
            random.shuffle(file_list)
            total_steps = sum([len(self.dataloaders['train'][file]) for file in file_list])
            pbar = tqdm(total=total_steps, desc=f'Epoch {self.epoch} Train')
            
            for file in file_list:
                # 跨 step 状态：仅在该视频内部有效
                prev_img = None
                prev_pred = None
                prev_gt = None
                prev_mask = None
                for step, (imgs, targets, keypoints, mask) in enumerate(self.dataloaders['train'][file]):
                    b0, f0, c0, h0, w0 = imgs.shape
                    assert b0 == 1
                    
                    # 去掉batch维度：[B, T, C, H, W] -> [T, C, H, W]
                    imgs = imgs.squeeze(0).to(self.device)  # [T, C, H, W]
                    targets = targets.squeeze(0).to(self.device)  # [T, 1, H/8, W/8]
                    
                    # 处理mask，确保形状为 [T, H, W] 且与当前序列长度一致
                    if isinstance(mask, torch.Tensor) and mask.dim() > 2:
                        if mask.dim() == 5:  # [B, T, 1, H, W]
                            mask = mask.squeeze(0).squeeze(1)  # [T, H, W]
                        elif mask.dim() == 4:  # [B, T, H, W]
                            mask = mask.squeeze(0)  # [T, H, W]
                        # 其余 dim==3 保持不变
                        mask = mask.to(self.device)
                        if mask.dim() == 3 and mask.shape[0] != f0:
                            if mask.shape[0] == 1:
                                mask = mask.repeat(f0, 1, 1)
                            else:
                                mask = mask[:f0]
                    else:
                        mask = mask.to(self.device)
                        if mask.dim() == 2:
                            mask = mask.unsqueeze(0).repeat(f0, 1, 1)

                    with torch.set_grad_enabled(True):
                        total_loss = 0
                        wf = self.wf
                        wts = self.wts
                        frame_errors = []
                        counted_frames = 0

                        def align_to_target(out_hw, tgt_hw, msk_hw):
                            base_h, base_w = tgt_hw.shape[-2], tgt_hw.shape[-1]
                            out_aligned = out_hw
                            msk_aligned = msk_hw
                            if out_aligned.shape[-2:] != (base_h, base_w):
                                out_aligned = torch.nn.functional.interpolate(
                                    out_aligned.unsqueeze(0).unsqueeze(0), size=(base_h, base_w),
                                    mode='bilinear', align_corners=False
                                ).squeeze(0).squeeze(0)
                            if msk_aligned.shape[-2:] != (base_h, base_w):
                                msk_aligned = torch.nn.functional.interpolate(
                                    msk_aligned.unsqueeze(0).unsqueeze(0), size=(base_h, base_w),
                                    mode='nearest'
                                ).squeeze(0).squeeze(0)
                            return out_aligned, msk_aligned

                        if prev_img is None:
                            # 第一个 step：自配对f0只用于计算损失，不计入MAE统计
                            model_out0 = self.model(torch.stack([imgs[0], imgs[0]], dim=0))
                            if isinstance(model_out0, tuple):
                                out0, (density_s4_0, density_s3_0) = model_out0
                            else:
                                out0 = model_out0
                                density_s4_0, density_s3_0 = None, None
                            
                            o0, m0 = align_to_target(out0.squeeze(0), targets[0].squeeze(0), mask[0])
                            t0 = targets[0].squeeze(0)
                            o0m = o0 * m0
                            t0m = t0 * m0
                            loss0 = self.criterion(o0m, t0m)
                            total_loss += wf * loss0
                            
                            # 辅助损失：中间密度图
                            if density_s4_0 is not None:
                                aux_loss0 = self.waux * (self.aux_loss(density_s4_0, targets[0]) + 
                                                         self.aux_loss(density_s3_0, targets[0]))
                                total_loss += aux_loss0
                                # if step == 0 and file == file_list[0]:  # 只在第一个file的第一个step打印
                                #     print(f"[DEBUG] Aux loss active - S4: {density_s4_0.shape}, S3: {density_s3_0.shape}, aux_loss0: {aux_loss0.item():.4f}")
                            
                            counted_frames += 1
                            # 自配对帧不计入MAE，与验证口径一致

                            # f1 用 [f0,f1]
                            model_out1 = self.model(torch.stack([imgs[0], imgs[1]], dim=0))
                            if isinstance(model_out1, tuple):
                                out1, (density_s4_1, density_s3_1) = model_out1
                            else:
                                out1 = model_out1
                                density_s4_1, density_s3_1 = None, None
                            
                            o1, m1 = align_to_target(out1.squeeze(0), targets[1].squeeze(0), mask[1])
                            t1 = targets[1].squeeze(0)
                            o1m = o1 * m1
                            t1m = t1 * m1
                            loss1 = self.criterion(o1m, t1m)
                            total_loss += wf * loss1
                            
                            # 辅助损失
                            if density_s4_1 is not None:
                                aux_loss1 = self.waux * (self.aux_loss(density_s4_1, targets[1]) + 
                                                         self.aux_loss(density_s3_1, targets[1]))
                                total_loss += aux_loss1
                                # if step == 0 and file == file_list[0]:  # 只在第一个file的第一个step打印
                                #     print(f"[DEBUG] Aux loss active - S4: {density_s4_1.shape}, S3: {density_s3_1.shape}, aux_loss1: {aux_loss1.item():.4f}")
                            
                            pred_count1 = torch.sum(o1m).detach().cpu().numpy()
                            gt_count1 = keypoints[1].float().cpu().numpy() if keypoints.dim() == 1 else keypoints[0, 1].float().cpu().numpy()
                            frame_errors.append(pred_count1 - gt_count1)
                            counted_frames += 1

                            # 更新跨 step 状态
                            prev_img = imgs[1].detach()
                            prev_pred = out1.detach()
                            prev_gt = targets[1].detach()
                            prev_mask = mask[1].detach()
                        else:
                            # 后续 step：只计算新帧（第二帧）
                            model_out_cur = self.model(torch.stack([prev_img, imgs[1]], dim=0))
                            if isinstance(model_out_cur, tuple):
                                out_cur, (density_s4_cur, density_s3_cur) = model_out_cur
                            else:
                                out_cur = model_out_cur
                                density_s4_cur, density_s3_cur = None, None
                            
                            oc, mc = align_to_target(out_cur.squeeze(0), targets[1].squeeze(0), mask[1])
                            tc = targets[1].squeeze(0)
                            ocm = oc * mc
                            tcm = tc * mc
                            loss_c = self.criterion(ocm, tcm)
                            total_loss += wf * loss_c
                            
                            # 辅助损失
                            if density_s4_cur is not None:
                                aux_loss_c = self.waux * (self.aux_loss(density_s4_cur, targets[1]) + 
                                                          self.aux_loss(density_s3_cur, targets[1]))
                                total_loss += aux_loss_c
                                # if step == 1 and file == file_list[0]:  # 第一个file的第二个step打印
                                #     print(f"[DEBUG] Aux loss active (subsequent) - S4: {density_s4_cur.shape}, S3: {density_s3_cur.shape}, aux_loss: {aux_loss_c.item():.4f}")
                            
                            pred_countc = torch.sum(ocm).detach().cpu().numpy()
                            gt_countc = keypoints[1].float().cpu().numpy() if keypoints.dim() == 1 else keypoints[0, 1].float().cpu().numpy()
                            frame_errors.append(pred_countc - gt_countc)
                            counted_frames += 1

                            # 计数增量一致性损失：对齐上一帧预测/GT与当前帧
                            pp, pm = align_to_target(prev_pred.squeeze(0), prev_gt.squeeze(0), prev_mask)
                            ppm = pp * pm
                            pgm = prev_gt.squeeze(0) * pm
                            ts_loss = self.tsloss(ppm, ocm, pgm, tcm)
                            total_loss = total_loss + wts * ts_loss

                            # 更新跨 step 状态
                            prev_img = imgs[1].detach()
                            prev_pred = out_cur.detach()
                            prev_gt = targets[1].detach()
                            prev_mask = mask[1].detach()

                        avg_loss = total_loss / max(1, counted_frames)
                        self.optimizer.zero_grad()
                        avg_loss.backward()
                        self.optimizer.step()

                        # 统计MAE/MSE: 每个frame_error对应一次更新
                        for frame_error in frame_errors:
                            epoch_mse.update(frame_error * frame_error, 1)
                            epoch_mae.update(abs(frame_error), 1)
                        # 统计loss: 只更新一次，使用平均loss
                        epoch_loss.update(avg_loss.item(), counted_frames)
                        
                        avg_mae = np.mean([abs(e) for e in frame_errors])
                        pbar.update(1)
                        pbar.set_postfix({'Loss': f'{avg_loss.item():.4f}', 'MAE': f'{avg_mae:.2f}'})
            
            pbar.close()

            logging.info('Epoch {} Train, Loss: {:.2f}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                         .format(self.epoch, epoch_loss.get_avg(), np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg(),
                                 time.time() - epoch_start))
            model_state_dic = self.model.state_dict()
            save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
            torch.save({
                'epoch': self.epoch,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'model_state_dict': model_state_dic,
            }, save_path)
            self.save_list.append(save_path)  # control the number of saved models
        else:
            dataloader = self.dataloaders['train']
            pbar = tqdm(dataloader, desc=f'Epoch {self.epoch} Train')
            prev_img = None
            prev_pred = None
            prev_gt = None
            prev_mask = None
            for step, (imgs, targets, keypoints, mask) in enumerate(pbar):
                b0, f0, c0, h0, w0 = imgs.shape
                assert b0 == 1
                
                # 去掉batch维度：[B, T, C, H, W] -> [T, C, H, W]
                imgs = imgs.squeeze(0).to(self.device)  # [T, C, H, W]
                targets = targets.squeeze(0).to(self.device)  # [T, 1, H/8, W/8]
                mask = mask.to(self.device)  # [T, H/8, W/8] 或 [H/8, W/8]
                
                # 处理mask -> [T, H, W]
                if isinstance(mask, torch.Tensor) and mask.dim() > 2:
                    if mask.dim() == 5:
                        mask = mask.squeeze(0).squeeze(1)
                    elif mask.dim() == 4:
                        mask = mask.squeeze(0)
                    if mask.dim() == 3 and mask.shape[0] != f0:
                        if mask.shape[0] == 1:
                            mask = mask.repeat(f0, 1, 1)
                        else:
                            mask = mask[:f0]
                elif mask.dim() == 2:
                    mask = mask.unsqueeze(0).repeat(f0, 1, 1)

                with torch.set_grad_enabled(True):
                    total_loss = 0
                    wf = self.wf
                    wts = self.wts
                    frame_errors = []
                    counted_frames = 0

                    def align_to_target(out_hw, tgt_hw, msk_hw):
                        base_h, base_w = tgt_hw.shape[-2], tgt_hw.shape[-1]
                        out_aligned = out_hw
                        msk_aligned = msk_hw
                        if out_aligned.shape[-2:] != (base_h, base_w):
                            out_aligned = torch.nn.functional.interpolate(
                                out_aligned.unsqueeze(0).unsqueeze(0), size=(base_h, base_w),
                                mode='bilinear', align_corners=False
                            ).squeeze(0).squeeze(0)
                        if msk_aligned.shape[-2:] != (base_h, base_w):
                            msk_aligned = torch.nn.functional.interpolate(
                                msk_aligned.unsqueeze(0).unsqueeze(0), size=(base_h, base_w),
                                mode='nearest'
                            ).squeeze(0).squeeze(0)
                        return out_aligned, msk_aligned

                    if prev_img is None:
                        # 第一个 step：自配对f0只用于计算损失，不计入MAE统计
                        model_out0 = self.model(torch.stack([imgs[0], imgs[0]], dim=0))
                        if isinstance(model_out0, tuple):
                            out0, (density_s4_0, density_s3_0) = model_out0
                        else:
                            out0 = model_out0
                            density_s4_0, density_s3_0 = None, None
                        
                        o0, m0 = align_to_target(out0.squeeze(0), targets[0].squeeze(0), mask[0])
                        t0 = targets[0].squeeze(0)
                        o0m = o0 * m0
                        t0m = t0 * m0
                        loss0 = self.criterion(o0m, t0m)
                        total_loss += wf * loss0
                        
                        # 辅助损失
                        if density_s4_0 is not None:
                            aux_loss0 = self.waux * (self.aux_loss(density_s4_0, targets[0]) + 
                                                     self.aux_loss(density_s3_0, targets[0]))
                            total_loss += aux_loss0
                            # if step == 0:  # 第一个step打印
                            #     print(f"[DEBUG] Aux loss active - S4: {density_s4_0.shape}, S3: {density_s3_0.shape}, aux_loss0: {aux_loss0.item():.4f}")
                        
                        counted_frames += 1
                        # 自配对帧不计入MAE，与验证口径一致

                        model_out1 = self.model(torch.stack([imgs[0], imgs[1]], dim=0))
                        if isinstance(model_out1, tuple):
                            out1, (density_s4_1, density_s3_1) = model_out1
                        else:
                            out1 = model_out1
                            density_s4_1, density_s3_1 = None, None
                        
                        o1, m1 = align_to_target(out1.squeeze(0), targets[1].squeeze(0), mask[1])
                        t1 = targets[1].squeeze(0)
                        o1m = o1 * m1
                        t1m = t1 * m1
                        loss1 = self.criterion(o1m, t1m)
                        total_loss += wf * loss1
                        
                        # 辅助损失
                        if density_s4_1 is not None:
                            aux_loss1 = self.waux * (self.aux_loss(density_s4_1, targets[1]) + 
                                                     self.aux_loss(density_s3_1, targets[1]))
                            total_loss += aux_loss1
                            # if step == 0:  # 第一个step打印
                            #     print(f"[DEBUG] Aux loss active - S4: {density_s4_1.shape}, S3: {density_s3_1.shape}, aux_loss1: {aux_loss1.item():.4f}")
                        
                        pred_count1 = torch.sum(o1m).detach().cpu().numpy()
                        if keypoints.dim() > 1:
                            gt_count1 = keypoints[0, 1].float().cpu().numpy()
                        else:
                            gt_count1 = keypoints[1].float().cpu().numpy()
                        frame_errors.append(pred_count1 - gt_count1)
                        counted_frames += 1

                        prev_img = imgs[1].detach()
                        prev_pred = out1.detach()
                        prev_gt = targets[1].detach()
                        prev_mask = mask[1].detach()
                    else:
                        model_out_cur = self.model(torch.stack([prev_img, imgs[1]], dim=0))
                        if isinstance(model_out_cur, tuple):
                            out_cur, (density_s4_cur, density_s3_cur) = model_out_cur
                        else:
                            out_cur = model_out_cur
                            density_s4_cur, density_s3_cur = None, None
                        
                        oc, mc = align_to_target(out_cur.squeeze(0), targets[1].squeeze(0), mask[1])
                        tc = targets[1].squeeze(0)
                        ocm = oc * mc
                        tcm = tc * mc
                        loss_c = self.criterion(ocm, tcm)
                        total_loss += wf * loss_c
                        
                        # 辅助损失
                        if density_s4_cur is not None:
                            aux_loss_c = self.waux * (self.aux_loss(density_s4_cur, targets[1]) + 
                                                      self.aux_loss(density_s3_cur, targets[1]))
                            total_loss += aux_loss_c
                            # if step == 1:  # 第二个step打印
                            #     print(f"[DEBUG] Aux loss active (subsequent) - S4: {density_s4_cur.shape}, S3: {density_s3_cur.shape}, aux_loss: {aux_loss_c.item():.4f}")
                        
                        pred_countc = torch.sum(ocm).detach().cpu().numpy()
                        if keypoints.dim() > 1:
                            gt_countc = keypoints[0, 1].float().cpu().numpy()
                        else:
                            gt_countc = keypoints[1].float().cpu().numpy()
                        frame_errors.append(pred_countc - gt_countc)
                        counted_frames += 1

                        # Temporal smooth loss (pixel-wise consistency)
                        pp, pm = align_to_target(prev_pred.squeeze(0), prev_gt.squeeze(0), prev_mask)
                        ppm = pp * pm
                        pgm = prev_gt.squeeze(0) * pm
                        ts_loss = self.tsloss(ppm, ocm, pgm, tcm)
                        total_loss = total_loss + wts * ts_loss

                        prev_img = imgs[1].detach()
                        prev_pred = out_cur.detach()
                        prev_gt = targets[1].detach()
                        prev_mask = mask[1].detach()

                    avg_loss = total_loss / max(1, counted_frames)
                    
                    self.optimizer.zero_grad()
                    avg_loss.backward()
                    self.optimizer.step()

                    # 统计MAE/MSE: 每个frame_error对应一次更新
                    for frame_error in frame_errors:
                        epoch_mse.update(frame_error * frame_error, 1)
                        epoch_mae.update(abs(frame_error), 1)
                    # 统计loss: 只更新一次，使用平均loss
                    epoch_loss.update(avg_loss.item(), counted_frames)
                    
                    avg_mae = np.mean([abs(e) for e in frame_errors])
                    pbar.set_postfix({'Loss': f'{avg_loss.item():.4f}', 'MAE': f'{avg_mae:.2f}'})
            
            pbar.close()

            logging.info('Epoch {} Train, Loss: {:.2f}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                         .format(self.epoch, epoch_loss.get_avg(), np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg(),
                                 time.time()-epoch_start))
            model_state_dic = self.model.state_dict()
            save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
            torch.save({
                'epoch': self.epoch,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'model_state_dict': model_state_dic,
            }, save_path)
            self.save_list.append(save_path)  # control the number of saved models

    def val_epoch(self):
        args = self.args
        epoch_start = time.time()
        self.model.eval()  
        if 'fdst' in args.data_dir or 'ucsd' in args.data_dir or 'dronecrowd' in args.data_dir:
            sum_res = []  # 修正：将sum_res定义在这里
            file_list = sorted(os.listdir(os.path.join(args.data_dir, 'val')), key=int)
            total_steps = sum([len(self.dataloaders['val'][file]) for file in range(len(file_list))])
            pbar = tqdm(total=total_steps, desc=f'Epoch {self.epoch} Val')
            
            for file in range(len(file_list)):
                epoch_res = []
                
                if 'ucsd' in args.data_dir or 'fdst' in args.data_dir:
                    for step, (imgs, keypoints, masks) in enumerate(self.dataloaders['val'][file]):
                        # 统一为 4D [T,C,H,W]
                        if imgs.dim() == 5:
                            assert imgs.shape[0] == 1
                            imgs = imgs.squeeze(0)
                        f, c, h, w = imgs.shape
                        imgs = imgs.to(self.device)

                        # 统一 masks 为 [T,H,W]
                        if masks.dim() == 5:
                            masks = masks.squeeze(0).squeeze(1)
                        elif masks.dim() == 4 and masks.shape[1] == 1:
                            masks = masks.squeeze(1)
                        elif masks.dim() == 2:
                            masks = masks.unsqueeze(0).repeat(f, 1, 1)
                        masks = masks.to(self.device)

                        with torch.set_grad_enabled(False):
                            if step == 0:
                                # 第一个窗口：预测所有帧
                                # 帧0自配
                                model_out0 = self.model(torch.stack([imgs[0], imgs[0]], dim=0))
                                out0 = model_out0[0] if isinstance(model_out0, tuple) else model_out0
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
                                
                                # 帧1用[f0, f1]
                                if f > 1:
                                    model_out1 = self.model(torch.stack([imgs[0], imgs[1]], dim=0))
                                    out1 = model_out1[0] if isinstance(model_out1, tuple) else model_out1
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
                            else:
                                # 后续窗口：只预测最后一帧
                                model_out_cur = self.model(torch.stack([imgs[f-2], imgs[f-1]], dim=0))
                                out_cur = model_out_cur[0] if isinstance(model_out_cur, tuple) else model_out_cur
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

                            avg_mae = np.mean([abs(r) for r in epoch_res[-min(f if step == 0 else 1, len(epoch_res)):]])
                            pbar.update(1)
                            pbar.set_postfix({'MAE': f'{avg_mae:.2f}'})
                elif 'dronecrowd' in args.data_dir:
                    for step, (imgs, keypoints, masks) in enumerate(self.dataloaders['val'][file]):
                        if imgs.dim() == 5:
                            assert imgs.shape[0] == 1
                            imgs = imgs.squeeze(0)
                        f, c, h, w = imgs.shape
                        imgs = imgs.to(self.device)

                        if masks.dim() == 5:
                            masks = masks.squeeze(0).squeeze(1)
                        elif masks.dim() == 4 and masks.shape[1] == 1:
                            masks = masks.squeeze(1)
                        elif masks.dim() == 2:
                            masks = masks.unsqueeze(0).repeat(f, 1, 1)
                        masks = masks.to(self.device)

                        with torch.set_grad_enabled(False):
                            if step == 0:
                                # 第一个窗口：预测所有帧
                                model_out0 = self.model(torch.stack([imgs[0], imgs[0]], dim=0))
                                out0 = model_out0[0] if isinstance(model_out0, tuple) else model_out0
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
                                
                                if f > 1:
                                    model_out1 = self.model(torch.stack([imgs[0], imgs[1]], dim=0))
                                    out1 = model_out1[0] if isinstance(model_out1, tuple) else model_out1
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
                            else:
                                # 后续窗口：只预测最后一帧
                                model_out_cur = self.model(torch.stack([imgs[f-2], imgs[f-1]], dim=0))
                                out_cur = model_out_cur[0] if isinstance(model_out_cur, tuple) else model_out_cur
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

                            avg_mae = np.mean([abs(r) for r in epoch_res[-min(f if step == 0 else 1, len(epoch_res)):]])
                            pbar.update(1)
                            pbar.set_postfix({'MAE': f'{avg_mae:.2f}'})
                else:
                    for step, (imgs, keypoints, masks) in enumerate(self.dataloaders['val'][file]):
                        if imgs.dim() == 5:
                            assert imgs.shape[0] == 1
                            imgs = imgs.squeeze(0)
                        f, c, h, w = imgs.shape
                        imgs = imgs.to(self.device)

                        if masks.dim() == 5:
                            masks = masks.squeeze(0).squeeze(1)
                        elif masks.dim() == 4 and masks.shape[1] == 1:
                            masks = masks.squeeze(1)
                        elif masks.dim() == 2:
                            masks = masks.unsqueeze(0).repeat(f, 1, 1)
                        masks = masks.to(self.device)

                        with torch.set_grad_enabled(False):
                            if step == 0:
                                # 第一个窗口：预测所有帧
                                model_out0 = self.model(torch.stack([imgs[0], imgs[0]], dim=0))
                                out0 = model_out0[0] if isinstance(model_out0, tuple) else model_out0
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
                                
                                if f > 1:
                                    model_out1 = self.model(torch.stack([imgs[0], imgs[1]], dim=0))
                                    out1 = model_out1[0] if isinstance(model_out1, tuple) else model_out1
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
                            else:
                                # 后续窗口：只预测最后一帧
                                model_out_cur = self.model(torch.stack([imgs[f-2], imgs[f-1]], dim=0))
                                out_cur = model_out_cur[0] if isinstance(model_out_cur, tuple) else model_out_cur
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

                            avg_mae = np.mean([abs(r) for r in epoch_res[-min(f if step == 0 else 1, len(epoch_res)):]])
                            pbar.update(1)
                            pbar.set_postfix({'MAE': f'{avg_mae:.2f}'})
                
                epoch_res = np.array(epoch_res)
                if 'fdst' in args.data_dir or 'ucsd' in args.data_dir:
                    val_img_list = sorted(glob(os.path.join(args.data_dir+'/'+'val'+'/'+file_list[file], '*.jpg')),
                                          key=lambda x: int(x.split('/')[-1].split('.')[0]))
                else:
                    val_img_list = sorted(glob(os.path.join(args.data_dir+'/'+'val'+'/'+file_list[file], '*.jpg')),
                                          key=lambda x: int(x.split('_')[-1].split('.')[0]))
                
                # 新逻辑下，预测帧数应该等于图片总数，不需要截断
                # 但为了安全，保留检查
                if len(epoch_res) != len(val_img_list):
                    logging.warning(f"预测帧数 {len(epoch_res)} != 图片总数 {len(val_img_list)}")
                
                for e in epoch_res:
                    sum_res.append(e)
            sum_res = np.array(sum_res)
            mse = np.sqrt(np.mean(np.square(sum_res)))
            mae = np.mean(np.abs(sum_res))
            logging.info('Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                         .format(self.epoch, mse, mae, time.time() - epoch_start))

            model_state_dic = self.model.state_dict()
            if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
                self.best_mse = mse
                self.best_mae = mae
                logging.info("save best mse {:.2f} mae {:.2f} model epoch {}".format(self.best_mse,
                                                                                     self.best_mae,
                                                                                     self.epoch))
                if self.save_all:
                    torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model_{}.pth'.format(self.best_count)))
                    self.best_count += 1
                else:
                    torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model_{}.pth'.format(self.num)))

        elif 'venice' in args.data_dir:
            sum_res = []
            file_list = sorted(os.listdir(os.path.join(args.data_dir, 'val')), key=int)
            total_steps = sum([len(self.dataloaders['val'][file]) for file in range(len(file_list))])
            pbar = tqdm(total=total_steps, desc=f'Epoch {self.epoch} Val')
            
            for file in range(len(file_list)):
                epoch_res = []
                for step, (imgs, keypoints, masks) in enumerate(self.dataloaders['val'][file]):
                    b, f, c, h, w = imgs.shape
                    assert b == 1, 'the batch size should equal to 1 in validation mode'
                    
                    # 去掉batch维度：[B, T, C, H, W] -> [T, C, H, W]
                    imgs = imgs.squeeze(0).to(self.device)  # [T, C, H, W]
                    
                    # 处理masks
                    if masks.dim() > 3:  
                        masks = masks.squeeze(0).to(self.device)  # [T, H, W]
                    else:
                        masks = masks.to(self.device)  # [T, H, W] 或 [H, W]
                        if masks.dim() == 2:  # [H, W]
                            masks = masks.unsqueeze(0).repeat(f, 1, 1)  # [T, H, W]

                    with torch.set_grad_enabled(False):
                        if step == 0:
                            # 第一个窗口：预测所有帧
                            model_out0 = self.model(torch.stack([imgs[0], imgs[0]], dim=0))
                            out0 = model_out0[0] if isinstance(model_out0, tuple) else model_out0
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
                            
                            if f > 1:
                                model_out1 = self.model(torch.stack([imgs[0], imgs[1]], dim=0))
                                out1 = model_out1[0] if isinstance(model_out1, tuple) else model_out1
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
                        else:
                            # 后续窗口：只预测最后一帧
                            model_out_cur = self.model(torch.stack([imgs[f-2], imgs[f-1]], dim=0))
                            out_cur = model_out_cur[0] if isinstance(model_out_cur, tuple) else model_out_cur
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

                        pbar.update(1)
                        avg_mae = np.mean([abs(r) for r in epoch_res[-min(f if step == 0 else 1, len(epoch_res)):]])
                        pbar.set_postfix({'MAE': f'{avg_mae:.2f}'})
                
                epoch_res = np.array(epoch_res)
                val_img_list = sorted(glob(os.path.join(args.data_dir+'/'+'val'+'/'+file_list[file], '*.jpg')),
                                      key=lambda x: int(x.split('_')[-1].split('.')[0]))
                
                # 新逻辑下，预测帧数应该等于图片总数，不需要截断
                if len(epoch_res) != len(val_img_list):
                    logging.warning(f"Venice预测帧数 {len(epoch_res)} != 图片总数 {len(val_img_list)}")
                
                for e in epoch_res:
                    sum_res.append(e)
            sum_res = np.array(sum_res)
            mse = np.sqrt(np.mean(np.square(sum_res)))
            mae = np.mean(np.abs(sum_res))
            logging.info('Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                         .format(self.epoch, mse, mae, time.time() - epoch_start))

            model_state_dic = self.model.state_dict()
            if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
                self.best_mse = mse
                self.best_mae = mae
                logging.info("save best mse {:.2f} mae {:.2f} model epoch {}".format(self.best_mse,
                                                                                     self.best_mae,
                                                                                     self.epoch))
                if self.save_all:
                    torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model_{}.pth'.format(self.best_count)))
                    self.best_count += 1
                else:
                    torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model_{}.pth'.format(self.num)))

        else:
            epoch_res = []
            pbar = tqdm(self.dataloaders['val'], desc=f'Epoch {self.epoch} Val')
            for step, (imgs, keypoints, masks) in enumerate(pbar):
                b, f, c, h, w = imgs.shape
                assert b == 1, 'the batch size should equal to 1 in validation mode'
                
                # 去掉batch维度：[B, T, C, H, W] -> [T, C, H, W]
                imgs = imgs.squeeze(0).to(self.device)  # [T, C, H, W]
                
                # 处理masks
                if masks.dim() > 3:  
                    masks = masks.squeeze(0).to(self.device)  # [T, H, W]
                else:
                    masks = masks.to(self.device)  # [T, H, W] 或 [H, W]
                    if masks.dim() == 2:  # [H, W]
                        masks = masks.unsqueeze(0).repeat(f, 1, 1)  # [T, H, W]
                # 若时间长度与当前序列不一致，进行对齐
                if masks.dim() == 3 and masks.shape[0] != f:
                    if masks.shape[0] == 1:
                        masks = masks.repeat(f, 1, 1)
                    else:
                        masks = masks[:f]

                with torch.set_grad_enabled(False):
                    if step == 0:
                        # 第一个窗口：预测所有帧
                        model_out0 = self.model(torch.stack([imgs[0], imgs[0]], dim=0))
                        out0 = model_out0[0] if isinstance(model_out0, tuple) else model_out0
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
                        
                        if f > 1:
                            model_out1 = self.model(torch.stack([imgs[0], imgs[1]], dim=0))
                            out1 = model_out1[0] if isinstance(model_out1, tuple) else model_out1
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
                    else:
                        # 后续窗口：只预测最后一帧
                        model_out_cur = self.model(torch.stack([imgs[f-2], imgs[f-1]], dim=0))
                        out_cur = model_out_cur[0] if isinstance(model_out_cur, tuple) else model_out_cur
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
                
                avg_mae = np.mean([abs(r) for r in epoch_res[-min(f if step == 0 else 1, len(epoch_res)):]])
                pbar.set_postfix({'MAE': f'{avg_mae:.2f}'})
            
            pbar.close()

            epoch_res = np.array(epoch_res)
            val_img_list = sorted(glob(os.path.join(os.path.join(args.data_dir, 'val'), '*.jpg')),
                                  key=lambda x: int(x.split('_')[-1].split('.')[0]))
            
            # 新逻辑下，预测帧数应该等于图片总数，不需要截断
            if len(epoch_res) != len(val_img_list):
                logging.warning(f"其他数据集预测帧数 {len(epoch_res)} != 图片总数 {len(val_img_list)}")
            
            mse = np.sqrt(np.mean(np.square(epoch_res)))
            mae = np.mean(np.abs(epoch_res))
            logging.info('Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                         .format(self.epoch, mse, mae, time.time()-epoch_start))

            model_state_dic = self.model.state_dict()
            if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
                self.best_mse = mse
                self.best_mae = mae
                logging.info("save best mse {:.2f} mae {:.2f} model epoch {}".format(self.best_mse,
                                                                                     self.best_mae,
                                                                                     self.epoch))
                if self.save_all:
                    torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model_{}.pth'.format(self.best_count)))
                    self.best_count += 1
                else:
                    torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model_{}.pth'.format(self.num)))
