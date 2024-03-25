import os
import cv2
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from dataloader import get_train_augmentation, get_test_augmentation, get_loader, gt_to_tensor
from util.utils import AvgMeter
from util.metrics import Evaluation_metrics
from util.losses import Optimizer, Scheduler, Criterion
from model.net import EMSNet
from torch.autograd import Variable
import numpy as np
from math import exp

    
class Trainer():
    def __init__(self, args, save_path):
        super(Trainer, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.size = args.img_size
        self.tr_img_folder = os.path.join(args.data_path, args.dataset, 'Train/images/')
        self.tr_gt_folder = os.path.join(args.data_path, args.dataset, 'Train/masks/')
        self.tr_edge_folder = os.path.join(args.data_path, args.dataset, 'Train/edges/')

        self.train_transform = get_train_augmentation(img_size=args.img_size, ver=args.aug_ver)
        self.test_transform = get_test_augmentation(img_size=args.img_size)

        self.train_loader = get_loader(self.tr_img_folder, self.tr_gt_folder, self.tr_edge_folder, phase='train',
                                       batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                       transform=self.train_transform, seed=args.seed)


        self.te_img_folder = os.path.join(args.data_path, args.dataset, 'Test/images/')
        self.te_gt_folder = os.path.join(args.data_path, args.dataset, 'Test/masks/')
        self.test_loader = get_loader(self.te_img_folder, self.te_gt_folder, edge_folder=None, phase='test',
                                 batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, transform=self.test_transform)

        # Network
        self.model = EMSNet(args).to(self.device)
        self.model.resnet.resnet.load_state_dict(torch.load(args.premodel),strict=False)
        if args.multi_gpu:
            self.model = nn.DataParallel(self.model).to(self.device)

        # Loss and Optimizer
        self.criterion = Criterion(args)
        self.optimizer = Optimizer(args, self.model)
        self.scheduler = Scheduler(args, self.optimizer)
        self.structureloss = weightEdgeStructure()
        # Train / Validate
        min_loss = 9999
        early_stopping = 0
        t = time.time()
        for epoch in range(1, args.epochs + 1):
            torch.cuda.empty_cache()
            # val_loss, val_mae = self.test(args)
            train_loss, train_mae = self.training(args, epoch)
            val_loss,val_mae = self.test(args)
            if args.scheduler == 'Reduce':
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            # Save models
            if val_loss < min_loss:
                early_stopping = 0
                best_epoch = epoch
                min_mae = val_mae
                min_loss = val_loss
                torch.save(self.model.state_dict(), '%s/best.pth' % (save_path))
                print(f'-----------------SAVE:{best_epoch}epoch----------------')
            else:
                early_stopping += 1

            if early_stopping == args.patience + 125:
                break

        print(f'\nBest Val Epoch:{best_epoch} | Val Loss:{min_loss:.4f} | Val MAE:{min_mae:.4f} '
              f'time: {(time.time() - t) / 60:.3f}M')


        end = time.time()
        print(f'Total Process time:{(end - t) / 60:.3f}Minute')

    def training(self, args, epoch):
        self.model.train()
        train_loss = AvgMeter()
        sal_loss = AvgMeter()
        structure_loss = AvgMeter()
        train_mae = AvgMeter()

        for images, masks, edges in self.train_loader:
            images = torch.tensor(images, device=self.device, dtype=torch.float32)
            masks = torch.tensor(masks, device=self.device, dtype=torch.float32)
            edges = torch.tensor(edges, device=self.device, dtype=torch.float32)

            self.optimizer.zero_grad()
            pred2,pred1,pred0,p = self.model(images)
            loss0 = self.criterion(pred2, masks)
            loss1 = self.criterion(pred1, masks)/2
            loss2 = self.criterion(pred0, masks)/4
            loss3 = self.criterion(p, masks)/8
            structureloss = self.structureloss(pred2, masks, edges)

            salloss = loss0 + loss1+ loss2 + loss3
            loss = salloss + structureloss * 0.5
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), args.clipping)
            self.optimizer.step()

            # Metric
            mae = torch.mean(torch.abs(pred2 - masks))

            # log
            train_loss.update(loss.item(), n=images.size(0))
            train_mae.update(mae.item(), n=images.size(0))
            sal_loss.update(salloss.item(), n=images.size(0))
            structure_loss.update(structureloss.item(), n=images.size(0))

        print(f'Epoch:[{self.epoch:03d}/{args.epochs:03d}]')
        print(f'Train Loss:{train_loss.avg:.3f} | MAE:{train_mae.avg:.3f}')
        print(f'sal_loss:{sal_loss.avg:.3f} | structure_loss:{structure_loss.avg:.3f}')
        print('----------------------------------------------------------------------')
        return train_loss.avg, train_mae.avg


    def test(self, args):



        self.model.eval()
        test_loss = AvgMeter()
        test_mae = AvgMeter()
        Eval_tool = Evaluation_metrics(args.dataset, self.device)

        with torch.no_grad():
            for i, (images, masks, original_size, image_name) in enumerate(self.test_loader):
                images = torch.tensor(images, device=self.device, dtype=torch.float32)
                # masks = torch.tensor(masks, device=self.device, dtype=torch.float32)
                pred2,pred1,pred0,p= self.model(images)
                H, W = original_size

                for j in range(images.size(0)):
                    mask = gt_to_tensor(masks[j])
                    # # print('mask.size')
                    # # print(mask.size())
                    h, w = H[j].item(), W[j].item()
                    # # print('pred2.size')
                    # # print(pred2.size())
                    output = F.interpolate(pred2[j].unsqueeze(0), size=(h, w), mode='bilinear')
                    # print('output.size')
                    # print(output.size())
                    loss = self.criterion(output, mask)

                    # Metric
                    mae, max_f, avg_f, s_score = Eval_tool.cal_total_metrics(output, mask)

                    # log
                    test_loss.update(loss.item(), n=1)
                    test_mae.update(mae, n=1)

            test_loss = test_loss.avg
            test_mae = test_mae.avg
        print(f'Epoch:[{self.epoch:03d}/{args.epochs:03d}]')
        print(f'Test Loss:{test_loss:.4f} | MAE:{test_mae:.4f}')
        print('--------------------------------------------------------')
        return test_loss, test_mae


class Tester():
    def __init__(self, args):
        super(Tester, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.test_transform = get_test_augmentation(img_size=args.img_size)
        self.args = args

        # Network
        self.model = EMSNet(args).to(self.device)
        if args.multi_gpu:
            self.model = nn.DataParallel(self.model).to(self.device)

        self.model.load_state_dict(torch.load(args.loadmodel))
        self.criterion = Criterion(args)
        te_img_folder = os.path.join(args.data_path, args.dataset, 'Test/images/')
        te_gt_folder = os.path.join(args.data_path, args.dataset, 'Test/masks/')
        self.test_loader = get_loader(te_img_folder, te_gt_folder, edge_folder=None, phase='test',
                                      batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.num_workers, transform=self.test_transform)

        if args.save_map is not None:
            os.makedirs(os.path.join('mask/', self.args.model_name, self.args.dataset), exist_ok=True)

    def test(self):
        self.model.eval()

        with torch.no_grad():
            for i, (images, masks, original_size, image_name) in tqdm(enumerate(tqdm(self.test_loader))):
                images = torch.tensor(images, device=self.device, dtype=torch.float32)

                pred1, pred2, pred3, pred4 = self.model(images)
                H, W = original_size
                for i in range(images.size(0)):
                    h, w = H[i].item(), W[i].item()
                    output = F.interpolate(pred1[i].unsqueeze(0), size=(h, w), mode='bilinear')
                    # Save prediction map
                    if self.args.save_map is not None:
                        output = (output.squeeze().detach().cpu().numpy()*255.0).astype(np.uint8)   # convert uint8 type
                        cv2.imwrite(os.path.join('mask/', self.args.model_name,  self.args.dataset, image_name[i]+'.png'), output)

        return None
    
class weightEdgeStructure():
    def __init__(self):
        pass
    def __call__(self, pred, mask,edge):
        complexity = calculate_ssim_weighted_complexity_optimized(edge, pred, mask)
        return complexity
    

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def calculate_ssim_weighted_complexity_optimized(edge_mask, image, gt, kernel_size=11):
    ssim_module = SSIM(window_size=kernel_size).to(edge_mask.device)
    ssim_map = ssim_module(image, gt)
    # 使用边缘掩码进行加权
    weight = calculate_edge_weight(edge_mask)
    # print(weight.shape)
    # print(ssim_map.shape)
    # print(edge_mask.shape)
    weighted_ssim = edge_mask * (1+ weight) * (1-ssim_map)
    ssim_loss = weighted_ssim.sum()/edge_mask.sum()
    # print(ssim_loss)
    return ssim_loss

def calculate_edge_weight(mask, kernel_size=11):
    """
    计算每个边缘像素周围的边缘像素数与核大小的比例。
    
    :param mask: 二值边缘掩码（边缘像素为1，背景为0）
    :param kernel_size: 卷积核的大小
    :return: 边缘加权图
    """
    # 创建一个全为1的卷积核
    kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32)
    kernel = kernel.to(mask.device)
    # 对mask进行卷积操作
    # padding保证输出大小与输入相同
    edge_counts = F.conv2d(mask, kernel, padding=kernel_size // 2)

    # 计算边缘像素的加权值
    edge_weight = edge_counts / (kernel_size * kernel_size)

    return edge_weight
