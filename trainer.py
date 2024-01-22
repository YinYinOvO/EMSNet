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
from model.net import SDNet

class Tester():
    def __init__(self, args):
        super(Tester, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.test_transform = get_test_augmentation(img_size=args.img_size)
        self.args = args

        # Network
        self.model = SDNet(args).to(self.device)
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
