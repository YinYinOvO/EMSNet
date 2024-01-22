import os
import pprint
import random
import warnings
import torch
import numpy as np
from trainer import Tester

from config import getConfig
warnings.filterwarnings('ignore')
args = getConfig()


def main(args):
    print('<---- Training Params ---->')
    pprint.pprint(args)

    # Random Seed
    seed = args.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.action == 'test':
        datasets = ['DUTS', 'DUT-O', 'HKU-IS', 'ECSSD', 'PASCAL-S']
        for dataset in datasets:
            args.dataset = dataset
            Tester(args).test()



if __name__ == '__main__':
    main(args)