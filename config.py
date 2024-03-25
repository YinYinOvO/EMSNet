import argparse

def getConfig():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, default='train', help='Model Training or Testing options')
    parser.add_argument('--exp_num', default=0, type=str, help='experiment_number')
    parser.add_argument('--dataset', type=str, default='DUTS', help='DUTS')
    parser.add_argument('--data_path', type=str, default='data/')
    # Training parameter settings
    parser.add_argument('--img_size', type=int, default=384)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--criterion', type=str, default='API', help='API or bce')
    parser.add_argument('--scheduler', type=str, default='Reduce', help='Reduce or Step')
    parser.add_argument('--aug_ver', type=int, default=2, help='1=Normal, 2=Hard')
    parser.add_argument('--lr_factor', type=float, default=0.1)
    parser.add_argument('--clipping', type=float, default=2, help='Gradient clipping')
    parser.add_argument('--patience', type=int, default=5, help="Scheduler ReduceLROnPlateau's parameter & Early Stopping(+5)")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_map', type=bool, default=None, help='Save prediction map')

    parser.add_argument('--model_name', type=str, default='Test')
    parser.add_argument('--loadmodel', type=str, default='./results/best.pth')
    parser.add_argument('--premodel', type=str, default='./premodels/resnet50.pth')
    parser.add_argument('--save_path', type=str, default='results/test3/')
    # Hardware settings
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=8)
    cfg = parser.parse_args()

    return cfg
