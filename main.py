"""
    PyTorch training code for Wide Residual Networks:
    http://arxiv.org/abs/1605.07146
    2019 Jun Li
"""


import argparse
parser = argparse.ArgumentParser(description='Wide Residual Networks')

# Model options
parser.add_argument('--model', default='resnet', type=str)
parser.add_argument('--dataset', default='CIFAR10', type=str)
parser.add_argument('--dataroot', default='/scratch/liju2/data/', type=str)
parser.add_argument('--dtype', default='float', type=str)
parser.add_argument('--n_workers', default=0, type=int)

# Training options
parser.add_argument('--batchSize', default=128, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--weightDecay', default=0.0005, type=float)
parser.add_argument('--bnDecay', default=0, type=float)
parser.add_argument('--omega', default=0.1, type=float)
parser.add_argument('--grad_clip', default=0.1, type=float)
parser.add_argument('--epoch_step', default='[60,120,160]', type=str,
                    help='json list with epochs to drop lr on')
parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--optim_method', default='SGD', type=str)
parser.add_argument('--randomcrop_pad', default=4, type=float)

# Device options
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--save', default='', type=str,
                    help='save parameters and logs in this folder')
parser.add_argument('--ngpu', default=1, type=int,
                    help='number of GPUs to use for training')
parser.add_argument('--gpu_id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
                    
                    
def main():
    pass
    

    
if __name__ == '__main__':
    main()
