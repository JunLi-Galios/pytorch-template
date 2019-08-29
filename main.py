import os
import json

import torch
import models
import datasets
import optimizers
import lr_schedulers
import criterions
from runner import Runner
from logger import Logger


import argparse
parser = argparse.ArgumentParser(description='Pytorch deep learning template')
parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory name to save the model')

# Dataset options
parser.add_argument('--dataset', default='CIFAR10', type=str)
parser.add_argument('--dataroot', default='../datasets/', type=str)

# Model options
parser.add_argument('--model_name', default='vgg11_bn', type=str)
parser.add_argument('-mp', '--model_parameter', default='{"depth":18,"pretrained":false}', type=json.loads)
parser.add_argument('--resume_file', default='', type=str)


# Training options
parser.add_argument('--train_batch', default=128, type=int)
parser.add_argument('--test_batch', default=1024, type=int)
parser.add_argument('--optim_method', default='ADAM', type=str)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', default=0.0005, type=float)
parser.add_argument('--lr_scheduler', default='Step', type=str)
parser.add_argument('-lsv','--lr_scheduler_values', default='{"step_size":100}', type=json.loads)
parser.add_argument('--loss', default='CrossEntropy', type=str)
parser.add_argument('--epochs', default=1, type=int)
parser.add_argument('--save_interval', default=10, type=int)
parser.add_argument('--num_workers', default=0, type=int)

# Device options
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--save', default='', type=str,
                    help='save parameters and logs in this folder')
parser.add_argument('--ngpu', default=1, type=int,
                    help='number of GPUs to use for training')
parser.add_argument('--gpu_id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='enables CUDA training')
                    
def main():
    args = parser.parse_args()

    args.save_dir = "%s/%s" % (os.getcwd(), args.save_dir)
    if os.path.exists(args.save_dir) is False:
        os.makedirs(args.save_dir)

    
    logger = Logger(args.save_dir)
    logger.will_write(str(args) + "\n")

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() and args.no_cuda is False else 'cpu')

    train_dataset, test_dataset, num_classes = getattr(datasets, args.dataset)(args.dataroot)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.train_batch,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=args.num_workers)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=args.test_batch,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=args.num_workers)
    net = getattr(models, args.model_name)(num_classes=num_classes, **args.model_parameter).to(device)
    
    optimizer = getattr(optimizers, args.optim_method)(net.parameters(), args.lr, args.momentum, args.weight_decay)
    scheduler = getattr(lr_schedulers, args.lr_scheduler)(optimizer, **args.lr_scheduler_values)

    criterion = getattr(criterions, args.loss)().to(device)

    model = Runner(args.model_name, net, optimizer, device, criterion, args.epochs, logger, args.save_dir, args.save_interval, scheduler, args.resume_file)

    model.train(train_loader, test_loader)
    model.test(train_loader, test_loader)
    

 

if __name__ == "__main__":
    main()
