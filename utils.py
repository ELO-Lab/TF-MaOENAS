import torch
import os

OP_NAMES = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]

def to_string(ind):
    cell = ''
    node = 0
    for i in range(len(ind)):
        gene = ind[i]
        cell += '|' + OP_NAMES[gene] + '~' + str(node)
        node += 1
        if i == 0 or i == 2:
            node = 0
            cell += '|+'
    cell += '|'
    return cell

def get_num_classes(args):
    return 100 if args.dataset == 'cifar100' else 10 if args.dataset == 'cifar10' else 120

def get_input(args, train_loader):
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':    
        return torch.randn(len(train_loader), 3, 32, 32).to(args.device)
    else:
        return torch.randn(len(train_loader), 3, 16, 16).to(args.device)

def create_exp_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Experiment dir : {}'.format(path))


