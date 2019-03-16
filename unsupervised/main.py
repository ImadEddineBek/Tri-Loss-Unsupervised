import argparse
import os

from data_loader import get_loader
from solver import Solver
from torch.backends import cudnn


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    svhn_loader, mnist_loader, mnist_val_loader = get_loader(config)
    print("loaded")
    solver = Solver(config, svhn_loader, mnist_loader, mnist_val_loader)
    cudnn.benchmark = True

    # create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    '''if not os.path.exists(config.sample_path):
        os.makedirs(config.sample_path)'''
    if config.mode == 'train':
        solver.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--alpha_s', type=float, default=0.5)
    parser.add_argument('--alpha_t', type=float, default=0.8)
    parser.add_argument('--beta_c', type=float, default=1)
    parser.add_argument('--beta_sep', type=float, default=1.5)
    parser.add_argument('--beta_p', type=float, default=4)

    # training hyper-parameters
    parser.add_argument('--train_iters', type=int, default=1000)
    parser.add_argument('--pretrain_iters', type=int, default=20000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--lr_d', type=float, default=0.0003)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)

    # misc
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--mnist_path', type=str, default='../data/mnist')
    parser.add_argument('--svhn_path', type=str, default='../data/svhn')
    parser.add_argument('--log_step', type=int, default=10)

    # Possible values: "usps" , "svhn_extra" , "mnist" , "svhn"
    parser.add_argument('--source', type=str, default='mnist')
    parser.add_argument('--target', type=str, default='svhn')

    config = parser.parse_args()
    print(config)
    main(config)
