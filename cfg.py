# -*- coding: utf-8 -*-
# @Date    : 2019-07-25
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0

import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--latent_dim',
        type=int,
        default=128,
        help='dimensionality of the latent space:传入噪点的维度')
    parser.add_argument(
        '--bottom_width',
        type=int,
        default=4,
        help="the base resolution of the GAN：初始batches正方形边长的个数")
    parser.add_argument('--embed_dim', type=int, default=64,
                        help='The base channel num of gen ：初始batches每个正方形的大小 8x8')


    parser.add_argument('--eval_batch_size', type=int, default=64,help=('每次取突变的batch数目')) #10

    opt = parser.parse_args()

    return opt
