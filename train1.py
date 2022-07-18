import argparse
import torch
import torchvision
import torchvision.utils as vutils
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torch.autograd import Variable
from PIL import Image
from random import randint


import cv2
import numpy as np

import cfg
import model
from model import NetD


args = cfg.parse_args()

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=args.eval_batch_size)#一次训练所取的样本数
parser.add_argument('--imageSize', type=int, default=96)#图片的大小,这个训练集里面图片的大小都是一致的
parser.add_argument('--nz', type=int, default=args.latent_dim, help='size of the latent z vector')#暂时不知道
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--epoch', type=int, default=25, help='number of epochs to train for')#训练的轮数
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')#学习率
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')#参数
parser.add_argument('--data_path', default='data/', help='folder to train data')
parser.add_argument('--outf', default='myImgs/', help='folder to output images and model checkpoints')
opt = parser.parse_args()

#图像读入与预处理
transforms = torchvision.transforms.Compose([

    torchvision.transforms.Scale(opt.imageSize),
    torchvision.transforms.CenterCrop(opt.imageSize),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])#使用平均值和标准偏差对张量图像进行规格化,消除量纲

dataset = torchvision.datasets.ImageFolder(opt.data_path, transform=transforms)



def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [data, target]

dataloader = torch.utils.data.DataLoader(#数据取器
    dataset=dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    drop_last=True,

)


netG = model.Generator(args=args)
netD = NetD(opt.ndf)

criterion = nn.BCELoss()
optimizerG = torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerD = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

for epoch in range(1, opt.epoch + 1):
    for i, (imgs,_) in enumerate(dataloader):

        # print("size:")
        # print(imgs.shape)
        # print(type(imgs))
        # imgs = torch.squeeze(imgs)  # 若不标注删除第几维度，则会删除所有为1的维度
        # plt.imshow(imgs.reshape(opt.imageSize,opt.imageSize,3))
        # plt.imshow(imgs.numpy().reshape(opt.imageSize,opt.imageSize,3))

        # 固定生成器G，训练鉴别器D
        optimizerD.zero_grad()          #把模型中参数的梯度设为0
        ## 让D尽可能的把真图片判别为1
        imgs=imgs
        output = netD(imgs)
        output = torch.squeeze(output)#squeeze去掉维数为1的的维度 unsqueeze对维度进行扩充
        label.data.fill_(real_label)
        label=label
        errD_real = criterion(output, label)
        errD_real.backward()

        ## 让D尽可能把假图片判别为0
        label.data.fill_(fake_label)
        noise = torch.randn(opt.batchSize, opt.nz)
        noise=noise
        fake = netG(noise)  # 生成假图
        output = netD(fake.detach()) #避免梯度传到G，因为G不用更新
        output = torch.squeeze(output)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        errD = errD_fake + errD_real
        optimizerD.step()

        # 固定鉴别器D，训练生成器G
        optimizerG.zero_grad()
        # 让D尽可能把G生成的假图判别为1
        label.data.fill_(real_label)
        label = label
        output = netD(fake)
        output = torch.squeeze(output)


        errG = criterion(output, label)
        errG.backward()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.3f Loss_G %.3f'
              % (epoch, opt.epoch, i, len(dataloader), errD.item(), errG.item()))

    vutils.save_image(fake.data,
                      '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                      normalize=True)
    torch.save(netG.state_dict(), '%s/netG_%03d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_%03d.pth' % (opt.outf, epoch))






