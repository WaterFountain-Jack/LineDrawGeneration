# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os
from torch import nn, optim
from tqdm import tqdm
import cfg
import torchvision
import numpy as np
import torch
import model
import model2
import model3
import model4
import model6
import torchvision.utils as vutils


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
transforms = torchvision.transforms.Compose([

    torchvision.transforms.ToTensor(),
    #torchvision.transforms.Grayscale(num_output_channels=3),  # 彩色图像转灰度图像num_output_channels默认1
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # torchvision.transforms.Scale(128),
    # torchvision.transforms.CenterCrop(128),
    ])#使

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    '''
        transformer验证
    '''
    dataset = torchvision.datasets.ImageFolder('./testdata1', transform=transforms)
    dataloader = torch.utils.data.DataLoader(  # 数据取器
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        drop_last=True,
    )
    for i, (img) in enumerate(dataloader):
        img = img[0].to('cuda')
        vutils.save_image(img.data,'%s/%04d.png' % ('testoutput', i + 1),normalize=True)

    '''
        model6 fid 计算
    '''
    # dataset = torchvision.datasets.ImageFolder('./data1', transform=transforms)
    # dataloader = torch.utils.data.DataLoader(  # 数据取器
    #     dataset=dataset,
    #     batch_size=1,
    #     shuffle=False,
    #     drop_last=True,
    # )
    #
    #
    # path_model = "./netG_050.pth"
    # model_loaded = model6.Generator()
    # model_loaded.load_state_dict(torch.load(path_model))
    # model_loaded.to('cuda')
    #
    # for i, (img) in enumerate(dataloader):
    #     img = img[0].to('cuda')
    #     output_list = model_loaded(img)
    #     output_dict = output_list[0]
    #     vutils.save_image(output_dict.data,'%s/%04d.png' % ('output', i + 1),normalize=True)

    '''
        model 4
    '''
    # G = model4.Generator()
    # D = model4.NetD(128)
    # x = torch.Tensor(10, 3, 128, 128)
    #
    #
    #
    # print(D(x).size())
    # print(G(x).size())

    '''
    model 3
    '''
    # G = model3.Generator()
    # x = torch.Tensor(10,3,128,128)
    # print(G(x).size())

    '''
    model 2
    '''
    # args = cfg.parse_args()
    # G = model2.Generator(args=args)
    # z = torch.ones(args.eval_batch_size, args.latent_dim)  # 10  128
    # # z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.eval_batch_size, args.latent_dim)))
    # image = G(z)
    # print(image.size())



    '''
    model 1
    '''
    # args = cfg.parse_args()
    # G = model.Generator(args=args)
    # z = torch.ones(args.eval_batch_size, args.latent_dim) # 10  128
    # #z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.eval_batch_size, args.latent_dim)))
    # G(z)


    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
