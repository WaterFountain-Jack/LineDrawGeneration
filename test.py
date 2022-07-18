import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from model7 import Generator
from model7 import NetD
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml import sklearn2pmml


if __name__ == '__main__':
    a = torch.ones(1,2)
    a = a.unsqueeze(0)
    print(a.shape)
    # netG = Generator()
    # #sklearn2pmml(netG, "iris.pmml", with_repr=True)  # 输出PMML文件
    # torch.save(netG, 'G.pth')