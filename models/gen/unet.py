import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from collections import OrderedDict

from models.layers import CBR
from models.models_utils import weights_init, print_network


class _UNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.c0 = nn.Conv2d(in_ch, 64, 3, 1, 1)
        self.c1 = CBR(64, 128, bn=True, sample='down', activation=nn.LeakyReLU(0.2, True), dropout=False)
        self.c2 = CBR(128, 256, bn=True, sample='down', activation=nn.LeakyReLU(0.2, True), dropout=False)
        self.c3 = CBR(256, 512, bn=True, sample='down', activation=nn.LeakyReLU(0.2, True), dropout=False)
        self.c4 = CBR(512, 512, bn=True, sample='down', activation=nn.LeakyReLU(0.2, True), dropout=False)
        self.c5 = CBR(512, 512, bn=True, sample='down', activation=nn.LeakyReLU(0.2, True), dropout=False)
        self.c6 = CBR(512, 512, bn=True, sample='down', activation=nn.LeakyReLU(0.2, True), dropout=False)
        self.c7 = CBR(512, 512, bn=True, sample='down', activation=nn.LeakyReLU(0.2, True), dropout=False)

        self.dc0 = CBR(512, 512, bn=True, sample='up', activation=nn.ReLU(True), dropout=True)
        self.dc1 = CBR(1024, 512, bn=True, sample='up', activation=nn.ReLU(True), dropout=True)
        self.dc2 = CBR(1024, 512, bn=True, sample='up', activation=nn.ReLU(True), dropout=True)
        self.dc3 = CBR(1024, 512, bn=True, sample='up', activation=nn.ReLU(True), dropout=False)
        self.dc4 = CBR(1024, 256, bn=True, sample='up', activation=nn.ReLU(True), dropout=False)
        self.dc5 = CBR(512, 128, bn=True, sample='up', activation=nn.ReLU(True), dropout=False)
        self.dc6 = CBR(256, 64, bn=True, sample='up', activation=nn.ReLU(True), dropout=False)
        self.dc7 = nn.Conv2d(128, out_ch, 3, 1, 1)

    def forward(self, x):
        hs = [nn.LeakyReLU(0.2, True)(self.c0(x))]
        hs.append(self.c1(hs[0]))
        hs.append(self.c2(hs[1]))
        hs.append(self.c3(hs[2]))
        hs.append(self.c4(hs[3]))
        hs.append(self.c5(hs[4]))
        hs.append(self.c6(hs[5]))
        hs.append(self.c7(hs[6]))
        h = self.dc0(hs[-1])
        h = self.dc1(torch.cat((h, hs[-2]), 1))
        h = self.dc2(torch.cat((h, hs[-3]), 1))
        h = self.dc3(torch.cat((h, hs[-4]), 1))
        h = self.dc4(torch.cat((h, hs[-5]), 1))
        h = self.dc5(torch.cat((h, hs[-6]), 1))
        h = self.dc6(torch.cat((h, hs[-7]), 1))
        h = self.dc7(torch.cat((h, hs[-8]), 1))
        return h


class UNet(nn.Module):
    def __init__(self, in_ch, out_ch, gpu_ids):
        super().__init__()
        self.gpu_ids = gpu_ids

        self.gen = nn.Sequential(OrderedDict([('gen', _UNet(in_ch, out_ch))]))

        self.gen.apply(weights_init)

    def forward(self, x):
        if self.gpu_ids:
            return nn.parallel.data_parallel(self.gen, x, self.gpu_ids)
        else:
            return self.gen(x)
