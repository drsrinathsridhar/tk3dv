import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os, sys, argparse, math
import numpy as np

FileDirPath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(FileDirPath, '..'))

import ptUtils
import ptNets

class SimpleCAE(ptNets.ptNet):
    def __init__(self, Args=None, DataParallelDevs=None):
        super().__init__(InputArgs=Args)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class FCBottleNeck(nn.Module): # Just a fully connected bottleneck layer
    def __init__(self, InFeatureSize):
        super().__init__()
        self.FC1 = nn.Linear(InFeatureSize, 2048) # TODO: figure out sizes
        self.FC2 = nn.Linear(2048, 2048)
        self.FC3 = nn.Linear(2048, InFeatureSize)

    def forward(self, x):
        x_pe = x
        x_pe = F.relu(self.FC1(x_pe))
        x_pe = F.relu(self.FC2(x_pe))
        x_pe = self.FC3(x_pe)

        return x_pe

class DeepCAE(ptNets.ptNet):
    def __init__(self, Args=None, DataParallelDevs=None, InputChannels=1):
        super().__init__(Args)
        self.encoder = self.Encoder(InputChannels=InputChannels)
        self.decoder = self.Decoder(OutputChannels=InputChannels)
        self.fcbn = FCBottleNeck(self.encoder.FeatureSize) # Bottleneck identical to DeepPECAE

    def forward(self, x):
        z = self.encoder(x)
        z = self.fcbn(z)
        y = self.decoder(z)
        return y

    class Encoder(nn.Module):
        def __init__(self, InputChannels=1):
            super().__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(InputChannels, 32, 3, padding=1),  # batch x 32 x 28 x 28
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 32, 3, padding=1),  # batch x 32 x 28 x 28
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 64, 3, padding=1),  # batch x 64 x 28 x 28
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 64, 3, padding=1),  # batch x 64 x 28 x 28
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(2, 2)  # batch x 64 x 14 x 14
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1),  # batch x 128 x 14 x 14
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 128, 3, padding=1),  # batch x 128 x 14 x 14
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(128, 256, 3, padding=1),  # batch x 256 x 7 x 7
                nn.ReLU()
            )
            self.FeatureSize = 256*7*7

        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            batch_size = x.shape[0]
            out = out.view(batch_size, -1)
            return out


    class Decoder(nn.Module):
        def __init__(self, OutputChannels=1):
            super().__init__()
            self.layer1 = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),  # batch x 128 x 14 x 14
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.ConvTranspose2d(128, 128, 3, 1, 1),  # batch x 128 x 14 x 14
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.ConvTranspose2d(128, 64, 3, 1, 1),  # batch x 64 x 14 x 14
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.ConvTranspose2d(64, 64, 3, 1, 1),  # batch x 64 x 14 x 14
                nn.ReLU(),
                nn.BatchNorm2d(64)
            )
            self.layer2 = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 3, 1, 1),  # batch x 32 x 14 x 14
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.ConvTranspose2d(32, 32, 3, 1, 1),  # batch x 32 x 14 x 14
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.ConvTranspose2d(32, OutputChannels, 3, 2, 1, 1),  # batch x 1 x 28 x 28
                nn.ReLU()
            )

        def forward(self, x):
            batch_size = x.shape[0]
            out = x.view(batch_size, 256, 7, 7)
            out = self.layer1(out)
            out = self.layer2(out)
            return out

class DeepCAE5(DeepCAE):
    def __init__(self, Args=None, DataParallelDevs=None, InputChannels=1):
        super().__init__(Args, DataParallelDevs=DataParallelDevs, InputChannels=InputChannels)

    def forward(self, x):
        BatchSize = x.shape[0]
        SetSize = x.shape[1]
        nChannels = x.shape[2]
        assert (nChannels == 1)
        W = x.shape[3]
        H = x.shape[4]

        # Squeeze the set dimension into the channel dimension
        x_s = torch.squeeze(x, dim=2)
        # print(x_s.size())

        z = self.encoder(x_s)
        z = self.fcbn(z)
        y = self.decoder(z)
        # print(y.size())
        y = torch.unsqueeze(y, dim=2)
        # print(y.size())
        return y

class DeepPECAE(DeepCAE):
    def __init__(self, Args=None, DataParallelDevs=None):
        super().__init__(Args, DataParallelDevs=DataParallelDevs)
        self.encoder = self.Encoder()
        self.decoder = self.Decoder()
        self.permeq = self.PELayer(self.encoder.FeatureSize)

    def forward(self, x):
        BatchSize = x.shape[0]
        SetSize = x.shape[1]
        nChannels = x.shape[2]
        W = x.shape[3]
        H = x.shape[4]

        z_s = []
        for SetIdx in range(SetSize):
            z = self.encoder(x[:, SetIdx, :, :, :])
            z_s.append(z)
        z_s = torch.stack(z_s)

        z_mn = self.permeq(z_s) # Max-normalized features

        o_s = []
        for SetIdx in range(SetSize):
            o = self.decoder(z_mn[SetIdx, :, :])
            o_s.append(o)

        o_s = torch.stack(o_s, dim=1)

        # print('Input size:', x.size())
        # print('Set features size:', z_s.size())
        # print('Output size:', o_s.size())

        return o_s

    class PELayer(FCBottleNeck): # Permutation equivariant layer
        def __init__(self, InFeatureSize):
            super().__init__(InFeatureSize)

        def maxNormalize(self, x):
            SetSize, BatchSize, ZSize = x.size()
            # Max pool along feature dimension
            x_p = x.permute(1, 2, 0)
            MP = F.max_pool1d(x_p, SetSize)
            MP = MP.permute(2, 0, 1)
            MN = x - MP

            # print('x_p:', x_p.size())
            # print('MPSize:', MP.size())
            # print('MNSize:', MN.size())

            return MN

        def forward(self, x):
            # print('Permeq size', x.size())

            # Permutation equivariant layer is just a fully connected layer where the input features are max normalized
            x_pe = self.maxNormalize(x)
            x_pe = F.relu(self.FC1(x_pe))

            x_pe = self.maxNormalize(x_pe)
            x_pe = F.relu(self.FC2(x_pe))

            x_pe = self.maxNormalize(x_pe)
            x_pe = self.FC3(x_pe)

            return x_pe
