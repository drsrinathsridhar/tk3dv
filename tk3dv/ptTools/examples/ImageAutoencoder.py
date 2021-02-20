import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

import sys, os, argparse

FileDirPath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(FileDirPath, '../models'))
sys.path.append(os.path.join(FileDirPath, '..'))

from tk3dv.ptTools.models import UNet
from tk3dv.ptTools.models import SegNet
from tk3dv.ptTools.loaders import CameraDataset
from tk3dv.ptTools.loaders import GenericImageDataset

Parser = argparse.ArgumentParser(description='Sample code that uses the ptTools framework for training a UNet/SegNet autoencoder on Camera dataset.')
InputGroup = Parser.add_mutually_exclusive_group()
InputGroup.add_argument('--mode', help='Operation mode.', choices=['train', 'test'])
InputGroup.add_argument('--test-samples', help='Number of samples to use during testing.', default=30, type=int)

Loss = GenericImageDataset.GenericImageDataset.L2MaskLoss()

if __name__ == '__main__':
    Args, _ = Parser.parse_known_args()
    if len(sys.argv) <= 1:
        Parser.print_help()
        exit()

    if Args.mode == 'train':
        Net = UNet.UNet(in_shape=(3, 240, 320))

        TrainDevice = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        TrainData = CameraDataset.CameraDataset(root=Net.Config.Args.input_dir, train=True, download=True)
        print('[ INFO ]: Data has', len(TrainData), 'samples.')
        TrainDataLoader = torch.utils.data.DataLoader(TrainData, batch_size=Net.Config.Args.batch_size, shuffle=True, num_workers=4)

        # Train
        Net.fit(TrainDataLoader, Objective=Loss, TrainDevice=TrainDevice)
    elif Args.mode == 'test':
        Net = UNet.UNet()
        Net.loadCheckpoint()
        print(Net)

        TestDevice = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        TestData = CameraDataset.CameraDataset(root=Net.Config.Args.input_dir, train=False, download=True)
        print('[ INFO ]: Data has', len(TestData), 'samples.')

        # test(Args, TestData, Net, TestDevice)
