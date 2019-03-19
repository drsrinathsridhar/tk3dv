import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import numpy as np
import matplotlib.pyplot as plt

import sys, os, argparse

sys.path.append(os.path.join(FileDirPath, './models'))

import ptUtils, ptNets
import CAE

# Make both input and target be the same
class MNISTSpecialDataset(MNIST):
    def __getitem__(self, idx):
        Image, Label = super().__getitem__(idx)
        return Image, Image

def test(Args, TestData, Net, TestDevice):
    TestNet = Net.to(TestDevice)
    nSamples = min(Args.test_samples, len(TestData))
    print('[ INFO ]: Testing on', nSamples, 'samples')

    for i in range(nSamples):
        Image, _ = TestData[i]
        Image = Image.to(TestDevice)
        PredImage = TestNet(Image.unsqueeze_(0)).detach()
        plt.subplot(2, 1, 1)
        plt.imshow(Image.cpu().numpy().squeeze(), cmap='gray')
        plt.subplot(2, 1, 2)
        plt.imshow(PredImage.cpu().numpy().squeeze(), cmap='gray')
        plt.pause(1)


Parser = argparse.ArgumentParser(description='Sample code that uses the ptTools framework for training a simple autoencoder on MNIST.')
InputGroup = Parser.add_mutually_exclusive_group()
InputGroup.add_argument('--mode', help='Operation mode.', choices=['train', 'test'])
InputGroup.add_argument('--test-samples', help='Number of samples to use during testing.', default=30, type=int)

MNISTCAETrans = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (1.0,))
                                     ])

if __name__ == '__main__':
    Args, _ = Parser.parse_known_args()
    if len(sys.argv) <= 1:
        Parser.print_help()
        exit()

    if Args.mode == 'train':
        SampleNet = CAE.SimpleCAE()
        print(SampleNet)

        TrainDevice = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        TrainData = MNISTSpecialDataset(root=SampleNet.Args.input_dir, train=True, download=True, transform=MNISTCAETrans)
        print('[ INFO ]: Data has', len(TrainData), 'samples.')
        TrainDataLoader = torch.utils.data.DataLoader(TrainData, batch_size=SampleNet.Args.batch_size, shuffle=True, num_workers=4)

        # Train
        SampleNet.train(TrainDataLoader, Objective=nn.MSELoss(), TrainDevice=TrainDevice)
    elif Args.mode == 'test':
        SampleNet = CAE.SimpleCAE()
        SampleNet.loadCheckpoint()
        print(SampleNet)

        TestDevice = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        TestData = MNISTSpecialDataset(root=SampleNet.Args.input_dir, train=False, download=True, transform=MNISTCAETrans)
        print('[ INFO ]: Data has', len(TestData), 'samples.')

        test(Args, TestData, SampleNet, TestDevice)
