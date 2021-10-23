import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

import sys, os, argparse
import cv2, math
import numpy as np

FileDirPath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(FileDirPath, '../models'))
sys.path.append(os.path.join(FileDirPath, '..'))

from tk3dv.ptTools.models import SegNet
from tk3dv.ptTools.loaders import GenericImageDataset
import tk3dv.ptTools.ptUtils as ptUtils

Parser = argparse.ArgumentParser(description='Sample code that uses the ptTools framework for training a SegNet autoencoder on Camera dataset.')
InputGroup = Parser.add_mutually_exclusive_group()
InputGroup.add_argument('--mode', help='Operation mode.', choices=['train', 'val'])
InputGroup.add_argument('--val-samples', help='Number of samples to use during validation.', default=30, type=int)

Loss = GenericImageDataset.GenericImageDataset.L2MaskLoss()

def validate(Args, LossFunc, TestDataLoader, Net, TestDevice, OutDir=None):
    TestNet = Net.to(TestDevice)
    nSamples = min(Args.val_samples, len(TestDataLoader))
    print('Testing on ' + str(nSamples) + ' samples')

    if os.path.exists(OutDir) == False:
        os.makedirs(OutDir)

    ValLosses = []
    Tic = ptUtils.getCurrentEpochTime()
    for i, (Data, Targets) in enumerate(TestDataLoader, 0):  # Get each batch
        if i > (nSamples - 1): break
        DataTD = ptUtils.sendToDevice(Data, TestDevice)
        TargetsTD = ptUtils.sendToDevice(Targets[0].unsqueeze(0), TestDevice)

        Output = TestNet.forward(DataTD)
        Loss = LossFunc(Output, TargetsTD)
        ValLosses.append(Loss.item())

        if OutDir is not None:
            InputIm, GTOutTupRGB, GTOutTupMask = GenericImageDataset.GenericImageDataset.convertData(ptUtils.sendToDevice(Data, 'cpu'),
                                                                              ptUtils.sendToDevice(Targets[0], 'cpu'))
            _, PredOutTupRGB, PredOutTupMask = GenericImageDataset.GenericImageDataset.convertData(ptUtils.sendToDevice(Data, 'cpu'),
                                                                            ptUtils.sendToDevice(Output.detach(),
                                                                                                 'cpu'), isMaskNOX=True)
            cv2.imwrite(os.path.join(OutDir, 'frame_{}_color.png').format(str(i).zfill(3)),
                        cv2.cvtColor(InputIm, cv2.COLOR_BGR2RGB))

            OutTargetStr = ['nocs']

            for Targetstr, GT, Pred, GTMask, PredMask in zip(OutTargetStr, GTOutTupRGB, PredOutTupRGB, GTOutTupMask,
                                                             PredOutTupMask):
                cv2.imwrite(os.path.join(OutDir, 'frame_{}_{}_00gt.png').format(str(i).zfill(3), Targetstr),
                            cv2.cvtColor(GT, cv2.COLOR_BGR2RGB))
                cv2.imwrite(os.path.join(OutDir, 'frame_{}_{}_01pred.png').format(str(i).zfill(3), Targetstr),
                            cv2.cvtColor(Pred, cv2.COLOR_BGR2RGB))
                cv2.imwrite(os.path.join(OutDir, 'frame_{}_{}_02gtmask.png').format(str(i).zfill(3), Targetstr),
                            GTMask)
                cv2.imwrite(os.path.join(OutDir, 'frame_{}_{}_03predmask.png').format(str(i).zfill(3), Targetstr),
                            PredMask)

        # Print stats
        Toc = ptUtils.getCurrentEpochTime()
        Elapsed = math.floor((Toc - Tic) * 1e-6)
        done = int(50 * (i + 1) / len(TestDataLoader))
        sys.stdout.write(('\r[{}>{}] val loss - {:.16f}, elapsed - {}')
                         .format('=' * done, '-' * (50 - done), np.mean(np.asarray(ValLosses)),
                                 ptUtils.getTimeDur(Elapsed)))
        sys.stdout.flush()
    sys.stdout.write('\n')

    return ValLosses

if __name__ == '__main__':
    Args, _ = Parser.parse_known_args()
    if len(sys.argv) <= 1:
        Parser.print_help()
        exit()

    Net = SegNet.SegNet(n_classes=4, in_channels=3, Args=None, DataParallelDevs=None, withSkipConnections=False)

    if Args.mode == 'train':
        TrainDevice = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        TrainData = GenericImageDataset.GenericImageDataset(root=Net.Config.Args.input_dir, train=True, download=True, imgSize=(320, 240))
        print('[ INFO ]: Data has', len(TrainData), 'samples.')
        TrainDataLoader = torch.utils.data.DataLoader(TrainData, batch_size=Net.Config.Args.batch_size, shuffle=True, num_workers=4)

        # Train
        Net.fit(TrainDataLoader, Objective=Loss, TrainDevice=TrainDevice)
    elif Args.mode == 'val':
        Net.loadCheckpoint()

        ValDevice = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        ValData = GenericImageDataset.GenericImageDataset(root=Net.Config.Args.input_dir, train=False, download=True, imgSize=(320, 240))
        print('[ INFO ]: Data has', len(ValData), 'samples.')
        ValDataLoader = torch.utils.data.DataLoader(ValData, batch_size=1, shuffle=True,
                                                      num_workers=1)

        validate(Args, Loss, ValDataLoader, Net, ValDevice, OutDir=os.path.join(Net.ExptDirPath, 'ValResults'))