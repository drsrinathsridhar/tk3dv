import torch.utils.data
import numpy as np
import torch.nn.functional as F
import json
import matplotlib.pyplot as plt
import os, sys, argparse, zipfile, glob, random, pickle, cv2
from itertools import groupby

FileDirPath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(FileDirPath, '..'))
import ptUtils
from torch import nn

# This is the basic loader that loads all data without any model ID separation of camera viewpoint knowledge
class GenericImageDataset(torch.utils.data.Dataset):
    class L2MaskLoss(nn.Module):
        Thresh = 0.7 # PARAM
        def __init__(self, Thresh=0.7, MaskWeight=0.7, ImWeight=0.3): # PARAM
            super().__init__()
            self.MaskLoss = nn.BCELoss(size_average=True, reduce=True)
            self.Sigmoid = nn.Sigmoid()
            self.Thresh = Thresh
            self.MaskWeight = MaskWeight
            self.ImWeight = ImWeight

        def forward(self, output, target):
            return self.computeLoss(output, target)

        def computeLoss(self, output, target):
            OutIm = output
            nChannels = OutIm.size(1)
            TargetIm = target[0]
            if nChannels%4 != 0:
                raise RuntimeError('Empty or mismatched batch (should be multiple of 4). Check input.')
            if nChannels != TargetIm.size(1):
                raise RuntimeError('Out target {} size mismatch with nChannels {}. Check input.'.format(TargetIm.size(1), nChannels))

            BatchSize = OutIm.size(0)
            TotalLoss = 0
            Den = 0
            nOutIms = int(nChannels / 4)
            for i in range(0, nOutIms):
                Range = list(range(4*i, 4*(i+1)))
                TotalLoss += self.computeMaskedL2Loss(OutIm[:, Range, :, :], TargetIm[:, Range, :, :])
                Den += 1

            TotalLoss /= float(Den)

            return TotalLoss

        def computeMaskedL2Loss(self, output, target):
            BatchSize = target.size(0)
            TargetMask = target[:, -1, :, :]
            OutMask = output[:, -1, :, :].clone().requires_grad_(True)
            OutMask = self.Sigmoid(OutMask)

            MaskLoss = self.MaskLoss(OutMask, TargetMask)

            TargetIm = target[:, :-1, :, :].detach()
            OutIm = output[:, :-1, :, :].clone().requires_grad_(True)

            DiffNorm = torch.norm(OutIm - TargetIm, dim=1)  # Same size as WxH
            MaskedDiffNorm = torch.where(OutMask > self.Thresh, DiffNorm,
                                         torch.zeros(DiffNorm.size(), device=DiffNorm.device))
            NOCSLoss = 0
            for i in range(0, BatchSize):
                nNonZero = torch.nonzero(MaskedDiffNorm[i]).size(0)
                if nNonZero > 0:
                    NOCSLoss += torch.sum(MaskedDiffNorm[i]) / nNonZero
                else:
                    NOCSLoss += torch.mean(DiffNorm[i])

            Loss = (self.MaskWeight*MaskLoss) + (self.ImWeight*(NOCSLoss / BatchSize))
            return Loss

    @staticmethod
    def imread_rgb_torch(Path, Size=None): # Use only for loading RGB images
        ImageCV = cv2.imread(Path, -1)
        # Discard 4th channel since we are loading as RGB
        if ImageCV.shape[-1] != 3:
            ImageCV = ImageCV[:, :, :3]

        ImageCV = cv2.cvtColor(ImageCV, cv2.COLOR_BGR2RGB)
        if Size is not None:
            ImageCV = cv2.resize(ImageCV, dsize=Size, interpolation=cv2.INTER_NEAREST)
        Image = ptUtils.np2torch(ImageCV) # Range: 0-255

        return Image

    @staticmethod
    def saveData(Items, OutPath='.'):
        for Ctr, I in enumerate(Items, 0):
            if len(I.shape) == 3:
                I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(OutPath, 'item' + str(Ctr).zfill(4) + '.png'), I)

    @staticmethod
    def createMask(NOCSMap):
        LocNOCS = NOCSMap.type(torch.FloatTensor)

        Norm = torch.norm(LocNOCS, dim=0)
        ZeroT = torch.zeros((1, LocNOCS.size(1), LocNOCS.size(2)), dtype=LocNOCS.dtype, device=LocNOCS.device)
        OneT = torch.ones((1, LocNOCS.size(1), LocNOCS.size(2)), dtype=LocNOCS.dtype, device=LocNOCS.device) * 255 # Range: 0, 255
        Mask = torch.where(Norm >= 441.6729, ZeroT, OneT)  # 441.6729 == sqrt(3 * 255^2)
        return Mask.type(torch.FloatTensor)

    @staticmethod
    def applyMask(NOX, Thresh):
        # Input (only torch): 4-channels where first 3 are NOCS, last is mask
        # Output (numpy): 3-channels where the mask is used to mask out the NOCS
        if NOX.size()[0] != 4:
            raise RuntimeError('[ ERR ]: Input needs to be a 4 channel image.')

        NOCS = NOX[:3, :, :]
        Mask = NOX[3, :, :]

        MaskProb = ptUtils.torch2np(torch.squeeze(F.sigmoid(Mask)))
        Masked = np.uint8((MaskProb > Thresh) * 255)
        MaskedNOCS = ptUtils.torch2np(torch.squeeze(NOCS))
        MaskedNOCS[MaskProb <= Thresh] = 255

        return MaskedNOCS, Masked

    def __init__(self, root, train=True, download=True, transform=None, target_transform=None, imgSize=(640, 480), limit=None, loadMemory=False, loadLevel='all', FrameLoadStr=None, Required='VertexColors'):
        self.DataDir = root
        self.isTrainData = train
        self.isDownload = download
        self.Transform = transform
        self.TargetTransform = target_transform
        self.ImageSize = imgSize
        self.LoadMemory = loadMemory

        self.FileName = 'camera_dataset_v1.zip'
        self.DataURL = 'https://storage.googleapis.com/stanford_share/Datasets/camera_dataset_v1.zip'
        self.FrameLoadStr = ['VertexColors', 'NOCS'] if FrameLoadStr is None else FrameLoadStr
        self.Required = Required
        self.LoadType = loadLevel.lower() # 0 - All, 1 - GridSearch, 2 - Debug
        self.LoadLevel = {}
        self.LoadLevel['all'] = 1000
        self.LoadLevel['gridsearch'] = 100
        self.LoadLevel['debug'] = 10
        self.DataLimit = self.LoadLevel[self.LoadType] if limit is None else limit # limit takes precedence

        if self.Required not in self.FrameLoadStr:
            raise RuntimeError('FrameLoadStr should contain {}.'.format(self.Required))

        print('[ INFO ]: Loading dataset at level {} ({} samples).'.format(self.LoadType, self.LoadLevel[self.LoadType]))

        self.loadData()

    def loadData(self):
        self.FrameFiles = {}
        # First check if unzipped directory exists
        DatasetDir = os.path.join(ptUtils.expandTilde(self.DataDir), os.path.splitext(self.FileName)[0])
        if os.path.exists(DatasetDir) == False:
            DataPath = os.path.join(ptUtils.expandTilde(self.DataDir), self.FileName)
            if os.path.exists(DataPath) == False:
                if self.isDownload:
                    print('[ INFO ]: Downloading', DataPath)
                    ptUtils.downloadFile(self.DataURL, DataPath)

                if os.path.exists(DataPath) == False: # Not downloaded
                    raise RuntimeError('Specified data path does not exist: ' + DataPath)
            # Unzip
            with zipfile.ZipFile(DataPath, 'r') as File2Unzip:
                print('[ INFO ]: Unzipping.')
                File2Unzip.extractall(ptUtils.expandTilde(self.DataDir))

        FilesPath = os.path.join(DatasetDir, 'val/')
        if self.isTrainData:
            FilesPath = os.path.join(DatasetDir, 'train/')

        GlobPrepend = '_'.join(str(i) for i in self.FrameLoadStr)
        GlobCache = os.path.join(DatasetDir, 'glob_' + GlobPrepend + '.cache')

        if os.path.exists(GlobCache):
            print('[ INFO ]: Loading from glob cache:', GlobCache)
            with open(GlobCache, 'rb') as fp:
                for Str in self.FrameLoadStr:
                    self.FrameFiles[Str] = pickle.load(fp)
        else:
            print('[ INFO ]: Saving to glob cache:', GlobCache)

            for Str in self.FrameLoadStr:
                print(os.path.join(FilesPath, '*' + Str + '*.*'))
                self.FrameFiles[Str] = glob.glob(os.path.join(FilesPath, '*' + Str + '*.*'))
                self.FrameFiles[Str].sort()

            with open(GlobCache, 'wb') as fp:
                for Str in self.FrameLoadStr:
                    pickle.dump(self.FrameFiles[Str], fp)

        FrameFilesLengths = []
        for K, CurrFrameFiles in self.FrameFiles.items():
            if not CurrFrameFiles:
                raise RuntimeError('None data for {}'.format(K))
            if len(CurrFrameFiles) == 0:
                raise RuntimeError('No files found during data loading for', K)
            FrameFilesLengths.append(len(CurrFrameFiles))

        if len(set(FrameFilesLengths)) != 1:
            raise RuntimeError('Data corrupted. Sizes do not match', FrameFilesLengths)

        print('[ INFO ]: Found {} items in dataset.'.format(len(self)))
        DatasetLength = self.DataLimit

        if DatasetLength is None:
            print('[ INFO ]: Loading all items.')
        else:
            print('[ INFO ]: Loading only {} items.'.format(DatasetLength))
        for K in self.FrameFiles:
            self.FrameFiles[K] = self.FrameFiles[K][:DatasetLength]

    def __len__(self):
        return len(self.FrameFiles[self.FrameLoadStr[0]])

    def __getitem__(self, idx):
        RGB, LoadTup = self.loadImages(idx)
        LoadIms = torch.cat(LoadTup, 0)
        return RGB, (LoadIms,)

    def loadImages(self, idx):
        Frame = {}

        for K in self.FrameFiles:
            Frame[K] = self.imread_rgb_torch(self.FrameFiles[K][idx], Size=self.ImageSize).type(torch.FloatTensor)
            if K not in self.Required:
                Frame[K] = torch.cat((Frame[K], self.createMask(Frame[K])), 0).type(torch.FloatTensor)
            if self.Transform is not None:
                Frame[K] = self.Transform(Frame[K])

            # Convert range to 0.0 - 1.0
            Frame[K] /= 255.0

        GroupedFrameStr = [list(i) for j, i in groupby(self.FrameLoadStr, lambda a: ''.join([i for i in a if not i.isdigit()]))]
        # print(self.FrameLoadStr)
        # print(GroupedFrameStr)

        LoadTup = ()
        # Concatenate any peeled outputs
        for Group in GroupedFrameStr:
            # print(Group)
            Concated = ()
            for FrameStr in Group:
                # print(FrameStr)
                if self.Required in FrameStr: # Append manually
                    continue
                Concated = Concated + (Frame[FrameStr],)
            if len(Concated) > 0:
                LoadTup = LoadTup + (torch.cat(Concated, 0), )

        # print(len(LoadTup))
        # print(LoadTup[0].size())

        return Frame[self.Required], LoadTup

    def convertItem(self, idx, isMaskNOX=False):
        RGB, LoadTup = self.loadImages(idx)
        # RGB, Targets = self[idx]

        return self.convertData(RGB, LoadTup, isMaskNOX=isMaskNOX)

    @staticmethod
    def convertData(RGB, Targets, isMaskNOX=False):
        TargetsTup = Targets
        Color00 = ptUtils.torch2np(RGB[0:3].squeeze()).squeeze() * 255

        OutTupRGB = ()
        OutTupMask = ()
        # Convert range to 0-255
        for T in TargetsTup:
            T = T.squeeze() * 255
            nChannels = T.size(0)
            if nChannels != 4 and nChannels != 8:
                raise RuntimeError('Only supports 4/8-channel input. Passed {}'.format(nChannels))

            if nChannels == 4:
                if isMaskNOX:
                    MaskedT, _ = GenericImageDataset.applyMask(torch.squeeze(T[0:4]), Thresh=GenericImageDataset.L2MaskLoss.Thresh)
                    OutTupRGB = OutTupRGB + (MaskedT,)
                else:
                    OutTupRGB = OutTupRGB + (ptUtils.torch2np(T[0:3]).squeeze(),)
                OutTupMask = OutTupMask + (ptUtils.torch2np(T[3]).squeeze(),)
            else:
                if isMaskNOX:
                    MaskedT0, _ = GenericImageDataset.applyMask(torch.squeeze(T[0:4]), Thresh=GenericImageDataset.L2MaskLoss.Thresh)
                    MaskedT1, _ = GenericImageDataset.applyMask(torch.squeeze(T[4:8]), Thresh=GenericImageDataset.L2MaskLoss.Thresh)
                    OutTupRGB = OutTupRGB + (MaskedT0, MaskedT1)
                else:
                    OutTupRGB = OutTupRGB + (ptUtils.torch2np(T[0:3]).squeeze(), ptUtils.torch2np(T[4:7]).squeeze())
                OutTupMask = OutTupMask + (ptUtils.torch2np(T[3]).squeeze(), ptUtils.torch2np(T[7]).squeeze())

        return Color00, OutTupRGB, OutTupMask

    def saveItem(self, idx, OutPath='.'):
        Color00, OutTupRGB, OutTupMask = Data.convertItem(idx, isMaskNOX=True)
        GenericImageDataset.saveData((Color00,) + OutTupRGB, OutPath)

    def visualizeRandom(self, nSamples=10):
        nColsPerSample = len(self.FrameLoadStr)
        nCols = nColsPerSample
        nRows = min(nSamples, 10)
        nTot = nRows * nCols

        RandIdx = random.sample(range(0, len(self)), nRows)

        fig = plt.figure(0, figsize=(2, 10))
        for RowCtr, RandId in enumerate(RandIdx):
            Color00, OutTupRGB, OutTupMask = self.convertItem(RandId)

            DivFact = 1
            if np.max(Color00) > 1:
                DivFact = 255

            ax = fig.add_subplot(nRows, nCols, RowCtr*nCols + 1)
            if RowCtr == 0:
                ax.title.set_text(self.FrameLoadStr[0])
            plt.xticks([]), plt.yticks([]), plt.grid(False)
            plt.imshow(Color00 / DivFact)

            Ctr = 2
            for Out in OutTupRGB:
                ax = fig.add_subplot(nRows, nCols, RowCtr*nCols + Ctr)
                if RowCtr == 0:
                    ax.title.set_text(self.FrameLoadStr[Ctr - 1 ])
                plt.xticks([]), plt.yticks([]), plt.grid(False)
                plt.imshow(Out / DivFact)
                Ctr = Ctr + 1
        plt.show()

Parser = argparse.ArgumentParser()
Parser.add_argument('-d', '--data-dir', help='Specify the location of the directory to download and store the dataset', required=True)

if __name__ == '__main__':
    Args, _ = Parser.parse_known_args()

    Data = GenericImageDataset(root=Args.data_dir, train=True, download=True, imgSize=(320, 240), loadLevel='all')#, FrameLoadStr=['color00', 'normals00', 'nox00', 'pnnocs00', 'uv00', 'camera'])
    # Data.saveItem(random.randint(0, len(Data)))
    Data.visualizeRandom(10)
    # exit()

    LossUnitTest = GenericImageDataset.L2MaskLoss(0.7)
    DataLoader = torch.utils.data.DataLoader(Data, batch_size=4, shuffle=True, num_workers=1)
    for i, (Data, Targets) in enumerate(DataLoader, 0):  # Get each batch
        # DataTD = ptUtils.sendToDevice(Targets, 'cpu')
        # print('Data size:', Data.size())
        # print('Targets size:', len(Targets))
        Loss = LossUnitTest(Targets[0], Targets)
        print('Loss:', Loss.item())
        # Loss.backward()
