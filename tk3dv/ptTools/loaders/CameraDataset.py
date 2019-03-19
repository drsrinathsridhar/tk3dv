import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os, sys, math, argparse, zipfile, glob, cv2, random

FileDirPath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(FileDirPath, '..'))
sys.path.append(os.path.join(FileDirPath, '.'))
import ptUtils

class CameraDataset(torch.utils.data.Dataset):
    def __init__(self, root, train=True, download=True, transform=None, target_transform=None, trialrun=False, imgSize=(640, 480), limit=None, loadMemory=False):
        self.FileName = 'camera_dataset_v1.zip'
        self.DataURL = 'https://storage.googleapis.com/stanford_share/Datasets/camera_dataset_v1.zip'

        self.init(root, train, download, transform, target_transform, trialrun, imgSize, limit, loadMemory)
        self.loadData()

    def init(self, root, train=True, download=True, transform=None, target_transform=None, trialrun=False, imgSize=(640, 480), limit=None, loadMemory=False):
        self.DataDir = root
        self.isTrainData = train
        self.isDownload = download
        self.Transform = transform
        self.TargetTransform = target_transform
        self.TrialRun = trialrun
        self.ImageSize = imgSize
        self.DataLimit = limit
        self.LoadMemory = loadMemory

        self.Images = None
        self.Angles = None
        self.Predictions = None

    def loadData(self):
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

        self.RGBList = (glob.glob(FilesPath + '/*_VertexColors.png'))
        self.RGBList.sort()
        self.InstMaskList = (glob.glob(FilesPath + '/*_InstanceMask.png'))
        self.InstMaskList.sort()
        self.NOCSList = (glob.glob(FilesPath + '/*_NOCS.png'))
        self.NOCSList.sort()

        if self.RGBList is None or self.InstMaskList is None or self.NOCSList is None:
            raise RuntimeError('[ ERR ]: No files found during data loading.')

        if len(self.RGBList) != len(self.InstMaskList) or len(self.InstMaskList) != len(self.NOCSList):
            raise RuntimeError('[ ERR ]: Data corrupted. Sizes do not match')

        DatasetLength = self.DataLimit
        self.RGBList = self.RGBList[:DatasetLength]
        self.InstMaskList = self.InstMaskList[:DatasetLength]
        self.NOCSList = self.NOCSList[:DatasetLength]

        self.RGBs = []
        self.InstMasks = []
        self.NOCSs = []
        if self.LoadMemory:
            print('[ INFO ]: Loading all images to memory.')
            for RGBFile, InstMaskFile, NOCSFile in zip(self.RGBList, self.InstMaskList, self.NOCSList):
                RGB, InstMask, NOCS = self.loadImages(RGBFile, InstMaskFile, NOCSFile)
                self.RGBs.append(RGB)
                self.InstMasks.append(InstMask)
                self.NOCSs.append(NOCS)

    def __len__(self):
        return len(self.RGBList)

    @staticmethod
    def imread_rgb_torch(Path, Size=None): # Use only for loading RGB images
        ImageCV = cv2.imread(Path, -1)
        # Discard 4th channel since we are loading as RGB
        if ImageCV.shape[-1] != 3:
            ImageCV = ImageCV[:, :, :3]
        if Size is not None:
            ImageCV = cv2.resize(ImageCV, dsize=Size, interpolation=cv2.INTER_NEAREST) # Avoid aliasing artifacts
        Image = ptUtils.np2torch(ImageCV)
        return Image

    def loadImages(self, RGBFile, InstMaskFile, NOCSFile):
        RGB = self.imread_rgb_torch(RGBFile)
        InstMask = self.imread_rgb_torch(InstMaskFile)
        NOCS = self.imread_rgb_torch(NOCSFile)
        RGB, InstMask, NOCS = self.transform(RGB, InstMask, NOCS)

        return RGB, InstMask, NOCS

    def transform(self, RGB, InstMask, NOCS):
        if self.Transform is not None:
            RGB = self.Transform(RGB)
        if self.TargetTransform is not None:
            NOCS = self.TargetTransform(NOCS)

        return RGB, InstMask, NOCS

    def __getitem__(self, idx):
        if self.LoadMemory == False:
            RGB, InstMask, NOCS = self.loadImages(self.RGBList[idx], self.InstMaskList[idx], self.NOCSList[idx])
        else:
            RGB, InstMask, NOCS = self.RGBs[idx], self.InstMasks[idx], self.NOCSs[idx]

        return RGB, NOCS.type(torch.FloatTensor)#InstMask#, NOCS

    def visualizeRandom(self, nSamples=10):
        nCols = 2
        nRows = min(nSamples, 10)

        RandIdx = random.sample(range(1, len(self)), nRows)

        fig, ax = plt.subplots(nrows=nRows, ncols=nCols)
        Ctr = 0
        for row in ax:
            Data = self[RandIdx[Ctr]]
            for ImCtr, col in enumerate(row):
                TempData = Data[ImCtr]
                if ImCtr == 2:
                    TempData = ptUtils.np2torch(ptUtils.colorizeInstanceMask(ptUtils.torch2np(Data[ImCtr])))
                DispIm = ptUtils.torch2np(TempData).squeeze() / 255
                if len(DispIm.shape) == 2 or DispIm.shape[-1] == 1:
                    col.imshow(DispIm, cmap='gray')
                else:
                    col.imshow(DispIm)

            Ctr = Ctr + 1

        plt.show()

Parser = argparse.ArgumentParser()
Parser.add_argument('-d', '--DataDir', help='Specify the location of the directory to download and store CameraDataset', required=True)

if __name__ == '__main__':
    Args = Parser.parse_args()

    NormalizeTrans = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0., 0., 0.), (1., 1., 1.))]
    )
    CameraData = CameraDataset(root=Args.DataDir, train=False, download=True)#, transform=NormalizeTrans)
    CameraData.visualizeRandom(2)
