import requests, sys, os, glob, argparse, random
from datetime import datetime, timedelta
import torch
import numpy as np
from palettable.tableau import Tableau_20, BlueRed_12, ColorBlind_10, GreenOrange_12
from palettable.cartocolors.diverging import Earth_2
import matplotlib.pyplot as plt
import logging

# ptToolsLogger = logging.getLogger(__name__)
# ptToolsLogger.propagate = False
# # LogFormat = logging.Formatter('[ %(asctime)s ] [ {} ] [ %(levelname)-5.5s ]: %(message)s'.format(__name__))
# LogFormat = logging.Formatter('[ %(levelname)-5.5s ]: %(message)s')
# EmptyFormat = logging.Formatter('%(message)s')
# StdoutHandler = logging.StreamHandler(sys.stdout)
# StdoutHandler.setFormatter(LogFormat)
# ptToolsLogger.addHandler(StdoutHandler)
# ptToolsLogger.setLevel(logging.DEBUG)

class ptLogger():
    def __init__(self, Stream=sys.stdout, OutFile=None):
        self.Terminal = Stream
        self.File = None
        if OutFile is not None:
            self.File = open(OutFile, 'w+')

    def addFile(self, OutFile):
        if OutFile is not None:
            self.File = open(OutFile, 'w+')

    def write(self, message):
        self.Terminal.write(message)
        if self.File is not None:
            self.File.write(message)

    def flush(self):
        self.Terminal.flush()
        if self.File is not None:
            self.File.flush()

# Utilities for PyTorch
def restricted_float(x):
    x = float(x)
    if x < -1.0 or x > 10.0:
        raise argparse.ArgumentTypeError('%r not in range [-1.0, 10.0]'%(x,))
    return x

def getCurrentEpochTime():
    return int((datetime.utcnow() - datetime(1970, 1, 1)).total_seconds() * 1e6)

def getZuluTimeString(StringFormat = '%Y-%m-%dT%H-%M-%S'):
    return datetime.utcnow().strftime(StringFormat)

def getLocalTimeString(StringFormat = '%Y-%m-%dT%H-%M-%S'):
    return datetime.now().strftime(StringFormat)

def getTimeString(TimeString='humanlocal'):
    TS = TimeString.lower()
    OTS = 'UNKNOWN'

    if 'epoch' in TS:
        OTS = str(getCurrentEpochTime())
    else:
        if 'zulu' in TS:
            OTS = getZuluTimeString(StringFormat='%Y-%m-%dT%H-%M-%SZ')
        elif 'local' in TS:
            OTS = getLocalTimeString(StringFormat='%Y-%m-%dT%H-%M-%S')
        elif 'eot' in TS: # End of time
            OTS = '9999-12-31T23-59-59'

    if 'human' in TS:
        OTS += '_' + str(getCurrentEpochTime())

    return OTS


def dhms(td):
    d, h, m = td.days, td.seconds//3600, (td.seconds//60)%60
    s = td.seconds - ( (h*3600) + (m*60) ) # td.seconds are the seconds remaining after days have been removed
    return d, h, m, s

def getTimeDur(seconds):
    Duration = timedelta(seconds=seconds)
    OutStr = ''
    d, h, m, s = dhms(Duration)
    if d > 0:
        OutStr = OutStr + str(d)+ ' d '
    if h > 0:
        OutStr = OutStr + str(h) + ' h '
    if m > 0:
        OutStr = OutStr + str(m) + ' m '
    OutStr = OutStr + str(s) + ' s'

    return OutStr

def downloadFile(url, filename):
    with open(expandTilde(filename), 'wb') as f:
        response = requests.get(url, stream=True)
        total = response.headers.get('content-length')

        if total is None:
            f.write(response.content)
        else:
            downloaded = 0
            total = int(total)
            for data in response.iter_content(chunk_size=max(int(total / 1000), 1024 * 1024)):
                downloaded += len(data)
                f.write(data)
                done = int(50 * downloaded / total)
                sys.stdout.write('\r[{}>{}]'.format('=' * done, '-' * (50 - done)))
                sys.stdout.flush()
    sys.stdout.write('\n')

def expandTilde(Path):
    if '~' == Path[0]:
        return os.path.expanduser(Path)

    return Path

def savePyTorchCheckpoint(CheckpointDict, OutDir, TimeString='humanlocal'):
    # CheckpointDict should have a model name, otherwise, will store as UNKNOWN
    Name = 'UNKNOWN'
    if 'Name' in CheckpointDict:
        Name = CheckpointDict['Name']

    OTS = getTimeString(TimeString)
    OutFilePath = os.path.join(expandTilde(OutDir), Name + '_' + OTS + '.tar')
    torch.save(CheckpointDict, OutFilePath)
    return OutFilePath

def loadPyTorchCheckpoint(InPath, map_location='cpu'):
    return torch.load(expandTilde(InPath), map_location=map_location)

def loadLatestPyTorchCheckpoint(InDir, CheckpointName='', map_location='cpu'):
    AllCheckpoints = glob.glob(os.path.join(expandTilde(InDir), CheckpointName + '*.tar'))
    if len(AllCheckpoints) <= 0:
        raise RuntimeError('No checkpoints stored in ' + InDir)
    AllCheckpoints.sort() # By name

    print('[ INFO ]: Loading checkpoint {}'.format(AllCheckpoints[-1]))
    return loadPyTorchCheckpoint(AllCheckpoints[-1])

def torch2np(ImageTorch):
    # OpenCV does [height, width, channels]
    # PyTorch stores images as [channels, height, width]
    if ImageTorch.size()[0] == 3:
            return np.transpose(ImageTorch.numpy(), (1, 2, 0))
    return ImageTorch.numpy()

def np2torch(ImageNP):
    # PyTorch stores images as [channels, height, width]
    # OpenCV does [height, width, channels]
    if ImageNP.shape[-1] == 3:
            return torch.from_numpy(np.transpose(ImageNP, (2, 0, 1)))
    return torch.from_numpy(ImageNP)

def colorizeInstanceMask(InstanceMask):
    InstanceMaskRGB = InstanceMask.copy()

    # Display each category with one color.
    # Note category is in the R channel
    UniqueCategories = np.unique(InstanceMaskRGB[:, :, 2])
    UniqueCategories = np.delete(UniqueCategories, np.argwhere(UniqueCategories == 255)).tolist()
    Palette = GreenOrange_12
    Ctr = 0  # random.randint(0, Earth_2.number-1)
    for CatID in UniqueCategories:
        Color = Palette.colors[Ctr].copy()
        if InstanceMaskRGB.shape[-1] == 4:
            Color.append(255)
        Instances = np.where(InstanceMaskRGB[:, :, 2] == CatID)
        InstanceMaskRGB[Instances] = Color
        Ctr = Ctr + 1
        Ctr = Ctr % Palette.number

    return InstanceMaskRGB

def saveLossesCurve(*args, **kwargs):
    plt.clf()
    ylim = 0
    for arg in args:
        if len(arg) <= 0:
            continue
        plt.plot(arg)
        ylim = ylim + np.mean(np.asarray(arg))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    if 'xlim' in kwargs:
        plt.xlim(kwargs['xlim'])
    if 'legend' in kwargs:
        if len(kwargs['legend']) > 0:
            plt.legend(kwargs['legend'])
    if 'title' in kwargs:
        plt.title(kwargs['title'])
    if ylim > 0:
        plt.ylim([0.0, ylim])
    if 'out_path' in kwargs:
        plt.savefig(kwargs['out_path'])
    else:
        print('[ WARN ]: No output path (out_path) specified. ptUtils.saveLossesCurve()')


class loadArgsFromFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        with values as f:
            parser.parse_known_args(f.read().split(), namespace)

def printArgs(Args):
    ArgsDict = vars(Args)
    for Arg in ArgsDict:
        if ArgsDict[Arg] is not None:
            if isinstance(ArgsDict[Arg], list):
                print('{:<15}:   {}'.format(Arg, ArgsDict[Arg]))
            else:
                print('{:<15}:   {:<50}'.format(Arg, ArgsDict[Arg]))

def seedRandom(seed):
    # NOTE: This gets us very close to deterministic but there are some small differences in 1e-4 and smaller scales
    print('[ INFO ]: Seeding RNGs with {}'.format(seed))
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def setDevice(RequestedGPUID):
    DeviceName ='cpu' # Default
    if RequestedGPUID >= -1:
        if torch.cuda.is_available() == False:
            print('[ WARN ]: No GPUs available. Using CPU.')
        else:
            if RequestedGPUID >= torch.cuda.device_count():
                print('[ WARN ]: GPU {} is unavailable. Using cuda:0.'.format(RequestedGPUID))
                DeviceName = 'cuda:0'
            elif RequestedGPUID == -1:
                DeviceName = 'cuda:0'
            else:
                DeviceName = 'cuda:' + str(RequestedGPUID)

    print('[ INFO ]: Using device: {}'.format(DeviceName))
    Device = torch.device(DeviceName)

    return Device

def configSerialize(Args, FilePath, isAppend=True):
    WriteMode = 'w'
    if isAppend:
        WriteMode = 'a'
    with open(FilePath, WriteMode) as File:
        ArgsDict = vars(Args)
        for Arg in ArgsDict:
            if type(ArgsDict[Arg]) is bool:
                if ArgsDict[Arg] == True: # Serialize only if true
                    File.write('--{}\n'.format(Arg.replace('_', '-')))
            elif type(ArgsDict[Arg]) is float:
                File.write('--{}={}\n'.format(Arg.replace('_', '-'), '{0:.6f}'.format(ArgsDict[Arg])))
            elif type(ArgsDict[Arg]) is list:
                ListStr = (', '.join(map(str, ArgsDict[Arg])))
                File.write('--{}={}\n'.format(Arg.replace('_', '-'), ListStr))
            else:
                File.write('--{}={}\n'.format(Arg.replace('_', '-'), ArgsDict[Arg]))

def makeDir(Path):
    if os.path.exists(Path) == False:
        os.makedirs(Path)

def setupGPUs(RequestedGPUList):
    if torch.cuda.is_available():
        DeviceList = RequestedGPUList
        MainGPUID = DeviceList[0]
        nDevs = torch.cuda.device_count()
        AvailableDevList = [i for i in range(0, nDevs)]  # All GPUs
        if len(DeviceList) == 1 and MainGPUID < 0:
            DeviceList = AvailableDevList
        if set(DeviceList).issubset(set(AvailableDevList)) == False:
            raise RuntimeError('Unable to find requested devices {}.'.format(DeviceList))
    else:
        DeviceList = [0]
        MainGPUID = 0

    return DeviceList, MainGPUID

def sendToDevice(TupleOrTensor, Device):
    TupleOrTensorTD = TupleOrTensor
    if isinstance(TupleOrTensorTD, tuple) == False and isinstance(TupleOrTensorTD, list) == False:
        TupleOrTensorTD = TupleOrTensor.to(Device)
    else:
        for Ctr in range(len(TupleOrTensor)):
            if isinstance(TupleOrTensor[Ctr], torch.Tensor):
                TupleOrTensorTD[Ctr] = TupleOrTensor[Ctr].to(Device)
            else:
                TupleOrTensorTD[Ctr] = TupleOrTensor[Ctr]

    return TupleOrTensorTD
