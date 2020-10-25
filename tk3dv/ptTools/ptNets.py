import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os, sys, argparse, math, glob, gc, traceback
import numpy as np

FileDirPath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(FileDirPath, '.'))

import ptUtils

def RestrictedFloat_N10_100(x):
    x = float(x)
    MinMax = [-10.0, 100.0]
    if x < MinMax[0] or x > MinMax[1]:
        raise argparse.ArgumentTypeError('{} not in range [{}, {}]'.format(x, MinMax[0], MinMax[1]))
    return x

class ptNetMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        return self.computeLoss(output, target)

    def computeLoss(self, output, target):
        loss = torch.mean((output - target) ** 2)
        return loss

class ptNetLoss(nn.Module):
    def __init__(self, Losses=[], Weights=[], Names=[]):
        super().__init__()
        if not Losses: # empty list
            self.Losses = [ptNetMSELoss()]
            self.Weights = [1.0]
            self.Names = ['Default MSE Loss']
        else:
            if len(Losses) != len(Weights):
                raise RuntimeError('ptNetLoss() given Losses and Weights don''t match.')

            self.Losses = Losses
            self.Weights = Weights
            self.Names = ['Subloss ' + str(i).zfill(2) for i in range(len(self.Losses))]
            for Ctr, n in enumerate(Names, 0):
                self.Names[Ctr] = n
            self.cleanUp()

    def __len__(self):
        return len(self.Losses)

    def getItems(self, withoutWeights=False):
        RetLossValsFloat = []
        if withoutWeights:
            for v in self.LossVals:
                RetLossValsFloat.append(v.item())
        else:
            for v in self.LossValsWeighted:
                RetLossValsFloat.append(v.item())

        return RetLossValsFloat

    def cleanUp(self):
        self.LossVals = [0.0] * len(self.Losses)
        self.LossValsWeighted = [0.0] * len(self.Losses)

    def forward(self, output, target):
        self.cleanUp()
        return self.computeLoss(output, target)

    def computeLoss(self, output, target):
        TotalLossVal = 0.0

        for Ctr, (l, w) in enumerate(zip(self.Losses, self.Weights), 0):
            LossVal = l.forward(output, target)
            self.LossVals[Ctr] = LossVal
            self.LossValsWeighted[Ctr] = w * LossVal
            TotalLossVal += self.LossValsWeighted[Ctr]

        return TotalLossVal

class ptNetExptConfig():
    def __init__(self, InputArgs=None, isPrint=True):
        self.Parser = argparse.ArgumentParser(description='Parse arguments for a PyTorch neural network.', fromfile_prefix_chars='@')

        # Search params
        self.Parser.add_argument('--learning-rate', help='Choose the learning rate.', required=False, default=0.001,
                            type=RestrictedFloat_N10_100)
        self.Parser.add_argument('--batch-size', help='Choose mini-batch size.', choices=range(1, 4096), metavar='1..4096',
                            required=False, default=128, type=int)

        # Machine-specific params
        self.Parser.add_argument('--expt-name', help='Provide a name for this experiment.')
        self.Parser.add_argument('--input-dir', help='Provide the input directory where datasets are stored.')
        # -----
        self.Parser.add_argument('--output-dir',
                            help='Provide the *absolute* output directory where checkpoints, logs, and other output will be stored (under expt_name).')
        self.Parser.add_argument('--rel-output-dir',
                            help='Provide the *relative* (pwd or config file) output directory where checkpoints, logs, and other output will be stored (under expt_name).')
        # -----
        self.Parser.add_argument('--epochs', help='Choose number of epochs.', choices=range(1, 10000), metavar='1..10000',
                            required=False, default=10, type=int)
        self.Parser.add_argument('--save-freq', help='Choose epoch frequency to save checkpoints. Zero (0) will only at the end of training [not recommended].', choices=range(0, 10000), metavar='0..10000',
                            required=False, default=5, type=int)

        self.Args, _ = self.Parser.parse_known_args(InputArgs)

        if self.Args.expt_name is None:
            raise RuntimeError('No experiment name (--expt-name) provided.')

        if self.Args.rel_output_dir is None and self.Args.output_dir is None:
            raise RuntimeError('One or both of --output-dir or --rel-output-dir is required.')

        if self.Args.rel_output_dir is not None: # Relative path takes precedence
            if self.Args.output_dir is not None:
                print('[ INFO ]: Relative path taking precedence to absolute path.')
            DirPath = os.getcwd() # os.path.dirname(os.path.realpath(__file__))
            for Arg in InputArgs:
                if '@' in Arg: # Config file is passed, path should be relative to config file
                    DirPath = os.path.abspath(os.path.dirname(ptUtils.expandTilde(Arg[1:]))) # Abs directory path of config file
                    break
            self.Args.output_dir = os.path.join(DirPath, self.Args.rel_output_dir)
            print('[ INFO ]: Converted relative path {} to absolute path {}'.format(self.Args.rel_output_dir, self.Args.output_dir))

        # Logging directory and file
        self.ExptDirPath = ''
        self.ExptDirPath = os.path.join(ptUtils.expandTilde(self.Args.output_dir), self.Args.expt_name)
        if os.path.exists(self.ExptDirPath) == False:
            os.makedirs(self.ExptDirPath)

        self.ExptLogFile = os.path.join(self.ExptDirPath, self.Args.expt_name + '_' + ptUtils.getTimeString('humanlocal') + '.log')
        # if os.path.exists(self.ExptLogFile) == False:
        with open(self.ExptLogFile, 'w+', newline='') as f:
            os.utime(self.ExptLogFile, None)

        sys.stdout = ptUtils.ptLogger(sys.stdout, self.ExptLogFile)
        sys.stderr = ptUtils.ptLogger(sys.stderr, self.ExptLogFile)

        if isPrint:
            print('-'*60)
            ArgsDict = vars(self.Args)
            for Arg in ArgsDict:
                if ArgsDict[Arg] is not None:
                    print('{:<15}:   {:<50}'.format(Arg, ArgsDict[Arg]))
                else:
                    print('{:<15}:   {:<50}'.format(Arg, 'NOT DEFINED'))
            print('-'*60)

    def getHelp(self):
        self.Parser.print_help()

    def serialize(self, FilePath, isAppend=True):
        ptUtils.configSerialize(self.Args, FilePath, isAppend)

class ptNet(nn.Module):
    def __init__(self, Args=None):
        super().__init__()

        self.Config = ptNetExptConfig(InputArgs=Args)

        # Defaults
        self.StartEpoch = 0
        self.ExptDirPath = self.Config.ExptDirPath
        self.SaveFrequency = self.Config.Args.save_freq if self.Config.Args.save_freq > 0 else self.Config.Args.epochs
        self.LossHistory = []
        self.ValLossHistory = []
        self.SeparateLossesHistory = []

    def loadCheckpoint(self, Path=None, Device='cpu'):
        if Path is None:
            self.ExptDirPath = os.path.join(ptUtils.expandTilde(self.Config.Args.output_dir), self.Config.Args.expt_name)
            print('[ INFO ]: Loading from latest checkpoint.')
            CheckpointDict = ptUtils.loadLatestPyTorchCheckpoint(self.ExptDirPath, map_location=Device)
        else: # Load latest
            print('[ INFO ]: Loading from checkpoint {}'.format(Path))
            CheckpointDict = ptUtils.loadPyTorchCheckpoint(Path)

        self.load_state_dict(CheckpointDict['ModelStateDict'])

    def setupCheckpoint(self, TrainDevice):
        LatestCheckpointDict = None
        AllCheckpoints = glob.glob(os.path.join(self.ExptDirPath, '*.tar'))
        if len(AllCheckpoints) > 0:
            LatestCheckpointDict = ptUtils.loadLatestPyTorchCheckpoint(self.ExptDirPath, map_location=TrainDevice)
            print('[ INFO ]: Loading from last checkpoint.')

        if LatestCheckpointDict is not None:
            # Make sure experiment names match
            if self.Config.Args.expt_name == LatestCheckpointDict['Name']:
                self.load_state_dict(LatestCheckpointDict['ModelStateDict'])
                self.StartEpoch = LatestCheckpointDict['Epoch']
                self.Optimizer.load_state_dict(LatestCheckpointDict['OptimizerStateDict'])
                self.LossHistory = LatestCheckpointDict['LossHistory']
                if 'ValLossHistory' in LatestCheckpointDict:
                    self.ValLossHistory = LatestCheckpointDict['ValLossHistory']
                else:
                    self.ValLossHistory = self.LossHistory
                if 'SeparateLossesHistory' in LatestCheckpointDict:
                    self.SeparateLossesHistory = LatestCheckpointDict['SeparateLossesHistory']
                else:
                    self.SeparateLossesHistory = self.LossHistory

                # Move optimizer state to GPU if needed. See https://github.com/pytorch/pytorch/issues/2830
                if TrainDevice != 'cpu':
                    for state in self.Optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(TrainDevice)
            else:
                print('[ INFO ]: Experiment names do not match. Training from scratch.')

    def validate(self, ValDataLoader, Objective, Device='cpu'):
        self.eval()         #switch to evaluation mode
        ValLosses = []
        Tic = ptUtils.getCurrentEpochTime()
        # print('Val length:', len(ValDataLoader))
        for i, (Data, Targets) in enumerate(ValDataLoader, 0):  # Get each batch
            DataTD = ptUtils.sendToDevice(Data, Device)
            TargetsTD = ptUtils.sendToDevice(Targets, Device)

            Output = self.forward(DataTD)
            Loss = Objective(Output, TargetsTD)
            ValLosses.append(Loss.item())

            # Print stats
            Toc = ptUtils.getCurrentEpochTime()
            Elapsed = math.floor((Toc - Tic) * 1e-6)
            done = int(50 * (i+1) / len(ValDataLoader))
            sys.stdout.write(('\r[{}>{}] val loss - {:.8f}, elapsed - {}')
                             .format('+' * done, '-' * (50 - done), np.mean(np.asarray(ValLosses)), ptUtils.getTimeDur(Elapsed)))
            sys.stdout.flush()
        sys.stdout.write('\n')
        self.train()     #switch back to train mode

        return ValLosses

    def fit(self, TrainDataLoader, Optimizer=None, Objective=nn.MSELoss(), TrainDevice='cpu', ValDataLoader=None):
        if Optimizer is None:
            # Optimizer = optim.SGD(NN.parameters(), lr=Args.learning_rate)  # , momentum=0.9)
            self.Optimizer = optim.Adam(self.parameters(), lr=self.Config.Args.learning_rate, weight_decay=1e-5) # PARAM
        else:
            self.Optimizer = Optimizer

        ObjectiveFunc = Objective
        if isinstance(ObjectiveFunc, ptNetLoss) == False:
            ObjectiveFunc = ptNetLoss(Losses=[ObjectiveFunc], Weights=[1.0])  # Cast to ptNetLoss

        self.setupCheckpoint(TrainDevice)

        print('[ INFO ]: Training on {}'.format(TrainDevice))
        self.to(TrainDevice)
        CurrLegend = ['Train loss', *ObjectiveFunc.Names]

        AllTic = ptUtils.getCurrentEpochTime()
        for Epoch in range(self.Config.Args.epochs):
            try:
                EpochLosses = [] # For all batches in an epoch
                EpochSeparateLosses = []  # For all batches in an epoch
                Tic = ptUtils.getCurrentEpochTime()
                for i, (Data, Targets) in enumerate(TrainDataLoader, 0):  # Get each batch
                    DataTD = ptUtils.sendToDevice(Data, TrainDevice)
                    TargetsTD = ptUtils.sendToDevice(Targets, TrainDevice)

                    self.Optimizer.zero_grad()

                    # Forward, backward, optimize
                    Output = self.forward(DataTD)

                    Loss = ObjectiveFunc(Output, TargetsTD)
                    Loss.backward()
                    self.Optimizer.step()
                    EpochLosses.append(Loss.item())
                    EpochSeparateLosses.append(ObjectiveFunc.getItems())

                    gc.collect() # Collect garbage after each batch

                    # Terminate early if loss is nan
                    isTerminateEarly = False
                    if math.isnan(EpochLosses[-1]):
                        print('[ WARN ]: NaN loss encountered. Terminating training and saving current model checkpoint (might be junk).')
                        isTerminateEarly = True
                        break

                    # Print stats
                    Toc = ptUtils.getCurrentEpochTime()
                    Elapsed = math.floor((Toc - Tic) * 1e-6)
                    TotalElapsed = math.floor((Toc - AllTic) * 1e-6)
                    # Compute ETA
                    TimePerBatch = (Toc - AllTic) / ((Epoch * len(TrainDataLoader)) + (i+1)) # Time per batch
                    ETA = math.floor(TimePerBatch * self.Config.Args.epochs * len(TrainDataLoader) * 1e-6)
                    done = int(50 * (i+1) / len(TrainDataLoader))
                    ProgressStr = ('\r[{}>{}] epoch - {}/{}, train loss - {:.8f} | epoch - {}, total - {} ETA - {} |').format('=' * done, '-' * (50 - done), self.StartEpoch + Epoch + 1, self.StartEpoch + self.Config.Args.epochs
                                             , np.mean(np.asarray(EpochLosses)), ptUtils.getTimeDur(Elapsed), ptUtils.getTimeDur(TotalElapsed), ptUtils.getTimeDur(ETA-TotalElapsed))
                    sys.stdout.write(ProgressStr.ljust(150))
                    sys.stdout.flush()
                sys.stdout.write('\n')

                self.LossHistory.append(np.mean(np.asarray(EpochLosses)))

                # Transpose and sum: https://stackoverflow.com/questions/47114706/python-sum-first-element-of-a-list-of-lists
                SepMeans = list(map(sum, zip(*EpochSeparateLosses)))
                SepMeans[:] = [x / len(EpochLosses) for x in SepMeans]
                self.SeparateLossesHistory.append(SepMeans)
                if ValDataLoader is not None:
                    ValLosses = self.validate(ValDataLoader, Objective, TrainDevice)
                    self.ValLossHistory.append(np.mean(np.asarray(ValLosses)))
                    # print('Last epoch val loss - {:.16f}'.format(self.ValLossHistory[-1]))
                    CurrLegend = ['Train loss', 'Val loss', *ObjectiveFunc.Names]

                # Always save checkpoint after an epoch. Will be replaced each epoch. This is independent of requested checkpointing
                self.saveCheckpoint(Epoch, CurrLegend, TimeString='eot', PrintStr='~'*3)

                isLastLoop = (Epoch == self.Config.Args.epochs-1) and (i == len(TrainDataLoader)-1)
                if (Epoch + 1) % self.SaveFrequency == 0 or isTerminateEarly or isLastLoop:
                    self.saveCheckpoint(Epoch, CurrLegend)
                    if isTerminateEarly:
                        break
            except (KeyboardInterrupt, SystemExit):
                print('\n[ INFO ]: KeyboardInterrupt detected. Saving checkpoint.')
                self.saveCheckpoint(Epoch, CurrLegend, TimeString='eot', PrintStr='$'*3)
                break
            except Exception as e:
                print(traceback.format_exc())
                print('\n[ WARN ]: Exception detected. *NOT* saving checkpoint. {}'.format(e))
                # self.saveCheckpoint(Epoch, CurrLegend, TimeString='eot', PrintStr='$'*3)
                break

        AllToc = ptUtils.getCurrentEpochTime()
        print('[ INFO ]: All done in {}.'.format(ptUtils.getTimeDur((AllToc - AllTic) * 1e-6)))

    def saveCheckpoint(self, Epoch, CurrLegend, TimeString='humanlocal', PrintStr='*'*3):
        CheckpointDict = {
            'Name': self.Config.Args.expt_name,
            'ModelStateDict': self.state_dict(),
            'OptimizerStateDict': self.Optimizer.state_dict(),
            'LossHistory': self.LossHistory,
            'ValLossHistory': self.ValLossHistory,
            'SeparateLossesHistory': self.SeparateLossesHistory,
            'Epoch': self.StartEpoch + Epoch + 1,
            'SavedTimeZ': ptUtils.getZuluTimeString(),
        }
        OutFilePath = ptUtils.savePyTorchCheckpoint(CheckpointDict, self.ExptDirPath, TimeString=TimeString)
        # ptUtils.saveLossesCurve(self.LossHistory, self.ValLossHistory, out_path=os.path.splitext(OutFilePath)[0] + '.png',
        #                         xlim = [0, int(self.Config.Args.epochs + self.StartEpoch)], legend=CurrLegend, title=self.Config.Args.expt_name)
        TSLH = list(map(list, zip(*self.SeparateLossesHistory))) # Transposed list
        ptUtils.saveLossesCurve(self.LossHistory, self.ValLossHistory, *TSLH, out_path=os.path.splitext(OutFilePath)[0] + '.png',
                                xlim = [0, int(self.Config.Args.epochs + self.StartEpoch)], legend=CurrLegend, title=self.Config.Args.expt_name)

        # print('[ INFO ]: Checkpoint saved.')
        print(PrintStr) # Checkpoint saved. 50 + 3 characters [>]

    def forward(self, x):
        print('[ WARN ]: This is an identity network. Override this in a derived class.')
        return x
