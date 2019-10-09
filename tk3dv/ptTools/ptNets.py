import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os, sys, argparse, math, glob, gc
import numpy as np

FileDirPath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(FileDirPath, '.'))

import logging, ptUtils

class ptNetExptConfig():
    def __init__(self, InputArgs=None, isPrint=True):
        self.Parser = argparse.ArgumentParser(description='Parse arguments for a PyTorch neural network.', fromfile_prefix_chars='@')

        # Search params
        self.Parser.add_argument('--learning-rate', help='Choose the learning rate.', required=False, default=0.001,
                            type=ptUtils.restricted_float)
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

        if self.Args.rel_output_dir is None and self.Args.output_dir is None:
            raise RuntimeError('[ ERR ]: One or both of --output-dir or --rel-output-dir is required.')

        if self.Args.rel_output_dir is not None: # Relative path takes precedence
            if self.Args.output_dir is not None:
                ptUtils.ptToolsLogger.info('Relative path taking precedence to absolute path.')
            DirPath = os.getcwd() # os.path.dirname(os.path.realpath(__file__))
            for Arg in InputArgs:
                if '@' in Arg: # Config file is passed, path should be relative to config file
                    DirPath = os.path.abspath(os.path.dirname(ptUtils.expandTilde(Arg[1:]))) # Abs directory path of config file
                    break
            self.Args.output_dir = os.path.join(DirPath, self.Args.rel_output_dir)
            ptUtils.ptToolsLogger.info('Converted relative path {} to absolute path {}'.format(self.Args.rel_output_dir, self.Args.output_dir))

        # Logging directory and file
        self.ExptDirPath = os.path.join(ptUtils.expandTilde(self.Args.output_dir), self.Args.expt_name)
        if os.path.exists(self.ExptDirPath) == False:
            os.makedirs(self.ExptDirPath)

        ExptLogFile = os.path.join(self.ExptDirPath, self.Args.expt_name + '.log')
        if os.path.exists(ExptLogFile) == False:
            with open(ExptLogFile, 'a'):
                os.utime(ExptLogFile, None)
        LogFileHandler = logging.FileHandler(ExptLogFile)
        LogFileHandler.setFormatter(ptUtils.LogFormat)
        ptUtils.ptToolsLogger.addHandler(LogFileHandler)

        if isPrint:
            ArgsDict = vars(self.Args)
            for Arg in ArgsDict:
                if ArgsDict[Arg] is not None:
                    print('{:<15}:   {:<50}'.format(Arg, ArgsDict[Arg]))
                else:
                    print('{:<15}:   {:<50}'.format(Arg, 'NOT DEFINED'))

    def getHelp(self):
        self.Parser.print_help()

    def serialize(self, FilePath, isAppend=True):
        ptUtils.configSerialize(self.Args, FilePath, isAppend)

class ptNet(nn.Module):
    def __init__(self, Args=None):
        super().__init__()

        self.Config = ptNetExptConfig(InputArgs=Args)
        self.ExptDirPath = self.Config.ExptDirPath

        # Defaults
        self.StartEpoch = 0
        self.ExptDirPath = ''
        self.SaveFrequency = self.Config.Args.save_freq if self.Config.Args.save_freq > 0 else self.Config.Args.epochs
        self.LossHistory = []
        self.ValLossHistory = []

    def loadCheckpoint(self, Path=None, Device='cpu'):
        if Path is None:
            self.ExptDirPath = os.path.join(ptUtils.expandTilde(self.Config.Args.output_dir), self.Config.Args.expt_name)
            ptUtils.ptToolsLogger.info('Loading from latest checkpoint.')
            CheckpointDict = ptUtils.loadLatestPyTorchCheckpoint(self.ExptDirPath, map_location=Device)
        else: # Load latest
            ptUtils.ptToolsLogger.info('Loading from checkpoint {}'.format(Path))
            CheckpointDict = ptUtils.loadPyTorchCheckpoint(Path)

        self.load_state_dict(CheckpointDict['ModelStateDict'])

    def setupCheckpoint(self, TrainDevice):
        LatestCheckpointDict = None
        AllCheckpoints = glob.glob(os.path.join(self.ExptDirPath, '*.tar'))
        if len(AllCheckpoints) > 0:
            LatestCheckpointDict = ptUtils.loadLatestPyTorchCheckpoint(self.ExptDirPath, map_location=TrainDevice)
            ptUtils.ptToolsLogger.info('Loading from last checkpoint.')

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

                # Move optimizer state to GPU if needed. See https://github.com/pytorch/pytorch/issues/2830
                if TrainDevice is not 'cpu':
                    for state in self.Optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(TrainDevice)
            else:
                ptUtils.ptToolsLogger.info('Experiment names do not match. Training from scratch.')

    def validate(self, ValDataLoader, Objective, Device='cpu'):
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
            sys.stdout.write(('\r[{}>{}] val loss - {:.16f}, elapsed - {}')
                             .format('=' * done, '-' * (50 - done), np.mean(np.asarray(ValLosses)), ptUtils.getTimeDur(Elapsed)))
            sys.stdout.flush()
        sys.stdout.write('\n')


        return ValLosses

    def fit(self, TrainDataLoader, Optimizer=None, Objective=nn.MSELoss(), TrainDevice='cpu', ValDataLoader=None):
        if Optimizer is None:
            # Optimizer = optim.SGD(NN.parameters(), lr=Args.learning_rate)  # , momentum=0.9)
            self.Optimizer = optim.Adam(self.parameters(), lr=self.Config.Args.learning_rate, weight_decay=1e-5)
        else:
            self.Optimizer = Optimizer

        self.setupCheckpoint(TrainDevice)

        ptUtils.ptToolsLogger.info('Training on {}'.format(TrainDevice))
        self.to(TrainDevice)
        CurrLegend = ['Train loss']

        AllTic = ptUtils.getCurrentEpochTime()
        for Epoch in range(self.Config.Args.epochs):
            EpochLosses = [] # For all batches in an epoch
            Tic = ptUtils.getCurrentEpochTime()
            for i, (Data, Targets) in enumerate(TrainDataLoader, 0):  # Get each batch
                DataTD = ptUtils.sendToDevice(Data, TrainDevice)
                TargetsTD = ptUtils.sendToDevice(Targets, TrainDevice)

                self.Optimizer.zero_grad()

                # Forward, backward, optimize
                Output = self.forward(DataTD)

                Loss = Objective(Output, TargetsTD)
                Loss.backward()
                self.Optimizer.step()
                EpochLosses.append(Loss.item())

                gc.collect() # Collect garbage after each batch

                # Terminate early if loss is nan
                isTerminateEarly = False
                if math.isnan(EpochLosses[-1]):
                    ptUtils.ptToolsLogger.warning('NaN loss encountered. Terminating training and saving current model checkpoint (might be junk).')
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
                sys.stdout.write(('\r[{}>{}] epoch - {}/{}, train loss - {:.16f} | epoch - {}, total - {} ETA - {} |')
                                 .format('=' * done, '-' * (50 - done), self.StartEpoch + Epoch + 1, self.StartEpoch + self.Config.Args.epochs
                                         , np.mean(np.asarray(EpochLosses)), ptUtils.getTimeDur(Elapsed), ptUtils.getTimeDur(TotalElapsed), ptUtils.getTimeDur(ETA-TotalElapsed)))
                sys.stdout.flush()
            sys.stdout.write('\n')

            self.LossHistory.append(np.mean(np.asarray(EpochLosses)))
            if ValDataLoader is not None:
                ValLosses = self.validate(ValDataLoader, Objective, TrainDevice)
                self.ValLossHistory.append(np.mean(np.asarray(ValLosses)))
                # print('Last epoch val loss - {:.16f}'.format(self.ValLossHistory[-1]))
                CurrLegend = ['Train loss', 'Val loss']

            isLastLoop = (Epoch == self.Config.Args.epochs-1) and (i == len(TrainDataLoader)-1)
            if (Epoch + 1) % self.SaveFrequency == 0 or isTerminateEarly or isLastLoop:
                CheckpointDict = {
                    'Name': self.Config.Args.expt_name,
                    'ModelStateDict': self.state_dict(),
                    'OptimizerStateDict': self.Optimizer.state_dict(),
                    'LossHistory': self.LossHistory,
                    'ValLossHistory': self.ValLossHistory,
                    'Epoch': self.StartEpoch + Epoch + 1,
                    'SavedTimeZ': ptUtils.getZuluTimeString(),
                }
                print('HERHERHERE: ', self.ExptDirPath)
                OutFilePath = ptUtils.savePyTorchCheckpoint(CheckpointDict, self.ExptDirPath)
                ptUtils.saveLossesCurve(self.LossHistory, self.ValLossHistory, out_path=os.path.splitext(OutFilePath)[0] + '.jpg',
                                        xlim = [0, int(self.Config.Args.epochs + self.StartEpoch)], legend=CurrLegend, title=self.Config.Args.expt_name)
                ptUtils.ptToolsLogger.info('Saved checkpoint and loss curve.')

                if isTerminateEarly:
                    break

        AllToc = ptUtils.getCurrentEpochTime()
        ptUtils.ptToolsLogger.info('All done in {} s.'.format(ptUtils.getTimeDur((AllToc - AllTic) * 1e-6)))

    def forward(self, x):
        ptUtils.ptToolsLogger.warning('This is an identity network. Override this in a derived class.')
        return x
