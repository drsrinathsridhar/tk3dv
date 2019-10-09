import logging, os, sys, time

OutFile = './log.log'

# ptToolsLogger = logging.getLogger()
# print('Propagate:', ptToolsLogger.propagate)
# # LogFormat = logging.Formatter('[ %(asctime)s ] [ {} ] [ %(levelname)-5.5s ]: %(message)s'.format(__name__))
# LogFormat = logging.Formatter('[ %(levelname)-5.5s ]: %(message)s')
# ptToolsLogger.setLevel(logging.DEBUG)
#
# ptToolsLogger.info('Zero')
#
# StrBuffer = ''
# EmptyFormat = logging.Formatter('%(message)s')
# StrHandler = logging.StreamHandler(StrBuffer)
# StrHandler.setFormatter(EmptyFormat)
# ptToolsLogger.addHandler(StrHandler)
#
# ptToolsLogger.info(StrBuffer)
# ptToolsLogger.info('One')
#
# StdoutHandler = logging.StreamHandler(sys.stdout)
# StdoutHandler.setFormatter(LogFormat)
# ptToolsLogger.addHandler(StdoutHandler)
#
# ptToolsLogger.info('Two')
#
# if os.path.exists(OutFile) == False:
#     with open(OutFile, 'a'):
#         os.utime(OutFile, None)
# LogFileHandler = logging.FileHandler(OutFile)
# LogFileHandler.setFormatter(LogFormat)
# ptToolsLogger.addHandler(LogFileHandler)
#
# ptToolsLogger.info('Three')
#
# time.sleep(10)
#
# ptToolsLogger.info('Four')
#


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(OutFile, 'w+')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger()

print('Hi')

import time
import sys

toolbar_width = 40

for i in range(toolbar_width):
    time.sleep(0.1) # do real work here
    # update the bar
    done = int(10 * (i+1) / toolbar_width)
    sys.stdout.write('\r[{}>{}]'.format('=' * done, '-' * (toolbar_width-done)))
    sys.stdout.flush()

sys.stdout.write("]\n") # this ends the progress bar
