# ptTools
A bunch of utils and tools for PyTorch.

## ptUtils
ptUtils contains utilities such as save load PyTorch checkpoints, load latest checkpoints, time counters, etc.

## ptNets
ptNets contains a base class for neural network training that encapsulates automatic model loading/saving.

## Sample
Run the following code for a minimal MNIST example that uses ptUtils.

Training:  
`python examples/MNIST.py --mode train --expt-name MNISTTest --input-dir <DIR_TO_DATA> --output-dir <OUTPUT_DIR>`

Testing:  
`python examples/MNIST.py --mode test --expt-name MNISTTest --input-dir <DIR_TO_DATA> --output-dir <OUTPUT_DIR>`

Note: When running the same experiment (specified by `--expt-name`) Use the same directory specified for `--input-dir` and `--output-dir`)

# Contact
Srinath Sridhar  
[srinaths@umich.edu][1]

[1]: [mailto:srinaths@umich.edu]
