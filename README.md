## Idea
+ Use neural network to predict victory of a DotA game based on heroes. 
    + After this is done, factors including MMR and party status might be considered.

## Parameters
+ Number of features: 216.
+ Connection rate: 0.8.
+ Neuron distribution: [216, 316, 20, 10, 1].
+ Activation function: symmetric sigmoid. 
+ Learning rate: 0.7.
+ Training set size: 148064.
+ Validation set size: not introduced YET.
+ Test set size: 16451.

## Language
+ Mainly Python 2. 
    + Due to some weird bugs of FANN, a C++ file is included as an intended workaround. (Nevertheless, the problem persists.)

## Libraries
+ FANN
    + Considering PyBrain for FANN's bug.

## Data
For now, data is borrowed from [here](http://kevintechnology.com/post/71621133663/using-machine-learning-to-recommend-heroes-for).

## Compiling
+ On OSX 10.10, mock_input.cpp is compiled with

        gcc -o mock mock_input.cpp -L /usr/local/lib -ldoublefann -lm -stdlib=libstdc++ -lstdc++ -I /usr/local/include/
    
## Current Issues
+ FANN can train a network with ~10% error rate on test set. However, some internal bugs of FANN prevent me from using the trained network.