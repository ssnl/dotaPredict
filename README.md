## Idea
+ Use neural network to predict victory of a DotA game based on heroes.
    + After this is done, factors including MMR and party status might be considered.
    + But those data are not as easy to get.

## Parameters
+ Number of features: 216, discrete {0, 1}.
+ Number of output: 1, discrete {0, 1}.
+ Connection rate: fully connected.
+ Neuron distribution: [216, 316, 20, 10, 1].
+ Activation function: sigmoid.
+ Learning rate: 0.5.
+ Total data: 164515.
+ Training set: 80%.
+ Validation set: 10%.
+ Test set size: 10%.

## Language
+ Mainly Python 2.

## Libraries
+ PyBrain + arac.
    + Had to pass on FANN due to some bug of the library.

## Data Souce
For now, data is borrowed from [here](http://kevintechnology.com/post/71621133663/using-machine-learning-to-recommend-heroes-for).

## Usage
+ I run `test_nn.py` with

        python test_nn.py > test.log

+ If you want to see how FANN works,
    + On OSX 10.10, mock_input.cpp is compiled with

            gcc -o mock mock_input.cpp -L /usr/local/lib -ldoublefann -lm -stdlib=libstdc++ -lstdc++ -I /usr/local/include/


## Progress
+ FANN can train a network with ~10% error rate on test set. However, some internal bugs of FANN prevent me from using the trained network.
+ ~5 epoches using PyBrain gives ~0.04 average error and ~11.5% error rate on test set.