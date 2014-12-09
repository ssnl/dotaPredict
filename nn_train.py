from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import SigmoidLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

from progressbar import ProgressBar, Bar, Percentage, FormatLabel, ETA
from cPickle import *

NUM_HEROES = 108
NUM_FEATURES = NUM_HEROES * 2

with open("dataset.pkl", "rb") as f:
    ds = load(f)
    print "data loaded"

train_ds, test_ds = ds.splitWithProportion(0.9)

net = buildNetwork(NUM_FEATURES, NUM_FEATURES + 100, 20, 10, 1, outclass = SigmoidLayer)

trainer = BackpropTrainer(net, train_ds)

trainer.trainUntilConvergence(verbose = True, maxEpochs = 150, continueEpochs = 15, validationProportion = 0.2)
