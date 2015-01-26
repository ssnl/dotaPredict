from pybrain.datasets import SupervisedDataSet
from pybrain.tools.customxml.networkreader import NetworkReader
from pybrain.supervised.trainers import BackpropTrainer
from os.path import isfile
import sys
from util import push_to_int
from util import *
from constants import *

assert isfile(NETWORK_FILE_NAME)

net = NetworkReader.readFrom(NETWORK_FILE_NAME)
print "Network loaded"

if __name__ == "__main__":
    args = sys.argv[1:]
    assert len(args) == 10
    out = float(net.activate(names_to_feature(args)))
    if out > 0.75:
        print "Radiant probably got this!"
    elif out > 0.5:
        print "Hard to say, but Radiant has some advantage."
    elif out > 0.25:
        print "Hard to say, but Dire has some advantage."
    else:
        print "Dire probably got this!"

