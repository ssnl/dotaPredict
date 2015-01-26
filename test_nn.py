from pybrain.datasets import SupervisedDataSet
from pybrain.tools.customxml.networkreader import NetworkReader
from pybrain.supervised.trainers import BackpropTrainer
from os.path import isfile
from util import feature_to_names, push_to_int, int_to_side
from constants import *

assert isfile(NETWORK_FILE_NAME)
assert isfile(TEST_FILE_NAME)

test_ds = SupervisedDataSet.loadFromFile(TEST_FILE_NAME)
print "Test dataset loaded"

net = NetworkReader.readFrom(NETWORK_FILE_NAME)
print "Network loaded"

trainer = BackpropTrainer(net)
trainer.testOnData(test_ds, verbose = True)

error = 0

for datum in test_ds:
    x, y = datum[0], datum[1][0]
    predict = push_to_int(net.activate(x))
    error += predict != y
    # print "Heroes: {0}, Result: {1}, Predict: {2}".format(", ".join(feature_to_names(x)), int_to_side(y), int_to_side(predict))

print "{0} errors out of {1} data".format(error, len(test_ds))
print "Error rate: {0}".format(float(error) / len(test_ds))