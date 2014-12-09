from pybrain.datasets import SupervisedDataSet
from pybrain.tools.customxml.networkreader import NetworkReader
from pybrain.supervised.trainers import BackpropTrainer
from os.path import isfile
from util import push_to_int

NETWORK_FILE_NAME = 'network.xml'
TEST_FILE_NAME = 'test.data'

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
    error += push_to_int(net.activate(x)) != y

print "{0} errors out of {1} data".format(error, len(test_ds))
print "Error rate: {0}".format(float(error) / len(test_ds))