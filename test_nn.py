from pybrain.datasets import SupervisedDataSet
from pybrain.tools.customxml.networkreader import NetworkReader
from pybrain.supervised.trainers import BackpropTrainer

test_ds = SupervisedDataSet.loadFromFile('test.data')

print "Test data loaded"

net = NetworkReader.readFrom('network.xml')

print "Network loaded"

trainer = BackpropTrainer(net)
trainer.testOnData(test_ds, verbose = True)

error = 0

def push_to_int(y):
    return 0 if y < 0.5 else 1

for datum in test_ds:
    x, y = datum[0], datum[1][0]
    error += push_to_int(net.activate(x)) != y

print "{0} errors out of {1} data".format(error, len(test_ds))
print "Error rate: {0}".format(float(error) / len(test_ds))