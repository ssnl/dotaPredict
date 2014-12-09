from pybrain.datasets import SupervisedDataSet
from pybrain.tools.customxml.networkreader import NetworkReader
from pybrain.supervised.trainers import BackpropTrainer

test_ds = SupervisedDataSet.loadFromFile('test.data')

print "Test data loaded"

net = NetworkReader.readFrom('network.xml')

print "Network loaded"

trainer = BackpropTrainer(net)
trainer.testOnData(test_ds, verbose = True)