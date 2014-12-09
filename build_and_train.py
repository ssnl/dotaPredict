from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import SigmoidLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader

from pymongo import MongoClient
from progressbar import ProgressBar, Bar, Percentage, FormatLabel, ETA

NUM_HEROES = 108
NUM_FEATURES = NUM_HEROES * 2

client = MongoClient()
db = client.dotabot
matches = db.matches

# Our training label vector, Y, is a bit vector indicating
# whether radiant won (1) or lost (-1)
NUM_MATCHES = matches.count()

ds = SupervisedDataSet(NUM_FEATURES, 1)

widgets = [FormatLabel('Processed: %(value)d/%(max)d matches. '), ETA(), ' ', Percentage(), ' ', Bar()]
pbar = ProgressBar(widgets = widgets, maxval = NUM_MATCHES).start()

for i, record in enumerate(matches.find()):
    y = 1.0 if record['radiant_win'] else -1.0
    x = [0.0 for _ in xrange(NUM_FEATURES)]
    players = record['players']
    for player in players:
        hero_id = player['hero_id'] - 1

        # If the left-most bit of player_slot is set,
        # this player is on dire, so push the index accordingly
        player_slot = player['player_slot']
        if player_slot >= 128:
            hero_id += NUM_HEROES

        x[hero_id] = 1.0

    ds.addSample(x, y)
    pbar.update(i)

pbar.finish()

print "Dataset built"

train_ds, test_ds = ds.splitWithProportion(0.9)

print "Training dataset and test dataset built"

net = buildNetwork(NUM_FEATURES, NUM_FEATURES + 100, 20, 10, 1, outclass = SigmoidLayer, fast = True)

print "Network built"

trainer = BackpropTrainer(net, train_ds)

for _ in xrange(50):
    print trainer.train()
    NetworkWriter.writeToFile(net, 'network.xml')

# trainer.trainUntilConvergence(verbose = True, maxEpochs = 150, continueEpochs = 15, validationProportion = 0.2)
