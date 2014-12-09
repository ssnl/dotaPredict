from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import SigmoidLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader
from pymongo import MongoClient
from progressbar import ProgressBar, Bar, Percentage, FormatLabel, ETA
from os.path import isfile
from cPickle import dump, load
import atexit

client = MongoClient()
db = client.dotabot
matches = db.matches

NUM_HEROES = 108
NUM_FEATURES = NUM_HEROES * 2
NUM_MATCHES = matches.count()
TRAIN_FILE_NAME = './data/train.data'
VALIDATION_FILE_NAME = './data/valid.data'
TEST_FILE_NAME = './data/test.data'
NETWORK_FILE_NAME = './data/network.xml'
NETWORK_VAL_FILE_NAME = './data/network_val.save'
ADDITIONAL_EPOCH = 10
MAX_EPOCH = 50
VALIDATION_RATIO = 0.1
TEST_RATIO = 0.1

# Dataset manipulation
if isfile(TRAIN_FILE_NAME) and isfile(VALIDATION_FILE_NAME) and isfile(TEST_FILE_NAME):
    test_ds = SupervisedDataSet.loadFromFile(TEST_FILE_NAME)
    valid_ds = SupervisedDataSet.loadFromFile(VALIDATION_FILE_NAME)
    train_ds = SupervisedDataSet.loadFromFile(TRAIN_FILE_NAME)
    print "Training, validation and test dataset loaded"
else:
    ds = SupervisedDataSet(NUM_FEATURES, 1)

    widgets = [FormatLabel('Processed: %(value)d/%(max)d matches. '), ETA(), ' ', Percentage(), ' ', Bar()]
    pbar = ProgressBar(widgets = widgets, maxval = NUM_MATCHES).start()

    for i, record in enumerate(matches.find()):
        y = 1.0 if record['radiant_win'] else 0.0
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

    train_ds, test_ds = ds.splitWithProportion(1 - VALIDATION_RATIO - TEST_RATIO)
    valid_ds, test_ds = test_ds.splitWithProportion(VALIDATION_RATIO / (VALIDATION_RATIO + TEST_RATIO))
    test_ds.saveToFile(TEST_FILE_NAME)
    valid_ds.saveToFile(VALIDATION_FILE_NAME)
    train_ds.saveToFile(TRAIN_FILE_NAME)
    print "Training, validation and test dataset built"

# Network manipulation
if isfile(NETWORK_FILE_NAME) and isfile(NETWORK_VAL_FILE_NAME):
    net = NetworkReader.readFrom(NETWORK_FILE_NAME)
    trainer = BackpropTrainer(net, train_ds, learningrate = 0.5)
    with open(NETWORK_VAL_FILE_NAME, "rb") as f:
        epoch, additional_left, best = load(f)
    print "Network loaded with best averge validation error {0}".format(best)
else:
    net = buildNetwork(NUM_FEATURES, NUM_FEATURES + 100, 20, 10, 1, outclass = SigmoidLayer, fast = True)
    trainer = BackpropTrainer(net, train_ds, learningrate = 0.5)
    epoch, additional_left, best = 0, ADDITIONAL_EPOCH, trainer.testOnData(valid_ds)
    print "Network built with averge validation error {0}".format(best)

# If something wrong happens..
def save_values():
    global best
    if avg_error < best:
        best = avg_error
        NetworkWriter.writeToFile(net, 'network.xml')
        print "Updated best network and saved to file"
        additional_left = ADDITIONAL_EPOCH
    with open(NETWORK_VAL_FILE_NAME, "wb") as f:
        dump((epoch, additional_left, best), f)
    print "Network values saved"

atexit.register(save_values)

# Training
while True:
    print "Average error on train after epoch {0}: {1}".format(epoch + 1, trainer.train())
    epoch += 1
    avg_error = trainer.testOnData(valid_ds)
    print "Average error on validation set after epoch {0}: {1}".format(epoch, avg_error)
    if avg_error < best:
        best = avg_error
        NetworkWriter.writeToFile(net, NETWORK_FILE_NAME)
        print "Updated best network and saved to file"
        additional_left = ADDITIONAL_EPOCH
    else:
        additional_left -= 1
    if additional_left < 0:
        print "Additional epoches used up"
        break
    if epoch == MAX_EPOCH:
        break

print "Training terminates after {0} epoches".format(epoch)

# trainer.trainUntilConvergence(verbose = True, maxEpochs = 150, continueEpochs = 15, validationProportion = 0.2)
