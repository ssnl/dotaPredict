from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import SigmoidLayer, TanhLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader
from pymongo import MongoClient
from progressbar import ProgressBar, Bar, Percentage, FormatLabel, ETA
from os.path import isfile
from cPickle import dump, load
from constants import *
from util import is_valid_match
import atexit

client = MongoClient()
db = client.dotabot
matches = db.matches

# Dataset manipulation
if isfile(TRAIN_FILE_NAME) and isfile(VALIDATION_FILE_NAME) and isfile(TEST_FILE_NAME):
    test_ds = SupervisedDataSet.loadFromFile(TEST_FILE_NAME)
    valid_ds = SupervisedDataSet.loadFromFile(VALIDATION_FILE_NAME)
    train_ds = SupervisedDataSet.loadFromFile(TRAIN_FILE_NAME)
    print "Training, validation and test datasets loaded"
else:
    ds = SupervisedDataSet(NUM_FEATURES, 1)

    widgets = [FormatLabel('Processed: %(value)d/%(max)d matches. '), ETA(), ' ', Percentage(), ' ', Bar()]
    pbar = ProgressBar(widgets = widgets, maxval = NUM_MATCHES).start()

    seen = set()
    r, d = 0, 0

    for i, record in enumerate(matches.find()):
        if record['match_id'] in seen:
            # print "Ignore redundant match {0}".format(record['match_id'])
            continue
        if not is_valid_match(record):
            # print "Ignore invalid match {0}".format(record['match_id'])
            continue
        seen.add(record['match_id'])
        y = 1.0 if record['radiant_win'] else 0.0
        if record['radiant_win']:
            r += 1
        else:
            d += 1
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

        y = 1.0 - y
        x = [0.0 for _ in xrange(NUM_FEATURES)]
        players = record['players']
        for player in players:
            hero_id = player['hero_id'] - 1

            # If the left-most bit of player_slot is set,
            # this player is on dire, so push the index accordingly
            player_slot = player['player_slot']
            if player_slot < 128:
                hero_id += NUM_HEROES

            x[hero_id] = 1.0

        ds.addSample(x, y)
        pbar.update(i)

    pbar.finish()
    print "Dataset built"
    print "Radiant {0}; Dire {1}".format(r, d)

    train_ds, test_ds = ds.splitWithProportion(1 - VALIDATION_RATIO - TEST_RATIO)
    valid_ds, test_ds = test_ds.splitWithProportion(VALIDATION_RATIO / (VALIDATION_RATIO + TEST_RATIO))
    test_ds.saveToFile(TEST_FILE_NAME)
    valid_ds.saveToFile(VALIDATION_FILE_NAME)
    train_ds.saveToFile(TRAIN_FILE_NAME)
    print "Training, validation and test dataset built"

# Network manipulation
if isfile(NETWORK_TEMP_FILE_NAME) and isfile(NETWORK_VAL_FILE_NAME):
    net = NetworkReader.readFrom(NETWORK_TEMP_FILE_NAME)
    trainer = BackpropTrainer(net, train_ds, learningrate = 0.05)
    with open(NETWORK_VAL_FILE_NAME, "rb") as f:
        epoch, additional_left, best = load(f)
    print "Network loaded with best averge validation error {0}".format(best)
else:
    net = buildNetwork(NUM_FEATURES, NUM_FEATURES + 100, 100, 20, 10, 1, \
        hiddenclass = SigmoidLayer, \
        outclass = SigmoidLayer, \
        fast = True, \
        bias = True)
    trainer = BackpropTrainer(net, train_ds, learningrate = 0.5)
    epoch, additional_left, best = 0, ADDITIONAL_EPOCH, trainer.testOnData(valid_ds)
    print "Network built with averge validation error {0}".format(best)

# If something wrong happens..
def save_values():
    global best, additional_left
    NetworkWriter.writeToFile(net, NETWORK_TEMP_FILE_NAME)
    if 'avg_error' in globals() and avg_error < best:
        best = avg_error
        NetworkWriter.writeToFile(net, NETWORK_FILE_NAME)
        print "Updated best network and saved to file"
        additional_left = ADDITIONAL_EPOCH
    with open(NETWORK_VAL_FILE_NAME, "wb") as f:
        dump((epoch, additional_left, best), f)
    print "Network values saved"

atexit.register(save_values)

# Training
while True:
    epoch += 1
    print "Average error on train after epoch {0}: {1}".format(epoch, trainer.train())
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
