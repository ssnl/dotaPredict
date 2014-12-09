from pymongo import MongoClient
from progressbar import ProgressBar, Bar, Percentage, FormatLabel, ETA
import numpy as np

client = MongoClient()
db = client.dotabot
matches = db.matches

NUM_HEROES = 108
NUM_FEATURES = NUM_HEROES * 2

# Our training label vector, Y, is a bit vector indicating
# whether radiant won (1) or lost (-1)
NUM_MATCHES = matches.count()

# Initialize training matrix
X = np.zeros((NUM_MATCHES, NUM_FEATURES), dtype=np.int8)


# Initialize training label vector
Y = np.zeros(NUM_MATCHES, dtype=np.int8)

widgets = [FormatLabel('Processed: %(value)d/%(max)d matches. '), ETA(), ' ', Percentage(), ' ', Bar()]
pbar = ProgressBar(widgets = widgets, maxval = NUM_MATCHES).start()

for i, record in enumerate(matches.find()):
    Y[i] = 1 if record['radiant_win'] else -1
    players = record['players']
    for player in players:
        hero_id = player['hero_id'] - 1

        # If the left-most bit of player_slot is set,
        # this player is on dire, so push the index accordingly
        player_slot = player['player_slot']
        if player_slot >= 128:
            hero_id += NUM_HEROES

        X[i, hero_id] = 1
    pbar.update(i)

pbar.finish()

print "Permuting, generating train and test sets."
indices = np.random.permutation(NUM_MATCHES)
test_indices = indices[0 : NUM_MATCHES / 10]
train_indices = indices[NUM_MATCHES / 10 : NUM_MATCHES]

X_test = X[test_indices]
Y_test = Y[test_indices]

X_train = X[train_indices]
Y_train = Y[train_indices]

widgets = [FormatLabel('Output training set: %(value)d/%(max)d matches. '), ETA(), ' ', Percentage(), ' ', Bar()]
pbar = ProgressBar(widgets = widgets, maxval = Y_train.size).start()

with open("./data/train.in", "wb") as f:
    f.write("{0} {1} 1\n".format(Y_train.size, NUM_FEATURES))
    for i in xrange(Y_train.size):
        f.write("{0}\n".format(" ".join(map(str, X_train[i]))))
        f.write("{0}\n".format(str(Y_train[i])))
        pbar.update(i)

pbar.finish()

widgets = [FormatLabel('Output test set: %(value)d/%(max)d matches. '), ETA(), ' ', Percentage(), ' ', Bar()]
pbar = ProgressBar(widgets = widgets, maxval = Y_train.size).start()

with open("./data/test.in", "wb") as f:
    f.write("{0} {1} 1\n".format(Y_test.size, NUM_FEATURES))
    for i in xrange(Y_test.size):
        f.write("{0}\n".format(" ".join(map(str, X_test[i]))))
        f.write("{0}\n".format(str(Y_test[i])))
        pbar.update(i)

pbar.finish()

# print "Saving output file now..."
# np.savez_compressed('test_%d.npz' % len(test_indices), X=X_test, Y=Y_test)
# np.savez_compressed('train_%d.npz' % len(train_indices), X=X_train, Y=Y_train)

# with open("train.in", "w") as train_f:
