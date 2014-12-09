import json
import sys
import numpy as np

NUM_HEROES = 108
NUM_FEATURES = NUM_HEROES * 2

heroes = json.load(open("./data/heroes.json", "r"))

heroes_d = {}
ids_d = {}

for h in heroes:
    heroes_d[h["localized_name"]] = h["id"] - 1
    ids_d[h["id"] - 1] = h["localized_name"]

def names_to_feature(names):
    assert len(names) == 10
    shift = 0
    appeared = set()
    X = [0.0 for _ in xrange(NUM_FEATURES)]
    for i in xrange(10):
        assert args[i] in heroes_d
        assert args[i] not in appeared
        appeared.add(args[i])
        if i >= 5:
            shift = NUM_HEROES
        X[heroes_d[args[i]] + shift] = 1.0
    return X

def feature_to_names(feature):
    feature = np.array(feature)
    indices = np.where(feature)[0]
    assert len(indices) == 10
    indices[5:] -= NUM_HEROES
    return map(ids_d.get, indices)

def push_to_int(y):
    return 0 if y < 0.5 else 1
