import json
import sys
import numpy as np
from constants import *

heroes = json.load(open("./data/heroes.json", "r"))

heroes_d = {}
ids_d = {}

for h in heroes:
    heroes_d[h["localized_name"]] = h["id"] - 1
    ids_d[h["id"] - 1] = h["localized_name"]

def is_valid_match(record):
    '''Returns True if the given match details result should be considered,
    and False otherwise.'''
    # No one left before the game ends
    for player in record['players']:
        if player['leaver_status'] is not 0:
            return False
    # Game is all pick, captain's mode, random draft, single draft,
    # all random, captain's draft or least played.
    # Filters off a lot of diretide games.
    if record['game_mode'] not in [1, 2, 3, 4, 5, 12, 16]:
        return False
    # Game is unranked, ranked, team match or tournament.
    if record['lobby_type'] not in [0, 2, 5, 7]:
        return False
    return True

def names_to_feature(names):
    assert len(names) == 10
    shift = 0
    appeared = set()
    X = [0.0 for _ in xrange(NUM_FEATURES)]
    for i in xrange(10):
        assert names[i] in heroes_d, "\"{0}\" not found".format(names[i])
        assert names[i] not in appeared, "\"{0}\" appeared more than once".format(names[i])
        appeared.add(names[i])
        if i >= 5:
            shift = NUM_HEROES
        X[heroes_d[names[i]] + shift] = 1.0
    return X

def feature_to_names(feature):
    feature = np.array(feature)
    indices = np.where(feature)[0]
    assert len(indices) == 10
    indices[5:] -= NUM_HEROES
    return map(ids_d.get, indices)

def push_to_int(y):
    assert 0 <= y <= 1
    return 0 if y < 0.5 else 1

def int_to_side(i):
    return "Radiant" if i else "Dire"
