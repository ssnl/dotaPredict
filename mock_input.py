import json
import sys
import tempfile
import subprocess


NUM_HEROES = 108
NUM_FEATURES = NUM_HEROES * 2

heroes = json.load(open("./heroes.json", "r"))

heroes_d = {}

for h in heroes:
    heroes_d[h["localized_name"]] = h["id"] - 1

if __name__ == "__main__":
    args = sys.argv[1:]
    assert len(args) == 10
    shift = 0
    appeared = set()
    X = [0.0 for _ in xrange(NUM_FEATURES)]
    for i in xrange(10):
        assert args[i] in heroes_d
        assert args[i] not in appeared
        appeared.add(args[i])
        if i >= 5:
            shift = 108
        X[heroes_d[args[i]] + shift] = 1.0
    with tempfile.NamedTemporaryFile() as f:
        f.write(",".join(map(str, X)) + "\n")
        f.flush()
        out = subprocess.check_output(["./mock", f.name])
    print out
    out = float(out)
    if out > 0.5:
        print "Radiant probably got this!"
    elif out > 0:
        print "Hard to say, but Radiant has some advantage."
    elif out > -0.5:
        print "Hard to say, but Dire has some advantage."
    else:
        print "Dire probably got this!"

