from pymongo import MongoClient

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
NETWORK_TEMP_FILE_NAME = './data/network_temp.xml'
NETWORK_VAL_FILE_NAME = './data/network_val.save'
ADDITIONAL_EPOCH = 20
MAX_EPOCH = 100
VALIDATION_RATIO = 0.1
TEST_RATIO = 0.1