from pyfann import libfann


ann = libfann.neural_net()

ann.create_from_file("./data/train.net")

# test outcome
print "Testing network"
test_data = libfann.training_data()
test_data.read_train_from_file("./data/test.in")

ann.reset_MSE()
ann.test_data(test_data)
print "MSE error on test data: %f" % ann.get_MSE()