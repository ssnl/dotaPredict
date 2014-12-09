from pyfann import *

NUM_HEROES = 108
NUM_FEATURES = NUM_HEROES * 2

connection_rate = 0.8
learning_rate = 0.7
desired_error = 0.1
max_iterations = 100
iterations_between_reports = 1

ann = libfann.neural_net()
ann.create_sparse_array(connection_rate, (NUM_FEATURES, NUM_FEATURES + 100, 20, 10, 1))
ann.set_activation_function_output(libfann.SIGMOID_SYMMETRIC)
ann.set_learning_rate(learning_rate)

ann.train_on_file("./data/train.in", max_iterations, iterations_between_reports, desired_error)

ann.save("./data/train.net")
