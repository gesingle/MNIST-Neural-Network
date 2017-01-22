# Trains recurrant neural network on MNIST handwritten digit database
# @author Tariq Rashid
# @author Garrett Singletary

import numpy
from neuralnetwork import neuralNetwork

class mnistNN:

	# number of input, hidden and output nodes
	input_nodes = 784
	hidden_nodes = 150
	output_nodes = 10

	# learning rate
	learning_rate = 0.3

	# create instance of neural network
	n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

	# load the mnist training data into a list
	training_data_file = open("mnist_train.txt", 'r')
	training_data_list = training_data_file.readlines()
	training_data_file.close()

	# train the neural network

	# go through all record in the training data set
	for record in training_data_list:
		# split the values by comma
		all_values = record.split(',')
		# scale and shift the inputs
		inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.1
		# create the target output values (all 0.01 except for the desired
		# label which is 0.99)
		targets = numpy.zeros(output_nodes) + 0.01
		# all_values[0] is the target label for this record
		targets[int(all_values[0])] = 0.99
		n.train(inputs, targets)
		pass

	# load the minist test data
	test_data_file = open("mnist_test.txt", 'r')
	test_data_list = test_data_file.readlines()
	test_data_file.close()

	# test the neural network

	# scorecard for how well the nn performs, initally empty
	scorecard = []

	# go through all the records in the test data set
	for record in test_data_list:
		# split the values by comma
		all_values = record.split(',')
		# correct answer is the first value
		correct_label = int(all_values[0])
		# scale and shift the inputs
		inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.1
		# query the network
		outputs = n.query(inputs)
		# the indes of the highest value corresponds to the label
		label = numpy.argmax(outputs)
		# append correct or incorrect answer to list
		if (label == correct_label):
			# network's answer matches correct answer, add 1 to scorecard
			scorecard.append(1)
		else:
			# network's answer doesn't match correct answer, add 0 to scorecard
			scorecard.append(0)
		pass

	# calculate the performance score (fraction of correct answers)
	scorecard_array = numpy.asarray(scorecard)
	print ("performance = ", scorecard_array.sum() / scorecard_array.size)




