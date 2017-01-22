import numpy
import matplotlib
import scipy.special


class neuralNetwork:

	# initialize the neural network
	def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
		# set number of nodes in each input, hidden, and output layer
		self.inodes = inputnodes
		self.hnodes = hiddennodes
		self.onodes = outputnodes

		# link weight matrices
		# weights inside the arrays are w_i_j where link is from node i to node j in next layer
		self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
		self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

		# learning rate
		self.lr = learningrate
		pass

		# activation function is the sigmoid function
		self.activation_function = lambda x: scipy.special.expit(x)

	#train the neural network
	def train(self, inputs_list, targets_list):
		# convert inputs list to 2d array
		inputs = numpy.array(inputs_list, ndmin=2).T
		targets = numpy.array(targets_list, ndmin=2).T

		# calculate signals into hidden layer
		hidden_inputs = numpy.dot(self.wih, inputs)
		# calculate the signals emerging from the hidden layer
		hidden_outputs = self.activation_function(hidden_inputs)

		# calculate signals into final output layer
		final_inputs = numpy.dot(self.who, hidden_outputs)
		# calculate the signals emerging from final output layer
		final_outputs = self.activation_function(final_inputs)

		# output layer error is the (target - actual)
		output_errors = targets - final_outputs
		# hidden layer error is the outout_errors split by weights and then 
		# recombined at hidden nodes
		hidden_errors = numpy.dot(self.who.T, output_errors)

		# update weights for links between the hidden and output layers
		self.who += self.lr * numpy.dot((output_errors * final_outputs 
			* (1.0 - final_outputs)), numpy.transpose(hidden_outputs))

		# update weights for the links between the input and hidden layers
		self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs 
			* (1.0 - hidden_outputs)), numpy.transpose(inputs))

		return final_outputs 

	# query the neural network 
	def query(self, inputs_list):
		# convert inputs list to 2d array
		inputs = numpy.array(inputs_list, ndmin=2).T

		# calculate signals into hidden layer
		hidden_inputs = numpy.dot(self.wih, inputs)
		# calculate the signals emerging from the hidden layer
		hidden_outputs = self.activation_function(hidden_inputs)
		#calculate signals into final output layer
		final_inputs = numpy.dot(self.who, hidden_outputs)
		# calculate the signals emerging fom the final output layer
		final_outputs = self.activation_function(final_inputs)

		return final_outputs

