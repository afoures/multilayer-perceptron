import random
import numpy as np
import math
import json
import sys
import matplotlib.pyplot as plt
import time


""" A bunch of activation function and their derivative"""
class	Sigmoid:
	@staticmethod
	def function(z):
		return 1 / (1 + np.exp(-z))

	@staticmethod
	def derivative(z):
		return Sigmoid.function(z) * (1 - Sigmoid.function(z))

class	Tanh:
	@staticmethod
	def function(z):
		return (2 / (1 + np.exp(-2 * z))) - 1

	@staticmethod
	def derivative(z):
		return 1 - Tanh.function(z)**2

class	SoftPlus:
	@staticmethod
	def function(z):
		return np.log(1 + np.exp(z))

	@staticmethod
	def derivative(z):
		return 1 / (1 + np.exp(-z))

class	Softmax:
	""" Softmax function will "squashes" the outputs in a vector, where each
	entry is in the range (0, 1] and all the entries add up to 1."""
	@staticmethod
	def function(z):
		return np.exp(z) / np.sum(np.exp(z), axis=0)

	@staticmethod
	def derivative(z):
		s = z.reshape(-1,1)
		return np.diagflat(s) - np.dot(s, s.T)

class	ReLU:
	@staticmethod
	def function(z):
		return z * (z > 0)

	@staticmethod
	def derivative(z):
		return 1. * (z > 0)

class	PReLU:
	alpha = 0.01
	@staticmethod
	def function(z):
		return z * (z > 0) + (PReLU.alpha * z) * (z <= 0)

	@staticmethod
	def derivative(z):
		return 1. * (z > 0) + PReLU.alpha * (z <= 0)

class	ELU:
	alpha = 0.01
	@staticmethod
	def function(z):
		return z * (z > 0) + (ELU.alpha * (np.exp(z) - 1)) * (z <= 0)

	@staticmethod
	def derivative(z):
		return 1. * (z > 0) + (ELU.alpha + ELU.function(z)) * (z <= 0)


""" A bunch of loss function"""
def	rCrossEntropyLoss(a, y, lmbda, weights, l):
	""" In the case of a=1.0 and y=1.0, (1-y)*np.log(1-a) returns nan.
	We use numpy.nan_to_num() to convert the nan if necessary"""
	regularization = 0.5 * (lmbda / l) * sum(np.linalg.norm(w) ** 2
											for w in weights)	
	return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a))) + regularization

def	crossEntropyLoss(a, y, lmbda, weigths, l):
	""" In the case of a=1.0 and y=1.0, (1-y)*np.log(1-a) returns nan.
	We use numpy.nan_to_num() to convert the nan if necessary"""
	return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

def	rQuadraticLoss(a, y, lmbda, weigths, l):
	regularization = 0.5 * (lmbda / l) * sum(np.linalg.norm(w) ** 2
											for w in weights)	
	return 0.5 * np.linalg.norm(a - y) ** 2 + regularization

def	quadraticLoss(a, y, lmbda, weigths, l):
	return 0.5 * np.linalg.norm(a - y) ** 2


def	loadMLP(filename):
	""" Take a json file as imput and use it to create a multilayer perceptron
	with the saved parameters"""
	try:
		f = open(filename, "r")
		data = json.load(f)
		f.close()
		loss = getattr(sys.modules[__name__], data['loss'])
		lmbda = getattr(sys.modules[__name__], data['lmbda'])
		""" Care here with the eval() function,
		use it only if you are sure about the imput"""
		layers = [(x, eval(y)) for x, y in data['layers']]
		mlp = MultilayerPerceptron(layers,
								loss=loss,
								lmbda=lmbda)
		mlp.weights = [np.array(w) for w in data['weights']]
		mlp.biases = [np.array(b) for b in data['biases']]
	except:
		print(f'>>> Couldn\'t load data from {filename}')
	else:
		print(f'>>> Multilayer perceptron fully loaded from {filename}')
		return mlp


class	MultilayerPerceptron:
	def __init__(self, layers,
				loss=crossEntropyLoss,
				lmbda=0.0):

		""" 'layers' is a list of tuples that represents the layers of the MLP
		where each tuples must follow this pattern :
		('nb_of_neurons', 'activation_class')
		'lmbda' is used for the regularization"""
		self.loss = loss
		self.layers = layers
		self.layers_size = [x for x, _ in layers]
		self.size = len(self.layers)
		self.lmbda = lmbda

		self.activation = [y for x, y in layers]

		""" Use numpy.random.seed()"""
		np.random.seed(1)
		
		""" Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron. Initialize the biases
        using a Gaussian distribution with mean 0 and standard deviation 1."""
		# self.weights = [np.random.randn(y, x)/np.sqrt(x)
		# 				for x, y in zip(self.layers_size[:-1], self.layers_size[1:])]
		# self.biases = [np.random.randn(y, 1) for y in self.layers_size[1:]]

		""" Initialize weights we he_uniform"""
		limit = np.sqrt(6 / self.layers_size[0])
		self.weights = [np.random.uniform(-limit, limit, (y, x))
						for x, y in zip(self.layers_size[:-1],
										self.layers_size[1:])]
		self.biases = [np.random.uniform(-limit, limit, (y, 1))
						for y in self.layers_size[1:]]

	def	feedforward(self, a):
		""" Use feedforward implementation to return the value 
		computed by the multilayer perceptron with 'a' as imput"""
		for i, (b, w) in enumerate(zip(self.biases, self.weights)):
			a = self.activation[i + 1].function(np.dot(w, a) + b)
		return a

	def	predict(self, dataset, show_debrief=True):
		acc = 0.0
		loss = 0.0
		l = len(dataset)
		if show_debrief:
			print(f">>> Starting prediction")
		for i, (x, y) in enumerate(dataset):
			a = self.feedforward(x)
			loss += self.loss(a, y, self.lmbda, self.weights, l)/l
			output = np.argmax(a)
			if y[output] == 1:
				acc += 1
			if show_debrief:
				print(f'> ({output:.0f}, {np.argmax(y):.0f}) -> '
					+ f'raw {a.transpose().tolist()[0]}')
		if show_debrief:
			print(f'>>> Binary cross entropy loss = {loss}')
			print(f'>>> Prediction is done, accuracy = '
					+ f'{acc*100/l:.2f}% ({acc:.0f}/{l})')
		return acc*100/l

	""" Stochastic/minibatch gradient descent"""
	def	gradientDescent(self, training_data, epochs, learning_rate,
						batch_size=1,
						regularization=False,
						evaluation_data=None,
						show_loss=False):

		print(f">>> Starting training with epochs={epochs}, "
				+ f"learning_rate={learning_rate}, "
				+ f"show_loss={show_loss}")
		start = time.clock()
		random.seed(1)
		
		self.training_loss, self.evaluation_loss = [], []
		len_data = len(training_data)
		for epoch in range(epochs):
			start_epoch = time.clock()
			random.shuffle(training_data)
			batches = [training_data[k:k+batch_size]
						for k in range(0, len_data, batch_size)]
			tmp_b = self.biases
			tmp_w = self.weights
			for batch in batches:
				delta_b = [np.zeros(b.shape) for b in self.biases]
				delta_w = [np.zeros(w.shape) for w in self.weights]
				for x, y in batch:
					gradient_b, gradient_w = self.backpropagation(x, y)
					delta_b = np.add(delta_b, gradient_b)
					delta_w = np.add(delta_w, gradient_w)
				if regularization is True:
					self.weights = ((1-(learning_rate*self.lmbda/len_data))
									* np.array(self.weights)
									- (learning_rate / len(batch)) * delta_w)
				else:
					self.weights -= (learning_rate / len(batch)) * delta_w
				self.biases -= (learning_rate / len(batch)) * delta_b
			loss = self.getLoss(training_data)
			self.training_loss.append(loss)
			if evaluation_data is not None:
				t_loss = self.getLoss(evaluation_data)
				self.evaluation_loss.append(t_loss)
				acc = self.predict(evaluation_data, show_debrief=False)
				print(f'> Epoch {epoch}/{epochs} - train_loss {loss:.3f} '
						+ f'- test_loss {t_loss:.3f} - accuracy {acc:.3f}% '
						+ f'- time {time.clock() - start_epoch:.3f}')
			else:
				print(f'> Epoch {epoch}/{epochs} - train_loss {loss:.3f} '
						+ f'- time {time.clock() - start_epoch:.3f}')

		print(f'>>> Training done in {time.clock() - start:.3f}s')
		
		if show_loss:
			self.plotData(evaluation_data)

	def	backpropagation(self, x, y):
		gradient_b = [np.zeros(b.shape) for b in self.biases]
		gradient_w = [np.zeros(w.shape) for w in self.weights]
		
		""" feedforward"""
		a = x
		imputs = [np.array(x)]
		vectors = []
		for i, (b, w) in enumerate(zip(self.biases, self.weights)):
			a = np.dot(w, a) + b
			vectors.append(a)
			a = self.activation[i+1].function(a)
			imputs.append(a)
		
		""" backpropagation"""
		error = imputs[-1] - y
		for layer in range(len(self.layers_size[:-1]) - 1, -1, -1):
			delta = error * self.activation[layer].derivative(vectors[layer])
			gradient_b[layer] = delta
			gradient_w[layer] = np.dot(delta, np.array(imputs[layer]).transpose())
			if layer > 0:
				error = np.dot(self.weights[layer].transpose(), delta)
		
		return gradient_b, gradient_w

	def	getLoss(self, data):
		loss = 0.0
		l = len(data)
		for x, y in data:
			a = self.feedforward(x)
			loss += self.loss(a, y, self.lmbda, self.weights, l)/l
		return loss
		
	def	plotData(self, evaluation_data):
		""" Display loss evolution"""
		if evaluation_data is not None:
			plt.plot(self.evaluation_loss, 'b--')
		plt.plot(self.training_loss, 'r-')
		plt.xlabel('epochs')
		plt.ylabel('loss')
		plt.show()

	def	save(self, filename):
		""" Save the MLP in a json file"""
		try:
			data = {'layers': [(size, activation.__name__)
								for size, activation in self.layers],
					'loss': str(self.loss.__name__),
					'weights': [w.tolist() for w in self.weights],
					'biases': [b.tolist() for b in self.biases],
					'lmbda': self.lmbda}
			f = open(filename, "w")
			json.dump(data, f)
			f.close()
		except:
			print(f'>>> Couldn\'t save data to {filename}')
		else:
			print(f'>>> Multilayer perceptron fully saved to {filename}')
