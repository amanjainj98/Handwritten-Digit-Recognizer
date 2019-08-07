import numpy as np

class FullyConnectedLayer:
	def __init__(self, in_nodes, out_nodes):
		# Method to initialize a Fully Connected Layer
		# Parameters
		# in_nodes - number of input nodes of this layer
		# out_nodes - number of output nodes of this layer
		self.in_nodes = in_nodes
		self.out_nodes = out_nodes
		# Stores the outgoing summation of weights * feautres 
		self.data = None

		# Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
		self.weights = np.random.normal(0,0.1,(in_nodes, out_nodes))	
		self.biases = np.random.normal(0,0.1, (1, out_nodes))
		###############################################
		# NOTE: You must NOT change the above code but you can add extra variables if necessary 

	def forwardpass(self, X):
		# print('Forward FC ',self.weights.shape)
		# Input
		# activations : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_nodes]
		# OUTPUT activation matrix		:[n X self.out_nodes]

		###############################################
		# TASK 1 - YOUR CODE HERE
		self.data = np.matmul(X,self.weights) + self.biases
		return sigmoid(self.data)
		# raise NotImplementedError
		###############################################
		
	def backwardpass(self, lr, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		# Update self.weights and self.biases for this layer by backpropagation
		n = activation_prev.shape[0] # batch size

		###############################################
		# TASK 2 - YOUR CODE HERE
		ds = derivative_sigmoid(self.data)
		# print(self.weights.shape)

		new_delta =  np.matmul(np.multiply(ds,delta),np.transpose(self.weights))

		self.weights -= lr*np.matmul(np.transpose(activation_prev),np.multiply(ds,delta))
		self.biases -= lr*np.multiply(ds,delta).sum(axis=0)

		
		return new_delta
		# raise NotImplementedError
		###############################################

class ConvolutionLayer:
	def __init__(self, in_channels, filter_size, numfilters, stride):
		# Method to initialize a Convolution Layer
		# Parameters
		# in_channels - list of 3 elements denoting size of input for convolution layer
		# filter_size - list of 2 elements denoting size of kernel weights for convolution layer
		# numfilters  - number of feature maps (denoting output depth)
		# stride	  - stride to used during convolution forward pass
		self.in_depth, self.in_row, self.in_col = in_channels
		self.filter_row, self.filter_col = filter_size
		self.stride = stride

		self.out_depth = numfilters
		self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
		self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

		# Stores the outgoing summation of weights * feautres 
		self.data = None
		
		# Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
		self.weights = np.random.normal(0,0.1, (self.out_depth, self.in_depth, self.filter_row, self.filter_col))	
		self.biases = np.random.normal(0,0.1,self.out_depth)
		

	def forwardpass(self, X):
		# print('Forward CN ',self.weights.shape)
		# Input
		# X : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_depth X self.in_row X self.in_col]
		# OUTPUT activation matrix		:[n X self.out_depth X self.out_row X self.out_col]

		###############################################
		# TASK 1 - YOUR CODE HERE
		# print(X.shape)
		output = np.zeros((n,self.out_depth,self.out_row,self.out_col))
		data = np.zeros((n,self.out_depth,self.out_row,self.out_col))

		for i in range(self.out_row):
			for j in range(self.out_col):
				for d in range(self.out_depth):
					for b in range(n):

						x1 = i*self.stride
						x2 = i*self.stride+self.filter_row
						y1 = j*self.stride
						y2 = j*self.stride+self.filter_col

						data[b,d,i,j] = np.sum(np.multiply(self.weights[d],X[b,:,x1:x2,y1:y2])) + self.biases[d]
						output[b,d,i,j] = sigmoid(data[b,d,i,j])

		self.data = data
		return output
		# raise NotImplementedError
		###############################################

	def backwardpass(self, lr, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		# Update self.weights and self.biases for this layer by backpropagation
		n = activation_prev.shape[0] # batch size

		###############################################
		new_delta = np.zeros((n , self.in_depth , self.in_row , self.in_col))
		ds = derivative_sigmoid(self.data)

		# for b in range(n):
		# 	for d in range(self.in_depth):
		# 		for x in range(self.in_row):
		# 			for y in range(self.in_col):
		# 				for d1 in range(self.out_depth):
		# 					for i in range((x-self.filter_row)//self.stride , x//self.stride):
		# 						for j in range((x-self.filter_row)//self.stride , x//self.stride):
		# 							new_delta[b,d,x,y] += self.weights[d1,d,x-i*self.stride,y-j*self.stride] * delta[b,d1,i,j] * ds[b,d1,i,j]

		
		updated_weights = self.weights
		updated_biases = self.biases

		
		for i in range(self.out_row):
			for j in range(self.out_col):
				for d in range(self.out_depth):
					for b in range(n):
					
						x1 = i*self.stride
						x2 = i*self.stride+self.filter_row
						y1 = j*self.stride
						y2 = j*self.stride+self.filter_col

						new_delta[b,:,x1:x2,y1:y2] += ds[b,d,i,j] * delta[b,d,i,j] * self.weights[d]

						updated_weights[d] -= lr * ds[b,d,i,j] * delta[b,d,i,j] * activation_prev[b,:,x1:x2,y1:y2]
						updated_biases[d] -= lr * ds[b,d,i,j] * delta[b,d,i,j]

		

		self.weights = updated_weights
		self.biases = updated_biases
		
		return new_delta
		# TASK 2 - YOUR CODE HERE
		###############################################
	
class AvgPoolingLayer:
	def __init__(self, in_channels, filter_size, stride):
		# Method to initialize a Convolution Layer
		# Parameters
		# in_channels - list of 3 elements denoting size of input for max_pooling layer
		# filter_size - list of 2 elements denoting size of kernel weights for convolution layer

		# NOTE: Here we assume filter_size = stride
		# And we will ensure self.filter_size[0] = self.filter_size[1]
		self.in_depth, self.in_row, self.in_col = in_channels
		self.filter_row, self.filter_col = filter_size
		self.stride = stride

		self.out_depth = self.in_depth
		self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
		self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

	def forwardpass(self, X):
		# print('Forward MP ')
		# Input
		# X : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_depth X self.in_row X self.in_col]
		# OUTPUT activation matrix		:[n X self.out_depth X self.out_row X self.out_col]

		###############################################
		# TASK 1 - YOUR CODE HERE

		output = np.zeros((n , self.out_depth , self.out_row , self.out_col))


		for i in range(self.out_row):
			for j in range(self.out_col):
				for d in range(self.out_depth):
					for b in range(n):

						x1 = i*self.stride
						x2 = i*self.stride+self.filter_row
						y1 = j*self.stride
						y2 = j*self.stride+self.filter_col

						output[b,d,i,j] += np.average(X[b,d,x1:x2,y1:y2])


		return output



		# raise NotImplementedError
		###############################################


	def backwardpass(self, alpha, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# activations_curr : Activations of current layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		n = activation_prev.shape[0] # batch size

		###############################################
		# TASK 2 - YOUR CODE HERE
		new_delta = np.zeros((n , self.in_depth , self.in_row , self.in_col))
		
		for i in range(self.out_row):
			for j in range(self.out_col):
				for d in range(self.out_depth):
					for b in range(n):

						x1 = i*self.stride
						x2 = i*self.stride+self.filter_row
						y1 = j*self.stride
						y2 = j*self.stride+self.filter_col

						new_delta[b,d,x1:x2,y1:y2] += (self.filter_col*self.filter_row)*delta[b,d,i,j]

		
		return new_delta
		# raise NotImplementedError
		###############################################


# Helper layer to insert between convolution and fully connected layers
class FlattenLayer:
    def __init__(self):
        pass
    
    def forwardpass(self, X):
        self.in_batch, self.r, self.c, self.k = X.shape
        return X.reshape(self.in_batch, self.r * self.c * self.k)

    def backwardpass(self, lr, activation_prev, delta):
        return delta.reshape(self.in_batch, self.r, self.c, self.k)


# Helper Function for the activation and its derivative
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
	return sigmoid(x) * (1 - sigmoid(x))
