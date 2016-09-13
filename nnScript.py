import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import time
import json

def initializeWeights(n_in,n_out):
	"""
	# initializeWeights return the random weights for Neural Network given the
	# number of node in the input layer and output layer

	# Input:
	# n_in: number of nodes of the input layer
	# n_out: number of nodes of the output layer
	   
	# Output: 
	# W: matrix of random initial weights with size (n_out x (n_in + 1))"""

	epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
	W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
	return W
	
	
	
def sigmoid(z):
	
	"""# Notice that z can be a scalar, a vector or a matrix
	# return the sigmoid of input z"""
	
	return  (1.0/(1.0 + np.exp(-1.0*z)))
	

def preprocess():
	""" Input:
	 Although this function doesn't have any input, you are required to load
	 the MNIST data set from file 'mnist_all.mat'.

	 Output:
	 train_data: matrix of training set. Each row of train_data contains 
	   feature vector of a image
	 train_label: vector of label corresponding to each image in the training
	   set
	 validation_data: matrix of training set. Each row of validation_data 
	   contains feature vector of a image
	 validation_label: vector of label corresponding to each image in the 
	   training set
	 test_data: matrix of training set. Each row of test_data contains 
	   feature vector of a image
	 test_label: vector of label corresponding to each image in the testing
	   set

	 Some suggestions for preprocessing step:
	 - divide the original data set to training, validation and testing set
		   with corresponding labels
	 - convert original data set from integer to double by using double()
		   function
	 - normalize the data to [0, 1]
	 - feature selection"""
	
	mat = loadmat('mnist_all.mat') #loads the MAT object as a Dictionary
	
	#Pick a reasonable size for validation data
	data = np.empty((0,784))
	label = np.array([])
	test_data = np.empty((0,784))
	test_label = np.array([])

	#Your code here
	for i in range(10):
		m = mat.get('train'+str(i))
		A = mat.get('test'+str(i))
		data = np.concatenate((data,m),axis=0)
		label = np.concatenate((label,np.array([i]*m.shape[0])),axis=0)
		test_data = np.concatenate((test_data,A),axis=0)
                test_label = np.concatenate((test_label,np.array([i]*A.shape[0])),axis=0)
        
        #normalizing
        data = data/255.0
        test_data = test_data/255.0
        
        #unnecessory feature removal
        deleteIndices = []
        for x in range(data.shape[1]):
                if((data[:,x] == data[0,x])).all():
                        deleteIndices += [x]
        data = np.delete(data,deleteIndices,axis=1)
        test_data = np.delete(test_data,deleteIndices,axis=1)
        
        #dividing data into test_data and validation_data
        a = range(data.shape[0])
	aperm = np.random.permutation(a)	
	train_data = np.array(data[aperm[0:50000],:])
	train_label = np.array(label[aperm[0:50000]])
       	validation_data = np.array(data[aperm[50000:],:])
	validation_label = np.array(label[aperm[50000:]])
	  	  
	print(train_label.shape,validation_label.shape,test_label.shape)

	return train_data, train_label, validation_data, validation_label, test_data, test_label
	
	
	

def nnObjFunction(params, *args):
	"""% nnObjFunction computes the value of objective function (negative log 
	%   likelihood error function with regularization) given the parameters 
	%   of Neural Networks, thetraining data, their corresponding training 
	%   labels and lambda - regularization hyper-parameter.

	% Input:
	% params: vector of weights of 2 matrices w1 (weights of connections from
	%	 input layer to hidden layer) and w2 (weights of connections from
	%	 hidden layer to output layer) where all of the weights are contained
	%	 in a single vector.
	% n_input: number of node in input layer (not include the bias node)
	% n_hidden: number of node in hidden layer (not include the bias node)
	% n_class: number of node in output layer (number of classes in
	%	 classification problem
	% training_data: matrix of training data. Each row of this matrix
	%	 represents the feature vector of a particular image
	% training_label: the vector of truth label of training images. Each entry
	%	 in the vector represents the truth label of its corresponding image.
	% lambda: regularization hyper-parameter. This value is used for fixing the
	%	 overfitting problem.
	   
	% Output: 
	% obj_val: a scalar value representing value of error function
	% obj_grad: a SINGLE vector of gradient value of error function
	% NOTE: how to compute obj_grad
	% Use backpropagation algorithm to compute the gradient of error function
	% for each weights in weight matrices.

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% reshape 'params' vector into 2 matrices of weight w1 and w2
	% w1: matrix of weights of connections from input layer to hidden layers.
	%	 w1(i, j) represents the weight of connection from unit j in input 
	%	 layer to unit i in hidden layer.
	% w2: matrix of weights of connections from hidden layer to output layers.
	%	 w2(i, j) represents the weight of connection from unit j in hidden 
	%	 layer to unit i in output layer."""
	
	n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
	
	biasval = 1
	w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
	w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1))) 
	bias = np.ones(training_data.shape[0])*biasval
	training_data = np.column_stack((training_data,bias))
	
	row_index = np.arange(training_label.shape[0])
	y = np.zeros((training_label.shape[0],10))
	y[row_index,training_label.astype(int)] = 1
	
	z = sigmoid(np.dot(training_data,w1.T))
	o = sigmoid(np.dot(np.column_stack((z,bias)),w2.T))
	dl = (y - o)*(1 - o)*o
	grad_w2 = -1.0 * np.dot(dl.T,np.column_stack((z,bias)))
	grad_w1 = -1.0 * np.dot(((1 - z) * z * np.dot(dl,np.delete(w2,n_hidden,axis=1))).T,training_data)
	obj_grad = np.append(grad_w1,grad_w2)
	obj_grad = (obj_grad + lambdaval*params)/training_data.shape[0]
	
	obj_val = np.sum((y - o)**2)/training_data.shape[0]*0.5 + lambdaval*0.5/training_data.shape[0]*(np.sum(np.square(w1)) + np.sum(np.square(w2)))
	print(obj_val)
	#Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
	#you would use code similar to the one below to create a flat array
	#obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)

	
	return obj_val, obj_grad



def nnPredict(w1,w2,data):
	
	"""% nnPredict predicts the label of data given the parameter w1, w2 of Neural
	% Network.

	% Input:
	% w1: matrix of weights of connections from input layer to hidden layers.
	%	 w1(i, j) represents the weight of connection from unit j in input 
	%	 layer to unit i in hidden layer.
	% w2: matrix of weights of connections from hidden layer to output layers.
	%	 w2(i, j) represents the weight of connection from unit i in output 
	%	 layer to unit j in hidden layer.
	% data: matrix of data. Each row of this matrix represents the feature 
	%	   vector of a particular image
	   
	% Output: 
	% label: a column vector of predicted labels""" 
	
	
	
	labels = np.zeros((data.shape[0], w2.shape[0]))
	z = np.zeros((data.shape[0],w1.shape[0]))
	o = np.zeros((data.shape[0],w2.shape[0]))
	
	biasval = 1
	bias = np.ones(data.shape[0])*biasval
	data = np.column_stack((data,bias))
	z = sigmoid(np.dot(data,w1.T))
	o = sigmoid(np.dot(np.column_stack((z,bias.T)),w2.T))
	labels = o.argmax(axis=1)
	print(labels.shape)
	'''for n in range(data.shape[0]):
		label = np.zeros(w2.shape[0])
		for i in range(w1.shape[0]):
			total = 0
			for j in range(w1.shape[1] - 1):
				total += data[n][j]*w1[i][j]
			z[i] = sigmoid(total+bias*w1[i][n_input])

		for i in range(w2.shape[0]):
			total = 0
			for j in range(w2.shape[1] - 1):
				total += z[j]*w2[i][j]
			o[i] = sigmoid(total+bias*w1[i][n_hidden])
		label[np.argmax(o[n])] = 1
		labels[n] = label'''

	return labels  
	



"""**************Neural Network Script Starts here********************************"""

outEntry = {}

preprocess_starttime = time.time()
train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();
	#backward propogation
	#backward propogation
outEntry['preprocess_time'] = time.time() - preprocess_starttime

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]; 
outEntry['n_input_nodes'] = n_input
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 80;
outEntry['n_hidden_nodes'] = n_hidden				   
# set the number of nodes in output unit
n_class = 10;				   

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

predicted_label = nnPredict(initial_w1,initial_w2,train_data)
init_acc = 100*np.mean((predicted_label == train_label).astype(float))
print('\n Training set Accuracy:' + str(init_acc) + '%')

# set the regularization hyper-parameter
lambdaval = 0.4;
outEntry['lambdaval'] = lambdaval

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' : 100}	# Preferred value.
training_starttime = time.time()
nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
outEntry['training_time'] = time.time() - training_starttime

#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))


#Test the computed parameters
prediction_test_starttime = time.time()
predicted_label = nnPredict(w1,w2,train_data)
outEntry['prediction_test_time'] = time.time() - prediction_test_starttime

#find the accuracy on Training Dataset
outEntry['training_accuracy'] = 100*np.mean((predicted_label == train_label).astype(float))
print('\n Training set Accuracy:' + str(outEntry['training_accuracy']) + '%')


prediction_validation_starttime = time.time()
predicted_label = nnPredict(w1,w2,validation_data)
outEntry['prediction_validation_time'] = time.time() - prediction_validation_starttime

#find the accuracy on Validation Dataset
outEntry['validation_accuracy'] = 100*np.mean((predicted_label == validation_label).astype(float))
print('\n Validation set Accuracy:' + str(outEntry['validation_accuracy']) + '%')

prediction_test_starttime = time.time()
predicted_label = nnPredict(w1,w2,test_data)
outEntry['prediction_test_time'] = time.time() - prediction_test_starttime

#find the accuracy on test Dataset
outEntry['test_accuracy'] = 100*np.mean((predicted_label == test_label).astype(float))
print('\n Test set Accuracy:' + str(outEntry['test_accuracy']) + '%')


with open('test_results.json','a') as logfile:
	json.dump(outEntry,logfile,indent = 4)
	logfile.write(',')
