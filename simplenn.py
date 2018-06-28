# usage : python simplenn.py -e 100

# import the libraries

import numpy as np
import argparse

# create the agrument parser and get the inputs
ap=argparse.ArgumentParser()
ap.add_argument('-e' , '--epochs' , type= float,required = 'True' , help = 'No of epochs')
ap.add_argument('-lr' , '--learning_rate' , type=float, default =0.01 , help = 'learning rate default is 0.01')
args = vars(ap.parse_args())

#compute the nonlinearity function : sigmoid

def sigmoid(x):
	return (1.0/(1+np.exp(-x)))
def d_sigmoid(x):
	return sigmoid(x)*(1-sigmoid(x))

#

# loss formula there are many  but here we wouls focus on mean squared error
'''
If yhat predictions generated from a sample of n data points on all variables, and  Y is the vector of observed values 
of the variable being predicted, then the within-sample MSE of the predictor is computed asMSE = 1/n*(sum(yi-yihat)**2
hence the MSE is the mean of the square of the difference between original observed value and the prediction of the ML
algorithm. taking square of the difference between the actual and the predicted value penalizes the network for error 
in quadratic term or more heavily, thus forcing to the algorithm to be more accurate in its predicitions before it is 
used for real life purpose and can be thought of like and regularizer preventing overfitting.'''

MSE =0.0

#since we have just the input layer and the output layer without any hidden layer this the most simple architecture.

# generate a input data

X = np.array([[1,2,3],[2,1,2],[2,1,1],[2,3,4]])
y=np.array([[1],[0],[1],[1]])

'''
the shape of the data is X.shape = (4,3) as we have four rows and each row having three features.
the output y is of the shape y.shape= (4,1)'''

input_dims = 3 # equal to the number of input features
output_dims = 1 # equal to the output of the NN.
W1 = np.random.randn(input_dims , output_dims)  
# we neglect the bias for now however bias can be added by bias = np.zeors((1, output_dims)) 
# Run the Program

for i in np.arange(args['epochs']):
	#take the inner product
	Neural_layer1=X.dot(W1)
	# bring in the non linearity
	Neural_layer_1_nonlin=sigmoid(Neural_layer1)
	MSE= np.square(Neural_layer1 -y)/len(Neural_layer_1_nonlin)
	print('loss in the epoch {} is {}' .format(i+1 , MSE.sum()))

	# SGD - stochastic Gradient Descent
	gradient_1_part_1 = (Neural_layer_1_nonlin - y)/ len(Neural_layer_1_nonlin)
	gradient_1_part_2 = d_sigmoid(Neural_layer1)
	gradient_1_part_3 = X
	gradient_1 = gradient_1_part_3.T.dot(gradient_1_part_1*gradient_1_part_2)
	
	# weight update
	W1 -= args['learning_rate']*gradient_1

#recalulate the inputs based on the changes in the weights

Neural_layer1= X.dot(W1)
Neural_layer_1_nonlin = sigmoid(Neural_layer1)

# print the results 
print('final output' , Neural_layer_1_nonlin[:,-1])
print('final output on rounding' , W1[:,-1])
print('actual labels' , y[:,-1])
print('W1',W1[:,-1])


