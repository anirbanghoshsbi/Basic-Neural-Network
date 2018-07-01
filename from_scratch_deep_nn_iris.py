#multi class flower classification

#usage :python deep_nn_iris.py -e 100

#import the necessary packages
import numpy as np
from csv import reader
import argparse

# load the data



def load_csv(filename):
	dataset = []
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup

#compute the nonlinearity function : sigmoid and softmax for classification

def softmaxx(x):
	e_x=np.exp(x-np.max(x))
	return e_x/e_x.sum(axis=0)

def sigmoid(x):
	return (1.0/(1+np.exp(-x)))
def d_sigmoid(x):
	return sigmoid(x)*(1-sigmoid(x))

#calculate accuracy ( I will be adding function for predictions and then calculate the accuracy later)
def accuracy_metric(actual , predicted):
	correct=0.0
	for i in range(len(actual)):
		if actual[i]==predicted[i]:
			correct+=1
	return correct/float(len(actual))*100		

#main program :
seed= 7
# create the agrument parser and get the inputs
ap=argparse.ArgumentParser()
ap.add_argument('-e' , '--epochs' , type= float,required = 'True' , help = 'No of epochs')
ap.add_argument('-lr' , '--learning_rate' , type=float, default =0.01 , help = 'learning rate default is 0.01')
args = vars(ap.parse_args())


#load the data
filename = 'iris.csv'
dataset = load_csv(filename)

# converting string to floats
for i in range(4):
	str_column_to_float(dataset, i)
# convert class column to int
lookup = str_column_to_int(dataset, 4)

# convert the dataset to numpy array	
dataset=np.array(dataset)
X= dataset[:,:-1]
y=dataset[:,-1]
y=y.reshape((y.shape[0],1))

input_dims = 4 # equal to the number of input features
hidden_dims = 4
output_dims = 3 # equal to the output of the NN here that is 3 'iris-setosa','iris-versicolor','iris-virginica'.
W1 = np.random.randn(input_dims , hidden_dims) 
W2 =np.random.randn(hidden_dims , output_dims)
# we neglect the bias for now however bias can be added by bias = np.zeors((1, output_dims)) 
# Run the Program

for i in np.arange(args['epochs']):
	#take the inner product
	Neural_layer1=X.dot(W1)
	# bring in the non linearity
	Neural_layer_1_nonlin=sigmoid(Neural_layer1)
	
	#layer two
	Neural_layer2= Neural_layer_1_nonlin.dot(W2)
	
	Neural_layer_2_nonlin=sigmoid(Neural_layer2)
	MSE= np.square(Neural_layer2 -y)/len(Neural_layer_2_nonlin[0])
	print('loss in the epoch {} is {}' .format(i+1 , MSE.sum()))


        # SGD-step 1
	gradient_2_part_1 = (Neural_layer_2_nonlin - y)
	gradient_2_part_2 = d_sigmoid(Neural_layer2)
	gradient_2_part_3 = Neural_layer_1_nonlin
	gradient_2= gradient_2_part_3.T.dot(gradient_2_part_1*gradient_2_part_2)
	# SGD - stochastic Gradient Descent-step2
	gradient_1_part_1 = (gradient_2_part_1*gradient_2_part_2).dot(W2.T)
	gradient_1_part_2 = d_sigmoid(Neural_layer1)
	gradient_1_part_3 = X
	gradient_1 = gradient_1_part_3.T.dot(gradient_1_part_1*gradient_1_part_2)
	
	# weight update
	W1 -= args['learning_rate']*gradient_1
	W2 -= args['learning_rate']*gradient_2

#recalulate the inputs based on the changes in the weights

Neural_layer1= X.dot(W1)
Neural_layer_1_nonlin = sigmoid(Neural_layer1)

Neural_layer2=Neural_layer_1_nonlin.dot(W2)
Neural_layer_2_nonlin = softmaxx(Neural_layer2)

# print the results 
print('final output' , Neural_layer_2_nonlin[:,-1])
print('final output on rounding' , W2[:,-1])
print('actual labels' , y[:,-1])
print('W2',W2[:,-1])
print(Neural_layer_2_nonlin)


