#Program by Milk
#CUI-Neural-net-Discovery-Approximator
# mini CaNaDAy
# Stripped down version of CNDA
# does not have the import and one-hot vector maker

version = 0.1

#import some stuffs
import os, errno
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import keras.backend as K
from keras.models import Model
from keras import optimizers
from keras.layers import Dense, Activation, Input, concatenate, Dropout
import sys
import re
import gc
import time
from fractions import gcd

#network data
# most of the variables will be set through the configuration file specified
config_file = ""

config_dat = {}

"""
# GET VALUES FROM setConfig()

NUM_PREDS = 3     #3
NUM_CUIS = 1      
LAYER1_SIZE = 200  #200
LAYER2_SIZE = 400   #400
LEARNING_RATE = 0.1
NUM_EPOCHS = 5
STEPS = 1
BATCH_SIZE = 10
MOMENTUM = 0.9
DROPOUT_AMT = 0.25
FLAGS=None
"""


#weight stuff - not actually used
use_weight = True
positives = 0
negatives = 0
weights = {
	0 : 1.,
	1 : 1.
}


#color output for errors when you really done goofed
BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)
def printColor(text, color=WHITE):
	seq = "\x1b[1;%dm" % (30+color) + text + "\x1b[0m"
	sys.stdout.write(seq)

# sets the data values for the configuration file
# must be run before initializing the neural network
# inputs:
#	file - the name of the file
#			data is in the format: 
#			<param>value ---- <layer3># of weights
# output:
#	config_dat - a dictionary with the parameters as the key and the value as the value
# 	ex/ 	config_dat["LAYER1_SIZE"] = 200
def setConfig(c_file):
	global config_dat

	#open the file
	try:
		config_set = open(c_file, "r")		#open the file
	except FileNotFoundError:
		printColor("ERROR!  File: '" + c_file + "' not found! Check the directory and spelling of the filename\n", RED)
	
	#extract the data
	conf_lines = config_set.read_lines()
	conf_lines = [s.strip() for s in conf_lines]

	#use regular expression matching on each line to look for the specific format
	for l in conf_lines:
		matchSet = re.match(r'<([a-zA-Z0-9]+)>([0-9]+)', l)
		config_set[matchSet.group(1)] = matchSet.group(2)


# data2nodes will be the oneHotVec set from the classify.pl program
"""
#breaks up the data and makes them into vectors and then into a 2d array
def data2Nodes():

	#prep the matrices
	global cui_data
	global cui_list
	global pred_data

	#prep the pos, neg, and class weights
	global positives
	global negatives
	global weights

	#this is gonna be a 2d array defined for the whole thing
	cui_in_indexes=[]
	pred_indexes=[]
	cui_out_indexes=[]

	cui_out_ct=0
	cui_out_val=[]

	#convert each line of the raw data input
	for i in range(len(cui_data)):
		line=cui_data[i]
		all_dat=re.split(r'\s+', line)

		#add the index of the input cuis and predicates found in the line to the array
		cui_in_indexes.append([i, cui_list[all_dat[0]]])
		pred_indexes.append([i, pred_data[all_dat[1]]])

		#add the index of the output cuis to the array
		for c in all_dat[2:]:
			cui_out_indexes.append([i, cui_list[c]])
			cui_out_val.append(cui_list[c])				#add to the output set
			cui_out_ct+=1								

		output_ct = len(all_dat) -2				#<--- i have no idea what this is

		positives += output_ct					#add the size of the output list
		negatives += (NUM_CUIS-output_ct)		#add the rest

	#create the class weights
	if(use_weight):
		factor = gcd(positives, negatives)
		
		weights = {
			0: positives/factor,
			1: negatives/factor
		}

		print("Weight ratio: +: " + str(positives) + " -: " + str(negatives))
		print("Factored: 0: " + str(weights[0]) + " 1: " + str(weights[1]))
		

	#transpose and convert to rows and columns
	cui_in_row, cui_in_col = zip(*cui_in_indexes)
	pred_in_row, pred_in_col = zip(*pred_indexes)
	cui_out_row, cui_out_col = zip(*cui_out_indexes)

	#make into sparse matrices
	input_cuis = sparse.csr_matrix(([1]*len(cui_data), (cui_in_row, cui_in_col)), shape=(len(cui_data), NUM_CUIS))
	input_preds = sparse.csr_matrix(([1]*len(cui_data), (pred_in_row, pred_in_col)), shape=(len(cui_data), NUM_PREDS))
	output_cuis = sparse.csr_matrix(([1]*cui_out_ct, (cui_out_row, cui_out_col)), shape=(len(cui_data), NUM_CUIS))

	'''
	#get the ones that weren't mentioned in the output
	odds = list(set(range(NUM_CUIS))-set(cui_out_val))
	print(range(NUM_CUIS))
	print(list(set(cui_out_val)))
	print(odds)
	'''

	#print(weights)

	return input_cuis, input_preds, output_cuis
"""


#Step 1: take input, train and test
#Step 2: ???
#Step 3: Profit
def process_nn(input_cuis, input_preds, output_cuis):
	global pred_array

	#parse the inputs and ready for the neural net
	printColor("----VECTORIZING----\n", GREEN)
	#input_cuis, input_preds, output_cuis = data2Nodes()

	#get the one-hot vectors from classify.pl


	#print(input_cuis.todense())
	#print(output_cuis.todense())

	#create the base neural net
	printColor("----TRAINING----\n", GREEN)
	start_time = time.time()
	untrained_nn = neural_net(input_cuis, input_preds, output_cuis, False)

	#run the untrained neural net to get the base stats
	base_accuracy = untrained_nn.evaluate_generator(generator=batch_gen([input_cuis, input_preds], output_cuis, batch_size=BATCH_SIZE), steps=STEPS)
	print("Base [Loss, Accuracy, Precision, Recall]: %s" % base_accuracy)

	#train the neural net
	#trained_nn = test_montreal(input_cuis, input_preds, output_cuis, True)
	trained_nn = neural_net(input_cuis, input_preds, output_cuis, True)
	printColor("***** TIME TO TRAIN: %s seconds *****\n" % (time.time() - start_time), BLUE)

	# Score accuracy
	printColor("----PREDICTING ~ %s v. %s----\n" % (test_in_cui, test_out_cui), GREEN)

		
	#format the prediction data
	predict_cui_df = pd.DataFrame(columns=(sorted(cui_list)))
	for i in range(NUM_PREDS):
		predict_cui_df.loc[i] = cui_one_hot(test_in_cui)

	predict_pred_df = pd.DataFrame(columns=pred_array)
	for i in range(NUM_PREDS):
		predict_pred_df.loc[i] = pred_one_hot(pred_array[i])
		
	'''
	predict_cui_df = pd.DataFrame(columns=(["cui1", "cui2", "cui3", "cui4", "cui5"]))
	for i in range(3):
		predict_cui_df.loc[i] = np.array([0,0,0,0,1])
	predict_pred_df = pd.DataFrame(columns=["a", "b", "c"])
	predict_pred_df.loc[0] = np.array([1,0,0])
	predict_pred_df.loc[1] = np.array([0,1,0])
	predict_pred_df.loc[2] = np.array([0,0,1])
	'''


	#make predictions
	predictions = trained_nn.predict(x={"cui_input": predict_cui_df, "pred_input":predict_pred_df}, verbose=1)
	for i in range(len(predictions)):
		if(show_predict):
			print("%s: %s" % (test_in_cui + " " + pred_array[i] + " " + test_out_cui, predictions[i]))
		output_predictions(i, predictions[i])
		rankPredictions(i)
		#print("Prediction %s: %s" % (i + 1, predictions[i]))
	


"""
	Neural network creation
	
	Does that neural network thing or something
	
	Parameters
	-----------
	cui_in : scipy.sparse matrix
		Sparse matrix of all of the one hot vector sets for the input cuis
	pred_in : scipy.sparse matrix
		Sparse matrix of all of the one hot vector sets for the input predicates 
	cui_out : scipy.sparse matrix
		Sparse matrix of all of the one hot vector sets for the output cuis
	train_me : boolean
		Train the neural network or return it immediately after it is created

	Returns
	--------
	Model
		The neural network model
"""
def one_hot_neural_net(num_cuis, num_preds, cui_in, pred_in, train_me):

	#build the layers as designed from the customization dictionary
	cui_in_layer = Input(shape=(num_cuis, ), dtype='float32', name='cui_input')
	concept_layer = Dense(units=config_set['LAYER1_SIZE'], activation='relu', input_dim=num_cuis, name='concept_rep')[cui_in_layer]
	pred_in_layer = Input(shape=(num_preds, ), dtype='float32', name='pred_input')										#predicate input layer

	know_in = concatenate([concept_layer, pred_in_layer])																#concatenate the predicate layer to the cui layer
	know_layer = Dense(units=config_set["LAYER2_SIZE"], activation='relu', input_dim=NUM_PREDS)(know_in)								#knowledge representation layer

	#dropper = Dropout(config_set["DROPOUT"])(know_layer)
	#cui_out_layer = Dense(units=num_cuis, activation='sigmoid', name='cui_output')(dropper)

	cui_out_layer = Dense(units=num_cuis, activation='sigmoid', name='cui_output')(know_layer)

	#create the optimizers and metrics for the output 
	sgd = optimizers.SGD(lr=config_set["LEARNING_RATE"], momentum=config_set["MOMENTUM"])
	model.compile(loss=bce, optimizer=sgd, metrics=['accuracy', precision, recall])

	#model.fit_generator(generator=batch_gen([cui_in, pred_in], cui_out.toarray(), BATCH_SIZE), steps_per_epoch=STEPS, epochs=NUM_EPOCHS, shuffle=False)
	
	#train the model on the inputs
	if(train_me):
		model.fit_generator(generator=batch_gen([cui_in, pred_in], cui_out, config_set["BATCH_SIZE"]), steps_per_epoch=config_set["STEPS"], epochs=config_set["NUM_EPOCHS"], shuffle=False)
		#model.fit_generator(generator=batch_gen([cui_in, pred_in], cui_out, BATCH_SIZE), steps_per_epoch=STEPS, class_weight=weights, epochs=NUM_EPOCHS, shuffle=False)

	return model


#custom binary cross entropy with logits option
def bce(y_pred, y_true):
	return K.binary_crossentropy(y_pred, y_true, from_logits=True)

#custom categorical cross entropy with logits option
def cce(y_pred, y_true):
	return K.categorical_crossentropy(y_pred, y_true, from_logits=True)

#fixes the input for a sparse matrix
def batch_gen(X, y, batch_size):
	samples_per_epoch = len(cui_data)
	number_of_batches = samples_per_epoch/batch_size 		#determines how many batches based on the batch_size specified and the size of the dataset
	counter=0

	#get randomly shuffled data from the sparse matrix
	shuffle_index=np.arange(np.shape(y)[0])					#where to start the random index
	np.random.shuffle(shuffle_index)
	for x2 in X:						#2 parts - cui_in and pred_in
		x2 = x2[shuffle_index, :]			
	y = y[shuffle_index, :]				#matching y output

	#shuffle until the epoch is finished
	while 1:
		index_batch = shuffle_index[batch_size*counter:batch_size*(counter+1)]
		X_batches = []
		for x2 in X:
			X_batches.append(x2[index_batch,:].todense())		#unpack the matrix so it can be read into the NN
		y_batch = y[index_batch,:].todense()					#unpack the output
		counter += 1

		yield(X_batches,np.asarray(y_batch))					#feed into the neural network
		#print(X_batches)
		#print(np.asarray(y_batch))
		if(counter < number_of_batches):						#shuffle again
			np.random.shuffle(shuffle_index)
			counter=0

#custom metric for precision
def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

#custom metric for recall
def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

#prints the predictions to a file
def output_predictions(i, predictions):
	#print("------PRINTING PREDICTIONS ~ " + pred_array[i] +"-------")

	#make a new directory
	try:
		os.makedirs("predict_data/" + file_prefix)
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise

	#make a new file
	out_file = open("predict_data/" + file_prefix + "/" + file_prefix + "_" + pred_array[i], "w")
	for p in range(len(predictions)):
		out_str = "%s, %s\n" % (round(predictions[p], 4), cui_array[p])
		out_file.write(out_str)
	out_file.close()
	

#sort the predictions based on confidence level and find the cui value ranking
def rankPredictions(i):
	pred_file = open("predict_data/" + file_prefix + "/" + file_prefix + "_" + pred_array[i], "r")
	pred_set=pred_file.readlines()        #retrieve the data
	pred_set.sort(reverse=True)                     #sort the lines

	#make a new directory
	try:
		os.makedirs("predict_data/" + file_prefix + "/sorted")
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise

	#make a new file
	out_file = open("predict_data/" + file_prefix + "/sorted/" + file_prefix + "_" + pred_array[i] + "_sorted", "w")
	
	#print out the sorted data
	for line in pred_set:
		out_file.write(line);
	out_file.write("\n")


	#print out the prediction value with the cui associated with it
	for i2 in range(len(pred_set)):
		if test_out_cui in pred_set[i2]:
			print("%s\t%d" % (pred_array[i], i2+1))
			break

	#close everything up
	pred_file.close()
	out_file.close()


############		LEGIT PART OF THE CODE			##############

#print the version
print("~ RUNNING CNDA ~")
print("~ Version " + str(version) + " ~")

#check if not enough arguments
if((len(sys.argv) >= 1) and (len(sys.argv) <= 4)):
	printColor("[ERROR]: Not enough arguments! Check if missing [filename, cui input, cui output, batch size]\n", RED)
	exit(1)

#user command line arguments handling
elif(len(sys.argv) > 1):
	config_file = str(sys.argv[1])
	cui_in_one_hot_file = str(sys.argv[2])
	pred_in_one_hot_file = str(sys.argv[3])
	cui_out_one_hot_file = str(sys.argv[4])

	printColor("INPUT ARGUMENTS:\n", YELLOW)
	print("* Config File           : 			" + config_file)
	print("* CUI input 1-hot File  : 			" + cui_in_one_hot_file)
	print("* PRED input 1-hot File : 			" + pred_in_one_hot_file)
	print("* CUI output 1-hot File : 			" + cui_out_one_hot_file)
	print("\n")



#set up the configuration file
printColor("----SETTING UP CONFIG---\n", YELLOW)
setConfig(sys.argv[2])

#read the cui + pred 1-hot files in

#run the neural network
printColor("----RUNNING NEURAL NETWORK----\n", GREEN)
process_nn()
