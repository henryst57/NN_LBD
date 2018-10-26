# THIS IS A SEMI-HARD CODED PROGRAM
# FOR THE PURPOSE OF CREATING RESULTS
# FROM THE EVALUATION OF TEST DATA


# clean up later to match trainNN.py





import os, errno
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import keras.backend as K
#from tensorflow.python import keras
import keras
from keras.models import Model, model_from_json, load_model
from keras import optimizers
from keras.utils.vis_utils import plot_model
from keras.layers import Dense, Activation, Input, concatenate, Dropout, Embedding, Flatten
import h5py
import sys
import re
import gc
import time




#model stuff
model = None
model_file = "../../../data/trained_nn_model.h5"
model_weights_file = "../../../data/trained_nn_model_weights.h5"

#evaluated file
#eval_file = "../../../data/mini_true"
eval_file = "../../../data/mini_known"

#key files
pred_key_file = "../../../data/known.predicate_key"
cui_key_file = "../../../data/known.cui_key"

# config stuff
learning_rate              = 0.001
momentum                   = 0.9
steps                      = 0
batch_size                 = 1


cui_occurence_data_length = 0
number_of_cuis = 0
number_of_predicates = 0


#############      FUNCTIONS       ################


#loads the model
def LoadModel():
	customs = {
		'BCE': BCE,
		'CCE': CCE,
		'Precision': Precision,
		'Recall' : Recall,
		'Matthews_Correlation' : Matthews_Correlation
		};
	model = load_model(model_file, 
		custom_objects=customs)

	model.load_weights(model_weights_file)
	print("Loaded the model from %s" % model_file)

	return model


#imports the lines for the eval file
def LoadEvalFile():
	eval_data = []

	#import the file for testing
	try:
		with open( eval_file, "r" ) as in_file:
			eval_data = in_file.readlines()
	except FileNotFoundError:
		print ( "GetConceptUniqueIdentifierData() - Error: Unable To Open File \"" + str( eval_file )+ "\"" )
		return -1
	finally:
		in_file.close()

	eval_data = [ line.strip() for line in eval_data ]	# Removes Trailing Space Characters From CUI Data Strings
	eval_data.sort()

	eval_len = len( eval_data )

	return eval_data, eval_len


#loads the unique keys from the 
def LoadUniqKeys():
	# original output:
	#	index id
	#   1 C00023820 	(cui)
	#   1 AFFECTS 		(pred)
	cui_uniq = {}
	pred_uniq = {}


	#read in cui key file
	try:
		with open( cui_key_file, "r" ) as cui_in_file:
			for line in cui_in_file:
				(index, cuiVal) = line.split()
				cui_uniq[str(cuiVal)] = int(index)
	except FileNotFoundError:
		print ( "LoadUniqKeys() - Error: Unable To Open File \"" + str( cui_key_file )+ "\"" )
		return -1
	finally:
		cui_in_file.close()

	#read in pred key file
	try:
		with open( pred_key_file, "r" ) as pred_in_file:
			for line in pred_in_file:
				(index, predVal) = line.split()
				pred_uniq[str(predVal)] = int(index)
	except FileNotFoundError:
		print ( "LoadUniqKeys() - Error: Unable To Open File \"" + str( pred_key_file )+ "\"" )
		return -1
	finally:
		pred_in_file.close()

	return cui_uniq, pred_uniq



#   Parses Through CUI Data And Generates Sparse Matrices For
#   Concept Input, Predicate Input and Concept Output Data
def GenerateNetworkMatrices():
	global cui_occurence_data_length
	global steps
	global number_of_cuis
	global number_of_predicates

	print_input_matrices = 0


	cui_occurence_data, cui_occurence_data_length = LoadEvalFile()
	unique_cui_data, unique_predicate_data = LoadUniqKeys()
	number_of_cuis = len(unique_cui_data)
	number_of_predicates = len(unique_predicate_data)

	# Sets The Number Of Steps If Not Specified
	if( steps == 0 ):
		print( "LoadUniqKeys() - Warning: Number Of Steps Not Specified / Generating Value Based On Data Size" )
		steps = int( cui_occurence_data_length / batch_size ) + ( 0 if( cui_occurence_data_length % batch_size == 0 ) else 1 )
		print( "LoadUniqKeys() - Number Of Steps: " + str( steps ) )


	
	print("Number of CUIS = " + str(number_of_cuis))
	print("Number of Predicates = " + str(number_of_predicates))

	# Check(s)
	if( cui_occurence_data_length == 0 ):
		print( "GenerateNetworkMatrices() - Error: No CUI Data In Memory / Was An Input CUI File Read Before Calling Method?" )
		return None, None, None

	print( "GenerateNetworkMatrices() - Generating Network Matrices" )

	concept_input_indices   = []
	predicate_input_indices = []
	concept_output_indices  = []
	concept_output_values   = []
	output_cui_count        = 0

	# Parses each line of raw data input and adds them to arrays for matrix generation
	print( "GenerateNetworkMatrices() - Parsing CUI Data / Generating Network Input-Output Data Arrays" )
	for i in range( cui_occurence_data_length ):
		not_found_flag = False
		line = cui_occurence_data[i]
		line_elements = re.split( r"\s+", line )

		#print("LEN:" +str(len(line_elements)))

		# Check(s)
		# If Subject CUI, Predicate and Object CUIs Are Not Found Within The Specified Unique CUI and Predicate Lists, Report Error and Skip The Line
		if( line_elements[0] not in unique_cui_data ):
			print( "GenerateNetworkMatrices() - Error: Subject CUI \"" + str( line_elements[0] ) + "\" Is Not In Unique CUI Data List" )
			not_found_flag = True
		if( line_elements[1] not in unique_predicate_data ):
			print( "GenerateNetworkMatrices() - Error: Predicate \"" + str( line_elements[1] ) + "\" Is Not In Unique Predicate Data List" )
			not_found_flag = True
		for element in line_elements[2:]:
			if( element not in unique_cui_data ):
				print( "GenerateNetworkMatrices() - Error: Object CUI \"" + str( element ) + "\" Is Not In Unique CUI Data List" )
				not_found_flag = True

		if( not_found_flag is True ): 
			continue

		# Add Unique Element Indices To Concept/Predicate Input List
		concept_input_indices.append( [i, unique_cui_data[ line_elements[0] ]] )
		predicate_input_indices.append( [i, unique_predicate_data[ line_elements[1] ]] )

		# Adds all Object CUI indices to the output CUI index array
		for element in line_elements[2:]:
			concept_output_indices.append( [i, unique_cui_data[element]] )
			concept_output_values.append( unique_cui_data[element] )
			output_cui_count += 1


	# Check(s)
	if( len( concept_input_indices ) is 0 ):
		print( "GenerateNetworkMatrices() - Error: Concept Input Indices List Contains No Data / Specified Subject CUIs In The Input Not File Not Within Unique CUI Data?" )
	if( len( predicate_input_indices ) is 0 ):
		print( "GenerateNetworkMatrices() - Error: Predicate Input Indices List Contains No Data / Specified Predicates In The Input Not File Not Within Unique CUI Data?" )
		print( "GenerateNetworkMatrices() -        Note: This May Be Reporting Due To Concept Input Indices List Erroring Out" )
	if( len( concept_output_indices ) is 0 ):
		print( "GenerateNetworkMatrices() - Error: Concept Output Indices List Contains No Data / Specified Object CUIs In The Input Not File Not Within Unique CUI Data?" )
		print( "GenerateNetworkMatrices() -        Note: This May Be Reporting Due To Concept Input or Predicate Indices List Erroring Out" )
	if( len( concept_input_indices ) is 0 or len( predicate_input_indices ) is 0 or len( concept_output_indices ) is 0 ): return None, None, None

	# Transpose the arrays, then convert to rows/columns
	print( "GenerateNetworkMatrices() - Transposing Index Data Arrays Into Row/Column Data" )
	concept_input_row,   concept_input_column   = zip( *concept_input_indices   )
	predicate_input_row, predicate_input_column = zip( *predicate_input_indices )
	concept_output_row,  concept_output_column  = zip( *concept_output_indices  )

	# Convert row/column data into sparse matrices
	print( "GenerateNetworkMatrices() - Converting Index Data Into Matrices" )
	concept_input_matrix   = sparse.csr_matrix( ( [1]*cui_occurence_data_length,    ( concept_input_row,   concept_input_column ) ),   shape = ( cui_occurence_data_length, number_of_cuis ) )
	predicate_input_matrix = sparse.csr_matrix( ( [1]*cui_occurence_data_length,    ( predicate_input_row, predicate_input_column ) ), shape = ( cui_occurence_data_length, number_of_predicates ) )
	concept_output_matrix  = sparse.csr_matrix( ( [1]*output_cui_count, ( concept_output_row,  concept_output_column ) ),  shape = ( cui_occurence_data_length, number_of_cuis ) )

	if( print_input_matrices is 1 ):
		print( "Compressed Sparse Matrix - Subject CUIs" )
		print( concept_input_matrix )
		print( "Original Dense Formatted Sparse Matrix" )
		print( concept_input_matrix.todense() )
		print( "Compressed Sparse Matrix - Predicates" )
		print( predicate_input_matrix )
		print( "Original Dense Formatted Sparse Matrix" )
		print( predicate_input_matrix.todense() )
		print( "Compressed Sparse Matrix - Object CUIs" )
		print( concept_output_matrix )
		print( "Original Dense Formatted Sparse Matrix" )
		print( concept_output_matrix.todense() )

	print( "GenerateNetworkMatrices() - Complete" )

	return concept_input_matrix, predicate_input_matrix, concept_output_matrix




########################
# # # EVALUMATION # # #
########################


#evaluates the data on the loaded model
def Evaluate(concept_input, predicate_input, concept_output, metricSet):
	 # Check(s)
	if( concept_input is None ):
		print( "Evaluate() - Error: Concept Input Contains No Data" )
	if( predicate_input is None ):
		print( "Evaluate() - Error: Predicate Input Contains No Data" )
	if( concept_output is None ):
		print( "Evaluate() - Error: Concept Output Contains No Data" )
	if( concept_input is None or predicate_input is None or concept_output is None ):
		return None

	model = LoadModel()

	#get the model together
	sgd = optimizers.SGD( lr = learning_rate, momentum = momentum )
	model.compile(loss = BCE, optimizer = sgd, metrics = metricSet)
	eval_metrics = model.evaluate_generator(generator = Batch_Gen([concept_input, predicate_input], concept_output, batch_size = batch_size ), steps = steps, verbose=1)



#   Fixes The Input For A Sparse Matrix
def Batch_Gen( X, Y, batch_size ):
	samples_per_epoch = cui_occurence_data_length
	number_of_batches = samples_per_epoch / batch_size      # determines how many batches based on the batch_size specified and the size of the dataset
	counter = 0

	# Get Randomly Shuffled Data From The Sparse Matrix
	shuffle_index = np.arange( np.shape( Y )[0] )                   # Where To Start The Random Index
	np.random.shuffle( shuffle_index )
	for x2 in X:                        # 2 parts - cui_in and pred_in
		x2 = x2[shuffle_index, :]
	Y = Y[shuffle_index, :]             # matching Y output

	# Shuffle Until The Epoch Is Finished
	while 1:
		index_batch = shuffle_index[ batch_size * counter : batch_size * ( counter + 1 ) ]
		X_batches = []
		for x2 in X:
			X_batches.append( x2[index_batch, :].todense() )    # unpack the matrix so it can be read into the NN
		y_batch = Y[index_batch, :].todense()                   # unpack the output
		counter += 1

		yield( X_batches, np.asarray( y_batch ) )               # feed into the neural network

		if( counter < number_of_batches ):                      # shuffle again
			np.random.shuffle( shuffle_index )
			counter = 0



#####################      METRIC EVALUATION    #########################



#   Custom Binary Cross Entropy With Logits Option
def BCE( y_pred, y_true ):
  return K.binary_crossentropy( y_pred, y_true, from_logits = True )

#   Custom Categorical Cross Entropy With Logits Option
def CCE( y_pred, y_true ):
  return K.categorical_crossentropy( y_pred, y_true, from_logits = True )

#   Custom Metric for Precision
""" Precision metric.

	Only computes a batch-wise average of precision.

	Computes the precision, a metric for multi-label classification of
	how many selected items are relevant.
	"""
def Precision( y_true, y_pred ):
	true_positives = K.sum( K.round( K.clip( y_true * y_pred, 0, 1 ) ) )
	predicted_positives = K.sum( K.round( K.clip( y_pred, 0, 1 ) ) )
	precision = true_positives / ( predicted_positives + K.epsilon() )
	return precision

#   Custom Metric For Recall
""" Recall metric.

	Only computes a batch-wise average of recall.

	Computes the recall, a metric for multi-label classification of
	how many relevant items are selected.
	"""
def Recall( y_true, y_pred ):
	true_positives = K.sum( K.round( K.clip( y_true * y_pred, 0, 1 ) ) )
	possible_positives = K.sum( K.round( K.clip( y_true, 0, 1 ) ) )
	recall = true_positives / ( possible_positives + K.epsilon() )
	return recall

#   Matthews Correlation - Custom Metric. Code from:
#   https://stackoverflow.com/questions/39895742/matthews-correlation-coefficient-with-keras
def Matthews_Correlation( y_true, y_pred ):
	y_pred_pos = K.round( K.clip( y_pred, 0, 1 ) )
	y_pred_neg = 1 - y_pred_pos

	y_pos = K.round( K.clip( y_true, 0, 1 ) )
	y_neg = 1 - y_pos

	tp = K.sum( y_pos * y_pred_pos )
	tn = K.sum( y_neg * y_pred_neg )

	fp = K.sum( y_neg * y_pred_pos )
	fn = K.sum( y_pos * y_pred_neg )

	numerator = ( tp * tn - fp * fn )
	denominator = K.sqrt( ( tp + fp ) * ( tp + fn ) * ( tn + fp ) * ( tn + fn ) )

	return numerator / ( denominator + K.epsilon() )





#########################
# # #  GOOD STUFFS  # # #
#########################
print("Startin' Evaluatin'....\n")

#LoadModel()

cui_input, predicate_input, cui_output = GenerateNetworkMatrices()

'''
print(cui_input)
print("\n\n")
print(predicate_input)
print("\n\n")
print(cui_output)
print("\n\n")
'''

#print ("\n\n\n%s - %s\n\n\n\n" % (cui_occurence_data_length, number_of_cuis))

if( cui_input != None and predicate_input != None and cui_output != None ):
	Evaluate( cui_input, predicate_input, cui_output, ['accuracy', Precision, Recall, Matthews_Correlation])

print("YEEHAW!")