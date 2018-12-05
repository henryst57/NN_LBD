##################################


# NEURAL NETWORK EVALUATOR
# uses the model from trainNN.py to 
# to predict classes on a dataset

# REVISED: 12.3.18

#import the libraries
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
import os

#reference Clint's code
# make sure main() is not called or the whole program will run
# this reference only allows the functions and variables to be called using the same namespace
trainNN_path = os.environ.get('NN_PATH', '..') + "/trainNN"
print(trainNN_path)
sys.path.append(trainNN_path)
#sys.path.append('../trainNN')
import trainNN as trainNN
from trainNN import *


########################
#                      #
#   GLOBAL VARIABLES   #
#                      #
########################

#model                       = None
model_file                  = ""      #this may be set by the config later 
model_weights_file          = ""
eval_file                   = ""
pred_key_file               = ""
cui_key_file                = ""
eval_output_file            = ""

cui_occurence_data          = {}
cui_occurence_data_length   = 0
unique_cui_data             = {}
unique_predicate_data       = {} 
identified_cuis = []
identified_predicates = []
unidentified_cuis = []
unidentified_predicates = []



#   Reads The Specified Configuration File Parameters Into Memory
#   and Sets The Appropriate Variable Data
#
#   Based off of TrainNN.py's ReadConfigFile() but works as an extension
#   to grab relevant parameters necessary for the evaluation
def ReadConfigFile_EVAL( config_file_path ):
	global model_file
	global model_weights_file
	global eval_file
	global pred_key_file
	global cui_key_file
	global eval_output_file

	ReadConfigFile(config_file_path)

	# Check(s)
	if CheckIfFileExists( config_file_path ) == False:
		PrintLog( "ReadConfigFile_EVAL() - Error: Specified File \"" + str( config_file_path ) + "\" Does Not Exist" )
		return -1

	# Assuming The File Exists, Check If The File Has Data
	if os.stat( config_file_path ).st_size == 0:
		PrintLog( "ReadConfigFile_EVAL() - Error: Specified File \"" + str( config_file_path ) + "\" Is Empty File" )
		return -1

	# Open The File And Read Data, Line-By-Line
	try:
		f = open( config_file_path, "r" )
	except FileNotFoundError:
		PrintLog( "ReadConfigFile_EVAL() - Error: Specified File \"" + str( config_file_path ) + "\" Does Not Exist" )
		return -1

	f1 = f.readlines()

	# Set Network Parameters Using Config File Data
	for line in f1:
		line = re.sub( r'<|>|\n', '', line )
		data = line.split( ":" )

		if data[0] == "ModelFile"           : model_file            = str( data[1] )
		if data[0] == "ModelWeights"        : model_weights_file    = str( data[1] )
		if data[0] == "EvaluateFile"        : eval_file             = str( data[1] )
		if data[0] == "CUIKeyFile"          : cui_key_file          = str( data[1] )
		if data[0] == "PredKeyFile"         : pred_key_file         = str( data[1] )
		if data[0] == "EvalOut"             : eval_output_file      = str( data[1] )
		
	f.close()

	#grab the key files as well if not specified
	if(trainNN.train_file):
		if(not cui_key_file or cui_key_file == ""):
			PrintLog("ReadConfigFile_EVAL() - Warning: CUI Key File not set - generating path from train_file")
			cui_key_file = (trainNN.train_file + ".cui_key")
		if(not pred_key_file or pred_key_file == ""):
			PrintLog("ReadConfigFile_EVAL() - Warning: Predicate Key File not set - generating path from from train_file")
			pred_key_file = (trainNN.train_file + ".predicate_key")
	elif(not trainNN.train_file and ((cui_key_file == "") or (pred_key_file == "")) ):
		PrintLog( "ReadConfigFile_EVAL() - Error: \"train_file\" Variable Not Set! Cannot create keys" )
		exit()

	#file existence checks
	if(not model_file):
		PrintLog("ReadConfigFile_EVAL() - Error: \"model_file\" variable not set!")
		exit()
	elif(not model_weights_file):	
		PrintLog("ReadConfigFile_EVAL() - Error: \"model_weights_file\" variable not set!")
		exit()
	elif(not eval_file):
		PrintLog("ReadConfigFile_EVAL() - Error: \"eval_file\" variable not set!")
		exit()

	if(not eval_output_file or (eval_output_file == "")):
		eval_output_file = "evaluation.out"

	OpenDebugFileHandle()

	# Print Settings To Console
	PrintLog( "=========================================================" )
	PrintLog( "-   Configuration File Settings - EVALUATE.py           -" )
	PrintLog( "=========================================================" )

	PrintLog( "    Evaluation Data File   : " + str( eval_file ) )
	PrintLog( "    Model File             : " + str( model_file ) )
	PrintLog( "    Model Weights File     : " + str( model_weights_file ) )
	PrintLog( "    CUI Key File           : " + str( cui_key_file ) )
	PrintLog( "    Predicate Key File     : " + str( pred_key_file ) )
	PrintLog( "    Eval Output File       : " + str( eval_output_file ) )
	
	PrintLog( "=========================================================" )
	PrintLog( "-                                                       -" )
	PrintLog( "=========================================================\n" )

	PrintLog( "ReadConfigFile() - Complete" )

	return 0





###########################       EVALUATION NN FUNCTIONS        #######################

#loads the model
def LoadModel():
	#set the custom objects based on TrainNN.py's delcared functions
	customs = {
		'BCE': BCE,
		'CCE': CCE,
		'Precision': Precision,
		'Recall' : Recall,
		'Matthews_Correlation' : Matthews_Correlation
		};

	#load the model from the specified file and with the objects
	model = load_model(model_file, 
		custom_objects=customs)

	#weights are needed as well
	model.load_weights(model_weights_file)
	PrintLog(("Loaded the model from " + model_file))

	#prints a summary of the model for debugging
	model.summary()

	#return the model back
	return model


#imports the lines for the eval file
def LoadEvalFile():
	global cui_occurence_data_length
	global cui_occurence_data


	#import the file for testing
	try:
		with open( eval_file, "r" ) as in_file:
			cui_occurence_data = in_file.readlines()
			if(len(cui_occurence_data) < 1):
				PrintLog(("LoadEvalFile() - Error: Evaluation file ["+ str(eval_file) +"] has no lines"))
	except FileNotFoundError:
		PrintLog( "LoadEvalFile() - Error: Unable To Open File \"" + str( eval_file )+ "\"" )
		return -1
	else:
		in_file.close()

	#strip and sort the data to save to a global variable
	cui_occurence_data = [ line.strip() for line in cui_occurence_data ]  # Removes Trailing Space Characters From CUI Data Strings
	cui_occurence_data.sort()

	#save the length to a global variable
	cui_occurence_data_length = len( cui_occurence_data )


#loads the unique keys from the 
def LoadUniqKeys():
	# original output:
	#   index id
	#   1 C00023820     (cui)
	#   1 AFFECTS       (pred)
	global unique_cui_data
	global unique_predicate_data



	#read in cui key file
	try:
		with open( cui_key_file, "r" ) as cui_in_file:
			clines = cui_in_file.readlines()
			#save to a hash function in the format
			#  ucd[CUI] = index
			if(len(clines) < 1):
				PrintLog("LoadUniqKeys() - Error: no lines found in cui_in_file " + cui_key_file)
				exit()

			for line in clines:
				(index, cuiVal) = line.split()
				unique_cui_data[str(cuiVal)] = int(index)
	except FileNotFoundError:
		PrintLog( "LoadUniqKeys() - Error: Unable To Open File \"" + str( cui_key_file )+ "\"" )
		return -1
	else:
		cui_in_file.close()

	#read in pred key file
	try:
		with open( pred_key_file, "r" ) as pred_in_file:
			plines = pred_in_file.readlines()
			#save to a hash function in the format
			#  ucd[CUI] = index
			if(len(plines) < 1):
				PrintLog("LoadUniqKeys() - Error: no lines found in pred_in_file " + pred_in_file)
				exit()

			for line in plines:
				(index, predVal) = line.split()
				unique_predicate_data[str(predVal)] = int(index)
	except FileNotFoundError:
		PrintLog( "LoadUniqKeys() - Error: Unable To Open File \"" + str( pred_key_file )+ "\"" )
		return -1
	else:
		pred_in_file.close()





#   Parses Through CUI Data And Generates Sparse Matrices For
#   Concept Input, Predicate Input and Concept Output Data

#   This GenerateNetworkMatrices() function generates the matrices based on
#   the already defined format and indexes of the list generated by TrainNN.py
#   to work with the Evaluation function
def GenerateNetworkMatrices():
	#redefine global variables locally
	global cui_occurence_data_length
	global steps
	global number_of_cuis
	global number_of_predicates
	global identified_cuis
	global unidentified_cuis
	global actual_train_data_length
	global cui_dense_input_mode
	global unique_cui_data
	global unique_predicate_data

	#debugger to print the matrices
	print_input_matrices = 0

	#load the keys and the list so the matrices can be made
	#based on how they were previously defined
	LoadEvalFile()
	LoadUniqKeys()

	#retrieve the number of unique training cuis and predicates
	number_of_cuis = len(unique_cui_data)
	number_of_predicates = len(unique_predicate_data)

	# Sets The Number Of Steps If Not Specified
	if( steps == 0 ):
		PrintLog( "LoadUniqKeys() - Warning: Number Of Steps Not Specified / Generating Value Based On Data Size" )
		steps = int( cui_occurence_data_length / trainNN.batch_size ) + ( 0 if( cui_occurence_data_length % trainNN.batch_size == 0 ) else 1 )
		PrintLog( "LoadUniqKeys() - Number Of Steps: " + str( steps ) )


	PrintLog("Number of CUIS = " + str(number_of_cuis))
	PrintLog("Number of Predicates = " + str(number_of_predicates))

	# Check(s)
	if( cui_occurence_data_length == 0 ):
		PrintLog( "GenerateNetworkMatrices() - Error: No CUI Data In Memory / Was An Input CUI File Read Before Calling Method?" )
		return None, None, None

	PrintLog( "GenerateNetworkMatrices() - Generating Network Matrices" )

	#initialize the values that will be populated later
	number_of_unique_cui_inputs       = 0
	number_of_unique_predicate_inputs = 0
	concept_input_indices   = []
	concept_input_values   = []
	predicate_input_indices = []
	predicate_input_values = []
	concept_output_indices  = []
	concept_output_values   = []
	output_cui_count        = 0

	# Parses each line of raw data input and adds them to arrays for matrix generation
	PrintLog( "GenerateNetworkMatrices() - Parsing CUI Data / Generating Network Input-Output Data Arrays" )
	
	index = 0
	number_of_skipped_lines = 0

	#go through the entire evaluation set
	for i in range( cui_occurence_data_length ):
		not_found_flag = False
		line = cui_occurence_data[i]
		line_elements = re.split( r"\s+", line )

		#PrintLog("LEN:" +str(len(line_elements)))

		# Check(s)
		# If Subject CUI, Predicate and Object CUIs Are Not Found Within The Specified Unique CUI and Predicate Lists, Report Error and Skip The Line
		if( line_elements[0] not in unique_cui_data ):
			PrintLog( "GenerateNetworkMatrices() - Error: Subject CUI \"" + str( line_elements[0] ) + "\" Is Not In Unique CUI Data List / Skipping Line " + str( i ) )
			not_found_flag = True
			if( line_elements[0] not in unidentified_cuis ): unidentified_cuis.append( line_elements[0] )
		else:
			if( line_elements[0] not in identified_cuis ):   identified_cuis.append( line_elements[0] )
		if( line_elements[1] not in unique_predicate_data ):
			PrintLog( "GenerateNetworkMatrices() - Error: Predicate \"" + str( line_elements[1] ) + "\" Is Not In Unique Predicate Data List / Skipping Line " + str( i ) )
			not_found_flag = True
			if( line_elements[1] not in unidentified_predicates ): unidentified_predicates.append( line_elements[1] )
		else:
			if( line_elements[1] not in identified_predicates ): identified_predicates.append( line_elements[1] )
		for element in line_elements[2:]:
			if( element not in unique_cui_data ):
				PrintLog( "GenerateNetworkMatrices() - Error: Object CUI \"" + str( element ) + "\" Is Not In Unique CUI Data List / Skipping Line " + str( i ) )
				not_found_flag = True
				if( element not in unidentified_cuis ): unidentified_cuis.append( element )
			else:
				if( element not in identified_cuis ):   identified_cuis.append( element )

		if( not_found_flag is True ):
			number_of_skipped_lines += 1
			continue

		# Add Actual Found Data To Found Data Length ( Used For Batch_Gen() )
		actual_train_data_length += 1
		
		subject_cui_index = unique_cui_data[ line_elements[0] ]
		predicate_index   = unique_predicate_data[ line_elements[1] ]
		
		# Add Unique Element Indices To Concept/Predicate Input List
		# Fetch CUI Sparse Indices (Multiple Indices Per Sparse Vector Supported / Association Vectors Supported)
		
		PrintLog( "GenerateNetworkMatrices() - Adding Index: " + str( index ) + ", Subject CUI Index: " + str( subject_cui_index ) + ", Value: 1" )
			
		concept_input_indices.append( [ index, subject_cui_index ] )
		concept_input_values.append( 1 )

		PrintLog( "GenerateNetworkMatrices() - Adding Index: " + str( index ) + ", Predicate Index: " + str( predicate_index ) + ", Value: 1" )
		predicate_input_indices.append( [ index, predicate_index ] )
		predicate_input_values.append( 1 )

		number_of_unique_cui_inputs       += 1
		number_of_unique_predicate_inputs += 1


		# Adds All Object CUI Indices To The Output CUI Index Array
		for element in line_elements[2:]:
			PrintLog( "GenerateNetworkMatrices() - Adding Index: " + str( index ) + ", Object CUI Index: " + str( unique_cui_data[element] ) + ", Value: " + str( unique_cui_data[element] ) )
			concept_output_indices.append( [ index, unique_cui_data[element] ] )
			concept_output_values.append( unique_cui_data[element] )

		index += 1


	# Check(s)
	if( len( concept_input_indices ) is 0 ):
		PrintLog( "GenerateNetworkMatrices() - Error: Concept Input Indices List Contains No Data / Specified Subject CUIs In The Input Not File Not Within Unique CUI Data?" )
	if( len( predicate_input_indices ) is 0 ):
		PrintLog( "GenerateNetworkMatrices() - Error: Predicate Input Indices List Contains No Data / Specified Predicates In The Input Not File Not Within Unique CUI Data?" )
		PrintLog( "GenerateNetworkMatrices() -        Note: This May Be Reporting Due To Concept Input Indices List Erroring Out" )
	if( len( concept_output_indices ) is 0 ):
		PrintLog( "GenerateNetworkMatrices() - Error: Concept Output Indices List Contains No Data / Specified Object CUIs In The Input Not File Not Within Unique CUI Data?" )
		PrintLog( "GenerateNetworkMatrices() -        Note: This May Be Reporting Due To Concept Input or Predicate Indices List Erroring Out" )
	if( len( concept_input_indices ) is 0 or len( predicate_input_indices ) is 0 or len( concept_output_indices ) is 0 ): return None, None, None


	# Set Up Sparse Matrices To Include All Specified CUI/Predicate Vectors
	matrix_cui_length       = len( identified_cuis )
	matrix_predicate_length = len( identified_predicates )

	# If Adjust For Unidentified Vectors == False, Then All Sparse Matrices Consist Of All Vectors In Vector Files
	if( adjust_for_unidentified_vectors is 0 ):
		matrix_cui_length       = len( unique_cui_data )
		matrix_predicate_length = len( unique_predicate_data )

	# Transpose The Arrays, Then Convert To Rows/Columns
	PrintLog( "GenerateNetworkMatrices() - Transposing Index Data Arrays Into Row/Column Data" )
	concept_input_row,   concept_input_column   = zip( *concept_input_indices   )
	predicate_input_row, predicate_input_column = zip( *predicate_input_indices )
	concept_output_row,  concept_output_column  = zip( *concept_output_indices  )

	# Convert Row/Column Data Into Sparse Matrices
	PrintLog( "GenerateNetworkMatrices() - Converting Index Data Into Matrices" )
	concept_input_matrix   = sparse.csr_matrix( ( concept_input_values,               ( concept_input_row,   concept_input_column ) ),   shape = ( number_of_unique_cui_inputs,       matrix_cui_length ) )
	predicate_input_matrix = sparse.csr_matrix( ( predicate_input_values,             ( predicate_input_row, predicate_input_column ) ), shape = ( number_of_unique_predicate_inputs, matrix_predicate_length ) )
	concept_output_matrix  = sparse.csr_matrix( ( [1]*len( concept_output_indices ),  ( concept_output_row,  concept_output_column ) ),  shape = ( number_of_unique_cui_inputs,       matrix_cui_length ) )

	if( print_input_matrices is 1 ):
		PrintLog( "Compressed Sparse Matrix - Subject CUIs" )
		PrintLog( concept_input_matrix )
		PrintLog( "Original Dense Formatted Sparse Matrix" )
		PrintLog( concept_input_matrix.todense() )
		PrintLog( "Compressed Sparse Matrix - Predicates" )
		PrintLog( predicate_input_matrix )
		PrintLog( "Original Dense Formatted Sparse Matrix" )
		PrintLog( predicate_input_matrix.todense() )
		PrintLog( "Compressed Sparse Matrix - Object CUIs" )
		PrintLog( concept_output_matrix )
		PrintLog( "Original Dense Formatted Sparse Matrix" )
		PrintLog( concept_output_matrix.todense() )

	PrintLog( "GenerateNetworkMatrices() - Complete" )


	#returns the CUI input matrix, Predicate input matrix, and CUI output matrix
	return concept_input_matrix, predicate_input_matrix, concept_output_matrix





#evaluates the data on the loaded model
def Evaluate(concept_input, predicate_input, concept_output, metricSet):
	global eval_output_file

	 # Check(s)
	if( concept_input is None ):
		PrintLog( "Evaluate() - Error: Concept Input Contains No Data" )
	if( predicate_input is None ):
		PrintLog( "Evaluate() - Error: Predicate Input Contains No Data" )
	if( concept_output is None ):
		PrintLog( "Evaluate() - Error: Concept Output Contains No Data" )
	if( concept_input is None or predicate_input is None or concept_output is None ):
		return None

	#load the model weights and architechture
	model = LoadModel()

	#recreate the model with the parameters set by the configuration
	sgd = optimizers.SGD( lr = trainNN.learning_rate, momentum = trainNN.momentum )
	model.compile(loss = BCE, optimizer = sgd, metrics = metricSet)
	eval_metrics = model.evaluate_generator(generator = Batch_Gen([concept_input, predicate_input], concept_output, batch_size = batch_size ), steps = steps, verbose=1)
	
	if(len(eval_metrics) < 1):
		PrintLog("Evaluate() - Error: Evaluation data unsuccessfully generated (no evaluation returned)")
		exit()

	#export the results from the evaluation to a file
	f = open(eval_output_file, "w+")
	f.write("EVALUATION METRICS:\n")
	for mv in range(len(metricSet)):

		m = metricSet[mv]
		
		#make a cleaner format for the output
		metricType = ""
		if(m == Precision):
			metricType = "Precision"
		elif(m == Recall):
			metricType = "Recall"
		elif(m == Matthews_Correlation):
			metricType = "Matthews_Correlation"
		elif(m == 'accuracy'):
			metricType = "Accuracy"
	

		#write to the output
		f.write(metricType + ": " + str(eval_metrics[mv]) + "\n")



############################################################################################
#                                                                                          #
#    Main                                                                                  #
#                                                                                          #
############################################################################################

#runs the evaluation code in the order necessary
# 1. read the config file (TrainNN.py and EvaluateNN.py version)
# 2. Generate the Network Matrices for the evaluation data set in the format of the original training data
# 3. Evaluate the given set based on accuracy, precision, recall, and MCC
def main():
	config_file = sys.argv[1]

	# Check(s)
	if( len( sys.argv ) < 2 ):
		PrintLog( "evaluateNN.py Main() - Error: No Configuration File Argument Specified" )
		exit()

	PrintLog(("evaluateNN.py Main() - Reading in Parameters from config file: %s\n", config_file))

	print("BATCH 1: " + str(trainNN.batch_size))
	#result = ReadConfigFile( config_file )
	result2 = ReadConfigFile_EVAL( config_file )
	
	#result2 = ReadConfigFile_EVAL( config_file )

	PrintLog("evaluateNN.py Main() - Generating Network Matrices from evaluation file\n" )

	cui_input, predicate_input, cui_output = GenerateNetworkMatrices()

	PrintLog("evaluateNN.py Main() - Evaluating data\n")

	if( cui_input != None and predicate_input != None and cui_output != None ):
		Evaluate( cui_input, predicate_input, cui_output, ['accuracy', Precision, Recall, Matthews_Correlation])
	else:
		PrintLog("evaluateNN.py Main() - Error: CUI input, Predicate Input, and/or CUI output matrices were not created successfully")
		exit()

	# Garbage Collection / Free Unused Memory
	CleanUp()

	CloseDebugFileHandle()

	PrintLog( "~Fin Eval" )



# Main Function Call To Run TrainNN
if __name__ == "__main__":
    main()



