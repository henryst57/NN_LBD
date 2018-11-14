##################################


# NEURAL NETWORK CLASSIFIER
# uses the model from trainNN.py to 
# to predict classes on a dataset

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

#reference Clint's code
sys.path.append('../trainNN')
#from trainNN import BCE, CCE, Matthews_Correlation, Precision, Recall, Batch_Gen
from trainNN import *



########################
#                      #
#   GLOBAL VARIABLES   #
#                      #
########################

#model                       = None
model_file                  = ""      #this may be set by the config later 
model_weights_file          = ""
pred_key_file               = ""
cui_key_file                = ""
prediction_out_file         = ""
model                       = None

cui_occurence_data          = {}
cui_occurence_data_length   = 0
unique_cui_data             = {}
unique_predicate_data       = {} 




#   Reads The Specified Configuration File Parameters Into Memory
#   and Sets The Appropriate Variable Data
def ReadConfigFile_CLASSIFY( config_file_path ):
	global model_file
	global model_weights_file
	global pred_key_file
	global cui_key_file
	global prediction_out_file

	# Check(s)
	if CheckIfFileExists( config_file_path ) == False:
		print( "ReadConfigFile_CLASSIFY() - Error: Specified File \"" + str( config_file_path ) + "\" Does Not Exist" )
		return -1

	# Assuming The File Exists, Check If The File Has Data
	if os.stat( config_file_path ).st_size == 0:
		print( "ReadConfigFile_CLASSIFY() - Error: Specified File \"" + str( config_file_path ) + "\" Is Empty File" )
		return -1

	# Open The File And Read Data, Line-By-Line
	try:
		f = open( config_file_path, "r" )
	except FileNotFoundError:
		print( "ReadConfigFile_CLASSIFY() - Error: Specified File \"" + str( config_file_path ) + "\" Does Not Exist" )
		return -1

	f1 = f.readlines()

	# Set Network Parameters Using Config File Data
	for line in f1:
		line = re.sub( r'<|>|\n', '', line )
		data = line.split( ":" )

		if data[0] == "ModelFile"           : model_file            = str( data[1] )
		if data[0] == "ModelWeights"        : model_weights_file    = str( data[1] )
		if data[0] == "CUIKeyFile"          : cui_key_file          = str( data[1] )
		if data[0] == "PredKeyFile"         : pred_key_file         = str( data[1] )
		if data[0] == "PredOut"             : prediction_out_file         = str( data[1] )
		
	f.close()

	#grab the key files as well if not specified
	if(train_file):
		if(cui_key_file == ""):
			cui_key_file = (train_file + ".cui_key")
		if(pred_key_file == ""):
			pred_key_file = (train_file + ".predicate_key")
	elif(not train_file and ((cui_key_file == "") or (pred_key_file == "")) ):
		PrintLog( "ReadConfigFile_CLASSIFY() - Error: \"train_file\" Variable Not Set!" )
		exit()


	OpenDebugFileHandle()

	# Print Settings To Console
	PrintLog( "=========================================================" )
	PrintLog( "-   Configuration File Settings - CLASSIFY.py           -" )
	PrintLog( "=========================================================" )

	PrintLog( "    Model File             : " + str( model_file ) )
	PrintLog( "    Model Weights File     : " + str( model_weights_file ) )
	PrintLog( "    CUI Key File           : " + str( cui_key_file ) )
	PrintLog( "    Predicate Key File     : " + str( pred_key_file ) )
	PrintLog( "    Prediction Output File : " + str( prediction_out_file ) )
	
	PrintLog( "=========================================================" )
	PrintLog( "-                                                       -" )
	PrintLog( "=========================================================\n" )

	PrintLog( "ReadConfigFile() - Complete" )

	return 0





###########################       EVALUATION NN FUNCTIONS        #######################

#loads the model
def LoadModel():
	global model
	customs = {
		'BCE': BCE,
		'CCE': CCE,
		'Precision': Precision,
		'Recall' : Recall,
		'Matthews_Correlation' : Matthews_Correlation
		};
	model = load_model(model_file, 
		custom_objects=customs)

	#
	model.load_weights(model_weights_file)
	print("Loaded the model from %s" % model_file)

	model.summary()


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
			for line in cui_in_file:
				(index, cuiVal) = line.split()
				unique_cui_data[str(cuiVal)] = int(index)
	except FileNotFoundError:
		print ( "LoadUniqKeys() - Error: Unable To Open File \"" + str( cui_key_file )+ "\"" )
		return -1
	else:
		cui_in_file.close()

	#read in pred key file
	try:
		with open( pred_key_file, "r" ) as pred_in_file:
			for line in pred_in_file:
				(index, predVal) = line.split()
				unique_predicate_data[str(predVal)] = int(index)
	except FileNotFoundError:
		print ( "LoadUniqKeys() - Error: Unable To Open File \"" + str( pred_key_file )+ "\"" )
		return -1
	else:
		pred_in_file.close()





#   Parses Through CUI Data And Generates Sparse Matrices For
#   Concept Input, Predicate Input and Concept Output Data
def GenerateInputMatrices(cuiInput, predicateInput):
	global steps
	global number_of_cuis
	global number_of_predicates
	global identified_cuis
	global unidentified_cuis
	global actual_train_data_length
	global cui_dense_input_mode
	global unique_cui_data
	global unique_predicate_data

	print_input_matrices = 0


	LoadUniqKeys()
	number_of_cuis = len(unique_cui_data)
	number_of_predicates = len(unique_predicate_data)
	
	print("Number of CUIS = " + str(number_of_cuis))
	print("Number of Predicates = " + str(number_of_predicates))

	print( "GenerateInputMatrices() - Generating Input Matrices" )

	number_of_unique_cui_inputs       = 1
	number_of_unique_predicate_inputs = 1
	concept_input_indices   = []
	concept_input_values   = []
	predicate_input_indices = []
	predicate_input_values = []
	concept_output_indices  = []
	concept_output_values   = []
	output_cui_count        = 0

	# Parses each line of raw data input and adds them to arrays for matrix generation
	print( "GenerateInputMatrices() - Parsing CUI Data / Generating Network Input-Output Data Arrays" )
	

	not_found_flag = False
	line_elements = [cuiInput, predicateInput]

		#print("LEN:" +str(len(line_elements)))

		# Check(s)
		# If Subject CUI, Predicate and Object CUIs Are Not Found Within The Specified Unique CUI and Predicate Lists, Report Error and Skip The Line
	if( line_elements[0] not in unique_cui_data ):
		print( "GenerateInputMatrices() - Error: Subject CUI \"" + str( line_elements[0] ) + "\" Is Not In Unique CUI Data List / Skipping Line ")
		not_found_flag = True
		if( line_elements[0] not in unidentified_cuis ): unidentified_cuis.append( line_elements[0] )
	else:
		if( line_elements[0] not in identified_cuis ):   identified_cuis.append( line_elements[0] )
	if( line_elements[1] not in unique_predicate_data ):
		print( "GenerateInputMatrices() - Error: Predicate \"" + str( line_elements[1] ) + "\" Is Not In Unique Predicate Data List / Skipping Line ")
		not_found_flag = True
		if( line_elements[1] not in unidentified_predicates ): unidentified_predicates.append( line_elements[1] )
	else:
		if( line_elements[1] not in identified_predicates ): identified_predicates.append( line_elements[1] )
		
		#cui and predicate both not found in the list
	if( not_found_flag is True ):
		return None, None;

		
	subject_cui_index = unique_cui_data[ line_elements[0] ]
	predicate_index   = unique_predicate_data[ line_elements[1] ]
		
	# Add Unique Element Indices To Concept/Predicate Input List
	# Fetch CUI Sparse Indices (Multiple Indices Per Sparse Vector Supported / Association Vectors Supported)
		
			
	concept_input_indices.append( [ 0, subject_cui_index ] )
	concept_input_values.append( 1 )

	predicate_input_indices.append( [ 0, predicate_index ] )
	predicate_input_values.append( 1 )

	# Set Up Sparse Matrices To Include All Specified CUI/Predicate Vectors
	matrix_cui_length       = len( identified_cuis )
	matrix_predicate_length = len( identified_predicates )

	# If Adjust For Unidentified Vectors == False, Then All Sparse Matrices Consist Of All Vectors In Vector Files
	if( adjust_for_unidentified_vectors is 0 ):
		matrix_cui_length       = len( unique_cui_data )
		matrix_predicate_length = len( unique_predicate_data )

	# Transpose The Arrays, Then Convert To Rows/Columns
	print( "GenerateInputMatrices() - Transposing Index Data Arrays Into Row/Column Data" )
	concept_input_row,   concept_input_column   = zip( *concept_input_indices   )
	predicate_input_row, predicate_input_column = zip( *predicate_input_indices )

	# Convert Row/Column Data Into Sparse Matrices
	print( "GenerateInputMatrices() - Converting Index Data Into Matrices" )
	concept_input_matrix   = sparse.csr_matrix( ( concept_input_values,               ( concept_input_row,   concept_input_column ) ),   shape = ( number_of_unique_cui_inputs,       matrix_cui_length ) )
	predicate_input_matrix = sparse.csr_matrix( ( predicate_input_values,             ( predicate_input_row, predicate_input_column ) ), shape = ( number_of_unique_predicate_inputs, matrix_predicate_length ) )

	if( print_input_matrices is 1 ):
		print( "Compressed Sparse Matrix - Subject CUIs" )
		print( concept_input_matrix )
		print( "Original Dense Formatted Sparse Matrix" )
		print( concept_input_matrix.todense() )
		print( "Compressed Sparse Matrix - Predicates" )
		print( predicate_input_matrix )
		print( "Original Dense Formatted Sparse Matrix" )
		print( predicate_input_matrix.todense() )

	print( "GenerateInputMatrices() - Complete" )

	return concept_input_matrix, predicate_input_matrix



#evaluates the data on the loaded model
def Classify(cui, pred, concept_input, predicate_input):
	global eval_output_file
	global model
	global unique_cui_data
	global prediction_out_file

	f = open(prediction_out_file, "a+")

	 # Check(s)
	if( concept_input is None ):
		print( "Evaluate() - Error: Concept Input Contains No Data" )
	if( predicate_input is None ):
		print( "Evaluate() - Error: Predicate Input Contains No Data" )
	if( concept_input is None or predicate_input is None ):
		return None

	predictions = model.predict(x={"CUI_OneHot_Input": concept_input, "Predicate_OneHot_Input":predicate_input}, verbose=1)
	f.write(cui + " " + pred + "\n")

	cuiPredictSet = {}
	thesePredictions = predictions[0]
	for key in unique_cui_data:
		cuiPredictSet[key] = thesePredictions[unique_cui_data[key]]


	#for i in range(len(thesePredictions)):
		#print(thesePredictions[i])
		#cuiPredictSet[unique_cui_data[i]] = thesePredictions[i]

	
	#rank them
	ranked = sorted(cuiPredictSet, key=cuiPredictSet.get, reverse=True)
	for i in ranked:

		f.write("\t%s = [%s]\n" % (i, cuiPredictSet[i]))


############################################################################################
#                                                                                          #
#    Main                                                                                  #
#                                                                                          #
############################################################################################




def main():

	config_file = sys.argv[1]

	# Check(s)
	if( len( sys.argv ) < 2 ):
		print( "classifyNN.py Main() - Error: No Configuration File Argument Specified" )
		exit()

	

	result = ReadConfigFile( config_file )
	result = ReadConfigFile_CLASSIFY( config_file )

	print("Startin' Classifyin'....\n")

	LoadModel()
	#get the model together
	sgd = optimizers.SGD( lr = learning_rate, momentum = momentum )
	model.compile(loss = BCE, optimizer = sgd, metrics = ['accuracy', Precision, Recall, Matthews_Correlation])

	cuiTest = input("Enter CUI: ")
	predTest = input("Enter Predicate: ")
	while(cuiTest and predTest):
		print("Converting to matrix format")

		cuiInputMat, predInputMat = GenerateInputMatrices(cuiTest, predTest)
		
		print(cuiInputMat)
		print("\n\n")

		Classify(cuiTest, predTest, cuiInputMat, predInputMat)

		cuiTest = input("Enter CUI: ")
		predTest = input("Enter Predicate: ")


	# Garbage Collection / Free Unused Memory
	CleanUp()

	CloseDebugFileHandle()

	print( "~Fin Classify" )



main()


