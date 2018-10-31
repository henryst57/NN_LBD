##################################


# NEURAL NETWORK EVALUATOR
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
eval_file                   = ""
pred_key_file               = ""
cui_key_file                = ""




#   Reads The Specified Configuration File Parameters Into Memory
#   and Sets The Appropriate Variable Data
def ReadConfigFile_EVAL( config_file_path ):
	global model_file
	global model_weights_file
	global eval_file
	global pred_key_file
	global cui_key_file

	# Check(s)
	if CheckIfFileExists( config_file_path ) == False:
		print( "ReadConfigFile_EVAL() - Error: Specified File \"" + str( config_file_path ) + "\" Does Not Exist" )
		return -1

	# Assuming The File Exists, Check If The File Has Data
	if os.stat( config_file_path ).st_size == 0:
		print( "ReadConfigFile_EVAL() - Error: Specified File \"" + str( config_file_path ) + "\" Is Empty File" )
		return -1

	# Open The File And Read Data, Line-By-Line
	try:
		f = open( config_file_path, "r" )
	except FileNotFoundError:
		print( "ReadConfigFile_EVAL() - Error: Specified File \"" + str( config_file_path ) + "\" Does Not Exist" )
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
		
	f.close()

	#grab the key files as well if not specified
	if(train_file):
		if(cui_key_file == ""):
			cui_key_file = (train_file + ".cui_key")
		if(pred_key_file == ""):
			pred_key_file = (train_file + ".predicate_key")
	elif(not train_file and ((cui_key_file == "") or (pred_key_file == "")) ):
		PrintLog( "ReadConfigFile_EVAL() - Error: \"train_file\" Variable Not Set!" )
		exit()


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
	
	PrintLog( "=========================================================" )
	PrintLog( "-                                                       -" )
	PrintLog( "=========================================================\n" )

	PrintLog( "ReadConfigFile() - Complete" )

	return 0





###########################       EVALUATION NN FUNCTIONS        #######################

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

	#
	model.load_weights(model_weights_file)
	print("Loaded the model from %s" % model_file)

	model.summary()

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

	eval_data = [ line.strip() for line in eval_data ]  # Removes Trailing Space Characters From CUI Data Strings
	eval_data.sort()

	eval_len = len( eval_data )

	return eval_data, eval_len


#loads the unique keys from the 
def LoadUniqKeys():
	# original output:
	#   index id
	#   1 C00023820     (cui)
	#   1 AFFECTS       (pred)
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
	global identified_cuis
	global unidentified_cuis
	global actual_train_data_length
	global cui_dense_input_mode

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
	print( "GenerateNetworkMatrices() - Parsing CUI Data / Generating Network Input-Output Data Arrays" )
	
	index = 0
	number_of_skipped_lines = 0

	for i in range( cui_occurence_data_length ):
		not_found_flag = False
		line = cui_occurence_data[i]
		line_elements = re.split( r"\s+", line )

		#print("LEN:" +str(len(line_elements)))

		# Check(s)
		# If Subject CUI, Predicate and Object CUIs Are Not Found Within The Specified Unique CUI and Predicate Lists, Report Error and Skip The Line
		if( line_elements[0] not in unique_cui_data ):
			print( "GenerateNetworkMatrices() - Error: Subject CUI \"" + str( line_elements[0] ) + "\" Is Not In Unique CUI Data List / Skipping Line " + str( i ) )
			not_found_flag = True
			if( line_elements[0] not in unidentified_cuis ): unidentified_cuis.append( line_elements[0] )
		else:
			if( line_elements[0] not in identified_cuis ):   identified_cuis.append( line_elements[0] )
		if( line_elements[1] not in unique_predicate_data ):
			print( "GenerateNetworkMatrices() - Error: Predicate \"" + str( line_elements[1] ) + "\" Is Not In Unique Predicate Data List / Skipping Line " + str( i ) )
			not_found_flag = True
			if( line_elements[1] not in unidentified_predicates ): unidentified_predicates.append( line_elements[1] )
		else:
			if( line_elements[1] not in identified_predicates ): identified_predicates.append( line_elements[1] )
		for element in line_elements[2:]:
			if( element not in unique_cui_data ):
				print( "GenerateNetworkMatrices() - Error: Object CUI \"" + str( element ) + "\" Is Not In Unique CUI Data List / Skipping Line " + str( i ) )
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
		
		print( "GenerateNetworkMatrices() - Adding Index: " + str( index ) + ", Subject CUI Index: " + str( subject_cui_index ) + ", Value: 1" )
			
		concept_input_indices.append( [ index, subject_cui_index ] )
		concept_input_values.append( 1 )

		print( "GenerateNetworkMatrices() - Adding Index: " + str( index ) + ", Predicate Index: " + str( predicate_index ) + ", Value: 1" )
		predicate_input_indices.append( [ index, predicate_index ] )
		predicate_input_values.append( 1 )

		number_of_unique_cui_inputs       += 1
		number_of_unique_predicate_inputs += 1


		# Adds All Object CUI Indices To The Output CUI Index Array
		for element in line_elements[2:]:
			print( "GenerateNetworkMatrices() - Adding Index: " + str( index ) + ", Object CUI Index: " + str( unique_cui_data[element] ) + ", Value: " + str( unique_cui_data[element] ) )
			concept_output_indices.append( [ index, unique_cui_data[element] ] )
			concept_output_values.append( unique_cui_data[element] )

		index += 1


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


	# Set Up Sparse Matrices To Include All Specified CUI/Predicate Vectors
	matrix_cui_length       = len( identified_cuis )
	matrix_predicate_length = len( identified_predicates )

	# If Adjust For Unidentified Vectors == False, Then All Sparse Matrices Consist Of All Vectors In Vector Files
	if( adjust_for_unidentified_vectors is 0 ):
		matrix_cui_length       = len( unique_cui_data )
		matrix_predicate_length = len( unique_predicate_data )

	# Transpose The Arrays, Then Convert To Rows/Columns
	print( "GenerateNetworkMatrices() - Transposing Index Data Arrays Into Row/Column Data" )
	concept_input_row,   concept_input_column   = zip( *concept_input_indices   )
	predicate_input_row, predicate_input_column = zip( *predicate_input_indices )
	concept_output_row,  concept_output_column  = zip( *concept_output_indices  )

	# Convert Row/Column Data Into Sparse Matrices
	print( "GenerateNetworkMatrices() - Converting Index Data Into Matrices" )
	concept_input_matrix   = sparse.csr_matrix( ( concept_input_values,               ( concept_input_row,   concept_input_column ) ),   shape = ( number_of_unique_cui_inputs,       matrix_cui_length ) )
	predicate_input_matrix = sparse.csr_matrix( ( predicate_input_values,             ( predicate_input_row, predicate_input_column ) ), shape = ( number_of_unique_predicate_inputs, matrix_predicate_length ) )
	concept_output_matrix  = sparse.csr_matrix( ( [1]*len( concept_output_indices ),  ( concept_output_row,  concept_output_column ) ),  shape = ( number_of_unique_cui_inputs,       matrix_cui_length ) )

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

	exit()

	#get the model together
	sgd = optimizers.SGD( lr = learning_rate, momentum = momentum )
	model.compile(loss = BCE, optimizer = sgd, metrics = metricSet)
	eval_metrics = model.evaluate_generator(generator = Batch_Gen([concept_input, predicate_input], concept_output, batch_size = batch_size ), steps = steps, verbose=1)



############################################################################################
#                                                                                          #
#    Main                                                                                  #
#                                                                                          #
############################################################################################




def main():

	config_file = sys.argv[1]

	# Check(s)
	if( len( sys.argv ) < 2 ):
		print( "evaluateNN.py Main() - Error: No Configuration File Argument Specified" )
		exit()

	

	result = ReadConfigFile( config_file )
	result = ReadConfigFile_EVAL( config_file )


	print("Startin' Evaluatin'....\n")

	LoadModel()

	exit()

	#cui_input, predicate_input, cui_output = GenerateNetworkMatrices()

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
	else:
		print("Aw shit")


	# Garbage Collection / Free Unused Memory
	CleanUp()

	CloseDebugFileHandle()

	print( "~Fin Eval" )



main()


