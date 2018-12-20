#!/usr/bin/python
############################################################################################
#                                                                                          #
#    Neural Network - Literature Based Discovery                                           #
#    -------------------------------------------                                           #
#                                                                                          #
#    Date:    10/02/2018                                                                   #
#    Revised: 12/19/2018                                                                   #
#                                                                                          #
#    Generates A Neural Network Using A Configuration File.                                #
#      - Supports Dense and Sparse Input Vectors In All Combinations Of CUI and            #
#        Predicate Input                                                                   #
#      - Outputs The Trained Network Model, Model Weights, Architecture and Visual         #
#        Model Depiction                                                                   #
#                                                                                          #
#    How To Run:                                                                           #
#        Adjust Parameters In Configuration File (Text File), Then Run via:                #
#                    "python trainNN.py config.cfg"                                        #
#                                                                                          #
#    See Readme File For Adjustable Parameters                                             #
#                                                                                          #
#    Authors:                                                                              #
#    --------                                                                              #
#    Megan Charity - charityml@vcu.edu                                                     #
#    Sam Henry     - henryst@vcu.edu                                                       #
#    Clint Cuffy   - cuffyca@vcu.edu                                                       #
#    VCU NLP Lab                                                                           #
#                                                                                          #
############################################################################################

import os, errno
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import keras.backend as K
from tensorflow.python import keras
from keras.callbacks import CSVLogger
from keras.models import Model, model_from_json, load_model
from keras import optimizers
from keras.utils.vis_utils import plot_model
from keras.layers import Dense, Activation, Input, concatenate, Dropout, Embedding, Flatten
from keras.preprocessing.sequence import pad_sequences              # <- Remove Me: Not Used 
import h5py
import sys
import re
import gc
import time
from fractions import gcd


############################################################################################
#                                                                                          #
#    Global Variables / Neural Network Parameters (Default Values)                         #
#                                                                                          #
############################################################################################
version                         = 0.36
number_of_predicates            = 0
number_of_cuis                  = 0
layer_1_size                    = 200
layer_2_size                    = 400
learning_rate                   = 0.1
number_of_epochs                = 5
steps                           = 1
batch_size                      = 10
momentum                        = 0.9
dropout_amt                     = 0.25
negative_sample_rate            = 5
print_key_files                 = 0
train_file                      = ""
concept_output_file             = ""
test_input_cui                  = ""
test_input_predicate            = ""
test_output_cui                 = ""
concept_vector_file             = ""
predicate_vector_file           = ""
predicate_list_file             = ""
training_stats_file             = ""
testing_stats_file              = ""
output_file_name                = ""
evaluation_file                 = ""
print_network_inputs            = 0
print_matrix_generation_stats   = 0
adjust_for_unidentified_vectors = 0
weight_dump_interval            = 0
process_eval_metrics_per_epoch  = 0
trainable_dense_embeddings      = False
shuffle_input_data              = False
cui_dense_input_mode            = False
predicate_dense_input_mode      = False
train_file_data_length          = 0
actual_train_data_length        = 0
cui_vector_length               = 0
predicate_vector_length         = 0
curr_training_data_index        = 0

# CUI/Predicate Data
training_data                   = []
unique_cui_data                 = {}
unique_predicate_data           = {}
cui_embedding_matrix            = []
predicate_embedding_matrix      = []

# Evaluation File Data
evaluation_data                 = []
eval_cui_input_matrix           = []
eval_predicate_input_matrix     = []
eval_cui_output_matrix          = []

# Stats Variables
identified_cuis                 = []             # CUIs Found In Unique CUI List During Matrix Generation
identified_predicates           = []             # Predicates Found In Unique CUI List During Matrix Generation
unidentified_cuis               = []             # CUIs Not Found In Unique CUI List During Matrix Generation
unidentified_predicates         = []             # Predicates Not Found In Unique CUI List During Matrix Generation

# Debug Log Variables
debug_log                       = 0
write_log                       = 0
debug_file_name                 = "nnlbd_log.txt"
debug_log_file                  = None

############################################################################################
#                                                                                          #
#    Sub-Routines                                                                          #
#                                                                                          #
############################################################################################

#   Print Statements To Console, Debug Log File Or Both
def PrintLog( print_str, force_print = None ):
    if( debug_log is 1 or force_print is 1 ): print( str( print_str ) )
    if( write_log is 1 and debug_log_file != None ):
        debug_log_file.write( str( print_str ) + "\n" )

#   Checks If The Specified File Exists and It Is A File (Not A Directory)
def CheckIfFileExists( file_path ):
    if os.path.exists( file_path ) and os.path.isfile( file_path ):
        return True
    return False

#   Open Write Log File Handle
def OpenDebugFileHandle():
    global debug_log_file
    if( write_log is 1 ): debug_log_file = open( debug_file_name, "w" )
    if( write_log is 1 and debug_log_file is not None ): print( "Debug Log File Enabled: Printing To \"" + str( debug_file_name ) + "\"" )

#   Close Debug Log File Handle
def CloseDebugFileHandle():
    global debug_log_file
    if( write_log is 1 and debug_log_file is not None ): debug_log_file.close

#   Open The Training Statistics File Handle
def OpenTrainingStatsFileHandle():
    global training_stats_file
    PrintLog( "OpenTrainingStatsFileHandle() - Creating and Opening Training Stats File Handle: \"" + training_stats_file + "\"" )
    training_stats_file = open( training_stats_file, "w" )

#   Write String To Training Statistics File Handle
def WriteStringToTrainingStatsFile( string ):
    if( training_stats_file is not "" ):
        PrintLog( "WriteStringToTrainingStatsFile() - Writing String To File: \"" + string + "\"" )
        training_stats_file.write( str( string ) + "\n" )

#   Close The Training Statistics File Handle
def CloseTrainingStatsFileHandle():
    global training_stats_file
    if( training_stats_file is not "" ):
        PrintLog( "CloseTrainingStatsFileHandle() - Closing Training Stats File Handle" )
        training_stats_file.close()
        training_stats_file = ""

#   Open The Testing Statistics File Handle
def OpenTestingStatsFileHandle():
    global testing_stats_file
    PrintLog( "OpenTestingStatsFileHandle() - Creating and Opening Testing Stats File Handle: \"" + testing_stats_file + "\"" )
    testing_stats_file = open( testing_stats_file, "w" )

#   Write String To Testing Statistics File Handle
def WriteStringToTestingStatsFile( string ):
    if( testing_stats_file is not "" ):
        PrintLog( "WriteStringToTestingStatsFile() - Writing String To File: \"" + string + "\"" )
        testing_stats_file.write( str( string ) + "\n" )

#   Close The Testing Statistics File Handle
def CloseTestingStatsFileHandle():
    global testing_stats_file
    if( testing_stats_file is not "" ):
        PrintLog( "CloseTestingStatsFileHandle() - Closing Testing Stats File Handle" )
        testing_stats_file.close()
        testing_stats_file = ""

#   Checks To See If The Input Is In Dense Or Sparse Format
#       False = Dense Format, True = Sparse Format
def IsDenseVectorFormat( vector ):
    # Split Based On "<>" Characters, Only Present In Sparse Format
    number_of_elements = vector.split( '<>' )
    if ( len( number_of_elements ) == 1 ): return True
    if ( len( number_of_elements ) >  1 ): return False

#   Reads The Specified Configuration File Parameters Into Memory
#   and Sets The Appropriate Variable Data
def ReadConfigFile( config_file_path ):
    global steps
    global momentum
    global debug_log
    global write_log
    global batch_size
    global train_file
    global dropout_amt
    global layer_1_size
    global layer_2_size
    global learning_rate
    global debug_log_file
    global test_input_cui
    global debug_file_name
    global evaluation_file
    global test_output_cui
    global print_key_files
    global number_of_epochs
    global output_file_name
    global shuffle_input_data
    global testing_stats_file
    global concept_output_file
    global concept_vector_file
    global predicate_list_file
    global training_stats_file
    global negative_sample_rate
    global print_network_inputs
    global test_input_predicate
    global weight_dump_interval
    global predicate_vector_file
    global trainable_dense_embeddings
    global print_matrix_generation_stats
    global adjust_for_unidentified_vectors

    # Check(s)
    if CheckIfFileExists( config_file_path ) == False:
        print( "ReadConfigFile() - Error: Specified File \"" + str( config_file_path ) + "\" Does Not Exist" )
        return -1

    # Assuming The File Exists, Check If The File Has Data
    if os.stat( config_file_path ).st_size == 0:
        print( "ReadConfigFile() - Error: Specified File \"" + str( config_file_path ) + "\" Is Empty File" )
        return -1

    # Open The File And Read Data, Line-By-Line
    try:
        f = open( config_file_path, "r" )
    except FileNotFoundError:
        print( "ReadConfigFile() - Error: Specified File \"" + str( config_file_path ) + "\" Does Not Exist" )
        return -1

    f1 = f.readlines()

    # Set Network Parameters Using Config File Data
    for line in f1:
        line = re.sub( r'<|>|\n', '', line.strip() )
        data = line.split( ":" )
        
        if data[0] == "DebugLog"                : debug_log                       = int( data[1] )
        if data[0] == "WriteLog"                : write_log                       = int( data[1] )
        if data[0] == "Momentum"                : momentum                        = float( data[1] )
        if data[0] == "BatchSize"               : batch_size                      = int( data[1] )
        if data[0] == "DropoutAMT"              : dropout_amt                     = float( data[1] )
        if data[0] == "Layer1Size"              : layer_1_size                    = int( data[1] )
        if data[0] == "Layer2Size"              : layer_2_size                    = int( data[1] )
        if data[0] == "LearningRate"            : learning_rate                   = float( data[1] )
        if data[0] == "TestInputCUI"            : test_input_cui                  = str( data[1] )
        if data[0] == "TestInputPredicate"      : test_input_predicate            = str( data[1] )
        if data[0] == "TestOutputCUI"           : test_output_cui                 = str( data[1] )
        if data[0] == "NumberOfSteps"           : steps                           = int( data[1] )
        if data[0] == "NumberOfEpochs"          : number_of_epochs                = int( data[1] )
        if data[0] == "TrainFile"               : train_file                      = str( data[1] )
        if data[0] == "NegativeSampleRate"      : negative_sample_rate            = int( data[1] )
        if data[0] == "PrintKeyFiles"           : print_key_files                 = int( data[1] )
        if data[0] == "ConceptVectorFile"       : concept_vector_file             = str( data[1] )
        if data[0] == "PredicateVectorFile"     : predicate_vector_file           = str( data[1] )
        if data[0] == "PredicateListFile"       : predicate_list_file             = str( data[1] )
        if data[0] == "TrainingStatsFile"       : training_stats_file             = str( data[1] )
        if data[0] == "TestingStatsFile"        : testing_stats_file              = str( data[1] )
        if data[0] == "PrintNetworkInputs"      : print_network_inputs            = int( data[1] )
        if data[0] == "PrintMatrixStats"        : print_matrix_generation_stats   = int( data[1] )
        if data[0] == "AdjustVectors"           : adjust_for_unidentified_vectors = int( data[1] )
        if data[0] == "TrainableDenseWeights"   : trainable_dense_embeddings      = int( data[1] )
        if data[0] == "ShuffleInputData"        : shuffle_input_data              = int( data[1] )
        if data[0] == "OutputFileName"          : output_file_name                = str( data[1] )
        if data[0] == "WeightDumpInterval"      : weight_dump_interval            = int( data[1] )
        if data[0] == "EvaluateFile"            : evaluation_file                 = str( data[1] )

    f.close()
    
    # Set Debug Log File Name To Match Output File Name
    if( write_log != 0 and output_file_name is not "" ):
        debug_file_name = output_file_name + "_" + debug_file_name
        PrintLog( "ReadConfigFile() - Setting Debug Log File Name: \"" + debug_file_name + "\"" )

    OpenDebugFileHandle()

    # Check(s)
    if( train_file is "" ):
        PrintLog( "ReadConfigFile() - Error: \"TrainFile\" Not Specified", 1 )
    if( CheckIfFileExists( train_file ) is False ):
        PrintLog( "ReadConfigFile() - Error: \"" + str( train_file ) + "\" Does Not Exist", 1 )
        exit()
    if( batch_size is 0 ):
        PrintLog( "ReadConfigFile() - Error: Batch_Size Variable Cannot Be <= \"0\" / Exiting Program", 1 )
        exit()
    if( concept_vector_file != "" and concept_vector_file == predicate_vector_file and ( predicate_list_file is "" or predicate_list_file is None ) ):
        PrintLog( "ReadConfigFile() - Error: When \"ConceptVectorFile\" == \"PredicateVectorFile\"", 1 )
        PrintLog( "ReadConfigFile() -        A Valid Predicate List Must Be Specified", 1 )
        exit()
    if( output_file_name is "" ):
        PrintLog( "ReadConfigFile() - Warning: Output File Name Is Empty / Setting To \"trained_nn\"", 1 )
        output_file_name = "trained_nn"
    if( weight_dump_interval >= number_of_epochs ):
        PrintLog( "ReadConfigFile() - Warning: Weight Dump Interval Parameter >= Number Of Epochs / Setting Weight Dump Interval = 0", 1 )
        weight_dump_interval = 0
    if( training_stats_file == "" ):
        PrintLog( "ReadConfigFile() - Warning: Training Stats File Not Specified / Training Metrics Will Computed And Reported On A Batch-Basis", 1 )
    if( testing_stats_file is "" or evaluation_file is "" ):
        PrintLog( "ReadConfigFile() - Warning: Testing Stats File/Evaluation File Is Not Specified / Testing Metrics Will Not Be Reported", 1 )
    if( concept_vector_file != "" and CheckIfFileExists( concept_vector_file ) == False ):
        PrintLog( "ReadConfigFile() - Error: Concept Vector File - \"" + str( concept_vector_file ) + "\" Does Not Exist", 1 )
    if( predicate_vector_file != "" and CheckIfFileExists( predicate_vector_file ) == False ):
        PrintLog( "ReadConfigFile() - Error: Predicate Vector File - \"" + str( predicate_vector_file ) + "\" Does Not Exist", 1 )
    if( ( concept_vector_file != "" and CheckIfFileExists( concept_vector_file ) == False )
        or ( predicate_vector_file != "" and CheckIfFileExists( predicate_vector_file ) == False ) ):
        exit()

    if( trainable_dense_embeddings is 1 ): trainable_dense_embeddings = True
    else:                                  trainable_dense_embeddings = False
    
    if( shuffle_input_data is 1 ): shuffle_input_data = True
    else:                          shuffle_input_data = False

    PrintLog( "=========================================================" )
    PrintLog( "~      Neural Network - Literature Based Discovery      ~" )
    PrintLog( "~         Version " + str( version ) + " (Based on CaNaDA v0.8)           ~" )
    PrintLog( "=========================================================\n" )

    PrintLog( "  Built on Tensorflow Version: 1.8.0" )
    PrintLog( "  Built on Keras Version: 2.1.5" )
    PrintLog( "  Installed TensorFlow Version: " + str( tf.__version__ ) )
    PrintLog( "  Installed Keras Version: "      + str( keras.__version__ )  + "\n" )

    # Print Settings To Console
    PrintLog( "=========================================================" )
    PrintLog( "-   Configuration File Settings                         -" )
    PrintLog( "=========================================================" )

    PrintLog( "    Train File                 : " + str( train_file ) )
    PrintLog( "    Concept Vector File        : " + str( concept_vector_file ) )
    PrintLog( "    Predicate Vector File      : " + str( predicate_vector_file ) )
    PrintLog( "    Predicate List File        : " + str( predicate_list_file ) )
    PrintLog( "    Training Stats File        : " + str( training_stats_file ) )
    PrintLog( "    Testing Stats File         : " + str( testing_stats_file ) )
    PrintLog( "    Evaluation File            : " + str( evaluation_file ) )
    PrintLog( "    Output File Name           : " + str( output_file_name ) )
    PrintLog( "    Batch Size                 : " + str( batch_size ) )
    PrintLog( "    Learning Rate              : " + str( learning_rate ) )
    PrintLog( "    Number Of Epochs           : " + str( number_of_epochs ) )
    PrintLog( "    Number Of Steps            : " + str( steps ) )
    PrintLog( "    Momentum                   : " + str( momentum ) )
    PrintLog( "    Dropout AMT                : " + str( dropout_amt ) )
    PrintLog( "    Layer 1 Size               : " + str( layer_1_size ) )
    PrintLog( "    Layer 2 Size               : " + str( layer_2_size ) )
    PrintLog( "    Negative Sample Rate       : " + str( negative_sample_rate ) )
    PrintLog( "    Print Key Files            : " + str( print_key_files ) )
    PrintLog( "    Print Network Inputs       : " + str( print_network_inputs ) )
    PrintLog( "    Print Matrix Stats         : " + str( print_matrix_generation_stats ) )
    PrintLog( "    Test Input CUI             : " + str( test_input_cui ) )
    PrintLog( "    Test Input Predicate       : " + str( test_input_predicate ) )
    PrintLog( "    Test Output CUI            : " + str( test_output_cui ) )
    PrintLog( "    Adjust Vectors             : " + str( adjust_for_unidentified_vectors ) )
    PrintLog( "    Trainable Dense Weights    : " + str( trainable_dense_embeddings ) )
    PrintLog( "    Shuffle Input Data         : " + str( shuffle_input_data ) )
    PrintLog( "    Weight Dump Interval       : " + str( weight_dump_interval ) )

    PrintLog( "=========================================================" )
    PrintLog( "-                                                       -" )
    PrintLog( "=========================================================\n" )

    PrintLog( "ReadConfigFile() - Complete" )

    return 0

def LoadVectorFile( vector_file_path, is_cui_vectors ):
    global number_of_cuis
    global unique_cui_data
    global cui_vector_length
    global number_of_predicates
    global cui_dense_input_mode
    global cui_embedding_matrix
    global unique_predicate_data
    global predicate_vector_length
    global predicate_dense_input_mode
    global predicate_embedding_matrix

    # Check(s)
    if( vector_file_path is None or vector_file_path is "" ):
        if( is_cui_vectors is True ):  PrintLog( "LoadVectorFile() - Warning: No CUI Vector File Specified", 1 )
        if( is_cui_vectors is False ): PrintLog( "LoadVectorFile() - Warning: No Predicate Vector File Specified", 1 )
        return -1

    if( CheckIfFileExists( vector_file_path ) is False ):
        PrintLog( "LoadVectorFile() - Error: Specified File \"" + str( vector_file_path ) + "\" Does Not Exist", 1 )
        return -1

    PrintLog( "LoadVectorFile() - Loading Vector File: \"" + vector_file_path +   "\"" )
    vector_data = None

    try:
        with open( vector_file_path, "r" ) as in_file:
            vector_data = in_file.readlines()
            vector_data.sort()
    except FileNotFoundError:
        PrintLog( "LoadVectorFile() - Error: Unable To Open File \"" + str( vector_file_path ) + "\"", 1 )
        return -1
    finally:
        in_file.close()

    # Check(s)
    if( vector_data is None ):
        PrintLog( "LoadVectorFile() - Error: Failed To Load Vector Data", 1 )
        return -1
    else:
        PrintLog( "LoadVectorFile() - Loaded " + str( len( vector_data ) ) + " Vector Elements" )

    # Choose The Second Element In The Vector Data And Check Vector Format (CUI)
    if( len( vector_data ) > 1 and is_cui_vectors is True ):
        cui_dense_input_mode       = IsDenseVectorFormat( vector_data[2].strip() )
        if( cui_dense_input_mode == True ):  PrintLog( "LoadVectorFile() - Detected Dense CUI Vector Format" )
        if( cui_dense_input_mode == False ): PrintLog( "LoadVectorFile() - Detected Sparse CUI Vector Format" )

    # Choose The Second Element In The Vector Data And Check Vector Format (Predicate):
    if( len( vector_data ) > 1 and is_cui_vectors is False ):
        predicate_dense_input_mode = IsDenseVectorFormat( vector_data[2].strip() )
        if( predicate_dense_input_mode == True ):  PrintLog( "LoadVectorFile() - Detected Dense Predicate Vector Format" )
        if( predicate_dense_input_mode == False ): PrintLog( "LoadVectorFile() - Detected Sparse Predicate Vector Format" )

    loaded_elements  = 0

    # Read Dense Vector Formatted File
    if( ( is_cui_vectors == True and cui_dense_input_mode == True ) or ( is_cui_vectors == False and predicate_dense_input_mode == True ) ):
        unique_index = 1

        PrintLog( "LoadVectorFile() - Parsing Dense Vector Data" )

        for vector in vector_data:
            vector = vector.strip()
            vector = vector.split()

            # Parse Header Information
            if( len( vector ) == 2 ):
                PrintLog( "LoadVectorFile() - Parsing Dense Vector Header Information" )
                number_of_vectors = int( vector[0] )

                if( is_cui_vectors is True  ):
                    cui_vector_length       = int( vector[1] )
                    PrintLog( "LoadVectorFile() - Number Of Vectors: " + str( number_of_vectors ) + ", CUI Vector Length: " + str( cui_vector_length ) )

                if( is_cui_vectors is False ):
                    predicate_vector_length = int( vector[1] )
                    PrintLog( "LoadVectorFile() - Number Of Vectors: " + str( number_of_vectors ) + ", Predicate Vector Length: " + str( predicate_vector_length ) )
                
                # Append Nothing To First Element / Index "0"
                PrintLog( "LoadVectorFile() - Adding First Embedding Matrix Element / Array Of All Zeros At Index \"0\"" )
                if( is_cui_vectors == True  ):
                    cui_embedding_matrix = np.zeros( ( number_of_vectors + 1, cui_vector_length ) )
                
                if( is_cui_vectors == False ):
                    predicate_embedding_matrix = np.zeros( ( number_of_vectors + 1, predicate_vector_length ) )
                
            # Parse CUI/Predicate Data
            else:
                label = vector[0]
                data  = vector[1:]

                # Add Unique CUI Index And Data
                if( is_cui_vectors is True and label not in unique_cui_data ):
                    PrintLog( "LoadVectorFile() -   Found Unique CUI: \"" + str( label ) + "\", Assigning Index: " + str( unique_index ) )
                    unique_cui_data[ label ] = unique_index

                    PrintLog( "LoadVectorFile() -   Appending CUI: \"" + str( label ) + "\" Vector Data To Embedding Matrix" )
                    cui_embedding_matrix[unique_index] = np.asarray( data, dtype='float32' )
                    loaded_elements += 1

                # Add Unique Predicate Index And Data
                if( is_cui_vectors is False and label not in unique_predicate_data ):
                    PrintLog( "LoadVectorFile() -   Found Unique Predicate: \"" + str( label ) + "\", Assigning Index: " + str( unique_index ) )
                    unique_predicate_data[ label ] = unique_index

                    PrintLog( "LoadVectorFile() -   Appending Predicate: \"" + str( label ) + "\" Vector Data To Embedding Matrix" )
                    predicate_embedding_matrix[unique_index] = np.asarray( data, dtype='float32' )
                    loaded_elements += 1

                unique_index += 1

        PrintLog( "LoadVectorFile() - Assigning Number Of CUIs/Number Of Predicates Values" )
        if( is_cui_vectors is True  ): number_of_cuis       = len( unique_cui_data )
        if( is_cui_vectors is False ): number_of_predicates = len( unique_predicate_data )

    # Read Sparse Formatted Vector File
    else:
        unique_index = 1

        PrintLog( "LoadVectorFile() - Parsing Sparse Vector Data" )
        
        # Append Nothing To First Element / Index "0"
        if( is_cui_vectors == True  ): cui_embedding_matrix.append( "" )
        if( is_cui_vectors == False ): predicate_embedding_matrix.append( "" )

        for vector in vector_data:
            vector = vector.strip()
            vector = vector.split( "<>" )
            label  = vector[0]
            data    = " ".join( vector[1:] )
            data    = re.sub( r',', ':', data )

            loaded_elements += len( data.split( " " ) )

            if( is_cui_vectors == True and label not in unique_cui_data ):
                PrintLog( "LoadVectorFile() -   Found Unique CUI: \"" + str( label ) + "\", Assigning Index: " + str( unique_index ) )
                unique_cui_data[ label ] = unique_index

                PrintLog( "LoadVectorFile() -   Appending CUI: \"" + str( label ) + "\" Vector Data To Embedding Matrix" )
                cui_embedding_matrix.append( data )
                unique_index += 1

            if( is_cui_vectors == False and label not in unique_predicate_data ):
                PrintLog( "LoadVectorFile() -   Found Unique Predicate: \"" + str( label ) + "\", Assigning Index: " + str( unique_index ) )
                unique_predicate_data[ label ] = unique_index

                PrintLog( "LoadVectorFile() -   Appending Predicate: \"" + str( label ) + "\" Vector Data To Embedding Matrix" )
                predicate_embedding_matrix.append( data )
                unique_index += 1

        PrintLog( "LoadVectorFile() - Assigning Number Of CUIs/Number Of Predicates/CUI Vector Length/Predicate Vector Length Values" )
        if( is_cui_vectors is True  ):
            number_of_cuis          = len( unique_cui_data )
            cui_vector_length       = len( unique_cui_data )
        if( is_cui_vectors is False ):
            number_of_predicates    = len( unique_predicate_data )
            predicate_vector_length = len( unique_predicate_data )

    PrintLog( "LoadVectorFile() -   CUI Vector Length                  : " + str(      cui_vector_length            ) )
    PrintLog( "LoadVectorFile() -   Predicate Vector Length            : " + str(      predicate_vector_length      ) )
    PrintLog( "LoadVectorFile() -   Number Of CUIs                     : " + str(      number_of_cuis               ) )
    PrintLog( "LoadVectorFile() -   Number Of Predicates               : " + str(      number_of_predicates         ) )
    PrintLog( "LoadVectorFile() -   Number Of CUI Embedding Data       : " + str( len( cui_embedding_matrix       ) ) )
    PrintLog( "LoadVectorFile() -   Number Of Predicate Embedding Data : " + str( len( predicate_embedding_matrix ) ) )

    PrintLog( "LoadVectorFile() - Loaded " + str( loaded_elements ) + " Elements" )
    PrintLog( "LoadVectorFile() - Complete" )
    return 0

def LoadVectorFileUsingPredicateList( vector_file_path, predicate_list_path ):
    global number_of_cuis
    global unique_cui_data
    global cui_vector_length
    global number_of_predicates
    global cui_dense_input_mode
    global cui_embedding_matrix
    global unique_predicate_data
    global predicate_vector_length
    global predicate_dense_input_mode
    global predicate_embedding_matrix

    # Check(s)
    if( vector_file_path is None or vector_file_path is "" ):
        PrintLog( "LoadVectorFileUsingPredicateList() - Error: No Vector File Specified", 1 )
        return -1

    if CheckIfFileExists( vector_file_path ) is False:
        PrintLog( "LoadVectorFileUsingPredicateList() - Error: Specified File \"" + str( vector_file_path ) + "\" Does Not Exist", 1 )
        return -1

    if( predicate_list_path is None or predicate_list_path is "" ):
        PrintLog( "LoadVectorFileUsingPredicateList() - Warning: No Predicate List File Specified", 1 )
        return -1

    if CheckIfFileExists( predicate_list_path ) is False:
        PrintLog( "LoadVectorFileUsingPredicateList() - Error: Specified File \"" + str( predicate_list_path ) + "\" Does Not Exist", 1 )
        return -1

    PrintLog( "LoadVectorFileUsingPredicateList() - Loading Predicate List: \"" + predicate_list_path + "\"" )

    # Load Predicate List
    predicate_list = []

    if( predicate_list_file is not None ):
        PrintLog( "LoadVectorFileUsingPredicateList() - Reading Predicate List" )
        try:
            with open( predicate_list_file, "r" ) as in_file:
                predicate_data = in_file.readlines()
                predicate_data.sort()
                for predicate in predicate_data:
                    predicate_list.append( predicate.strip() )
        except FileNotFoundError:
            PrintLog( "LoadVectorFileUsingPredicateList() - Error: Unable To Open File \"" + str( predicate_list_file ) + "\"", 1 )
            return -1
        finally:
            in_file.close()

    # Check(s)
    if( predicate_list is None ):
        PrintLog( "LoadVectorFileUsingPredicateList() - Error: Failed To Predicate List Data", 1 )
        return -1
    else:
        PrintLog( "LoadVectorFileUsingPredicateList() - Loaded " + str( len( predicate_list ) ) + " Predicate List Elements" )

    # Load The Vector File
    PrintLog( "LoadVectorFileUsingPredicateList() - Loading Vector File: \"" + vector_file_path +   "\"" )
    vector_data      = None

    try:
        with open( vector_file_path, "r" ) as in_file:
            vector_data = in_file.readlines()
    except FileNotFoundError:
        PrintLog( "LoadVectorFileUsingPredicateList() - Error: Unable To Open File \"" + str( vector_file_path ) + "\"", 1 )
        return -1
    finally:
        in_file.close()

    # Check(s)
    if( vector_data is None ):
        PrintLog( "LoadVectorFileUsingPredicateList() - Error: Failed To Load Vector Data", 1 )
        return -1
    else:
        PrintLog( "LoadVectorFileUsingPredicateList() - Loaded " + str( len( vector_data ) ) + " Vector Elements" )

    # Choose The Second Element In The Vector Data And Check Vector Format (CUI/Predicate)
    if( len( vector_data ) > 1 ):
        cui_dense_input_mode       = IsDenseVectorFormat( vector_data[2].strip() )
        predicate_dense_input_mode = cui_dense_input_mode
        if( cui_dense_input_mode == True ):  PrintLog( "LoadVectorFileUsingPredicateList() - Detected Dense CUI/Predicate Vector Format" )
        if( cui_dense_input_mode == False ):
            PrintLog( "LoadVectorFileUsingPredicateList() - Detected Sparse CUI/Predicate Vector Format"               , 1 )
            PrintLog( "LoadVectorFileUsingPredicateList() - Error: Sparse CUI/Predicate Vector Format Not Supported"   , 1 )
            PrintLog( "                                            Please Use Separate CUI/Predicate Vectors In Config", 1 )
            return -1

    loaded_cui_elements       = 1
    loaded_predicate_elements = 1

    PrintLog( "LoadVectorFileUsingPredicateList() - Parsing Vector Data" )

    # Read Dense Vector Formatted File
    if( cui_dense_input_mode == True ):
        PrintLog( "LoadVectorFileUsingPredicateList() - Parsing Dense Vector Data" )
        
        for vector in vector_data:
            vector = vector.strip()
            vector = vector.split()

            # Parse Header Information
            if( len( vector ) == 2 ):
                PrintLog( "LoadVectorFileUsingPredicateList() - Parsing Dense Vector Header Information" )
                number_of_vectors       = int( vector[0] )

                if( cui_dense_input_mode is True ):
                    cui_vector_length       = int( vector[1] )
                    PrintLog( "LoadVectorFileUsingPredicateList() - Number Of Vectors: " + str( number_of_vectors ) + ", CUI Vector Length: " + str( cui_vector_length ) )

                if( predicate_dense_input_mode is True ):
                    predicate_vector_length = int( vector[1] )
                    PrintLog( "LoadVectorFileUsingPredicateList() - Number Of Vectors: " + str( number_of_vectors ) + ", Predicate Vector Length: " + str( predicate_vector_length ) )
                
                # Append Nothing To First Element / Index "0"
                PrintLog( "LoadVectorFileUsingPredicateList() - Adding First Embedding Matrix Element / Array Of All Zeros At Index \"0\"" )
                if( cui_dense_input_mode == True  ):
                    cui_embedding_matrix.append( np.zeros( cui_vector_length ) )
                    predicate_embedding_matrix.append( np.zeros( predicate_vector_length ) )
                
            # Parse CUI/Predicate Data
            else:
                label = vector[0]
                data  = vector[1:]

                # Add Unique CUI Index And Data
                if( label not in predicate_list and label not in unique_cui_data ):
                    PrintLog( "LoadVectorFileUsingPredicateList() -   Found Unique CUI: \"" + str( label ) + "\", Assigning Index: " + str( loaded_cui_elements ) )
                    unique_cui_data[ label ] = loaded_cui_elements

                    PrintLog( "LoadVectorFileUsingPredicateList() -   Appending CUI: \"" + str( label ) + "\" Vector Data To Embedding Matrix" )
                    cui_embedding_matrix.append( np.asarray( data, dtype = 'float32' ) )
                    loaded_cui_elements += 1

                # Add Unique Predicate Index And Data
                if( label in predicate_list and label not in unique_predicate_data ):
                    PrintLog( "LoadVectorFileUsingPredicateList() -   Found Unique Predicate: \"" + str( label ) + "\", Assigning Index: " + str( loaded_predicate_elements ) )
                    unique_predicate_data[ label ] = loaded_predicate_elements

                    PrintLog( "LoadVectorFileUsingPredicateList() -   Appending Predicate: \"" + str( label ) + "\" Vector Data To Embedding Matrix" )
                    predicate_embedding_matrix.append( np.asarray( data, dtype = 'float32' ) )
                    loaded_predicate_elements += 1
            
        cui_embedding_matrix       = np.asarray( cui_embedding_matrix, dtype = 'float32' )
        predicate_embedding_matrix = np.asarray( predicate_embedding_matrix, dtype = 'float32' )

    # Read Sparse Formatted Vector File
    else:
        PrintLog( "LoadVectorFileUsingPredicateList() - Parsing Sparse Vector Data" )
        
        # Append Nothing To First Element / Index "0"
        cui_embedding_matrix.append( "" )
        predicate_embedding_matrix.append( "" )

        unique_cui_index       = 1
        unique_predicate_index = 1

        for vector in vector_data:
            vector = vector.strip()
            vector = vector.split( "<>" )
            label  = vector[0]
            data    = " ".join( vector[1:] )
            data    = re.sub( r',', ':', data )

            if( label not in predicate_list and label not in unique_cui_data ):
                PrintLog( "LoadVectorFileUsingPredicateList() -   Found Unique CUI: \"" + str( label ) + "\", Assigning Index: " + str( unique_cui_index ) )
                unique_cui_data[ label ] = unique_cui_index

                PrintLog( "LoadVectorFileUsingPredicateList() -   Appending CUI: \"" + str( label ) + "\" Vector Data To Embedding Matrix" )
                cui_embedding_matrix.append( data )
                loaded_cui_elements += len( data.split( " " ) )
                unique_cui_index    += 1

            if( label in predicate_list and label not in unique_predicate_data ):
                PrintLog( "LoadVectorFileUsingPredicateList() -   Found Unique Predicate: \"" + str( label ) + "\", Assigning Index: " + str( unique_predicate_index ) )
                unique_predicate_data[ label ] = unique_predicate_index

                PrintLog( "LoadVectorFileUsingPredicateList() -   Adjusting Remaining Indices In Predicate" )

                # Account For Specified Vector Index In Vector "Index,Value" Data Vs Relative Index
                # The First Element Of The Predicate Data Is Subtracted From The First Known Element
                # To Generate A Relative Index Vs The Actual Specified Index
                #    Ex: 10,1 -> 0,1 If The First Known Predicate Starts At Index 10
                data = data.split( " " )
                predicate_index_adjust_value = unique_cui_index + unique_predicate_index - 2
                
                for i in range( len( data ) ):
                    temp = data[i].split( ":" )
                    PrintLog( "LoadVectorFileUsingPredicateList() -   Old Index: " + str( temp[0] ) + ", New Index: " + str( int( temp[0] ) - predicate_index_adjust_value ) )
                    temp[0] = str( int( temp[0] ) - predicate_index_adjust_value )
                    data[i] = ":".join( temp )

                data = " ".join( data )

                PrintLog( "LoadVectorFileUsingPredicateList() -   Appending Predicate: \"" + str( label ) + "\" Vector Data To Embedding Matrix" )
                predicate_embedding_matrix.append( data )
                loaded_predicate_elements += len( data.split( " " ) )
                unique_predicate_index    += 1

        PrintLog( "LoadVectorFileUsingPredicateList() - Assigning CUI Vector Length/Predicate Vector Length Values" )

        cui_vector_length       = loaded_cui_elements - 1
        predicate_vector_length = loaded_predicate_elements - 1

        PrintLog( "LoadVectorFileUsingPredicateList() -   CUI Vector Length       : " + str( cui_vector_length ) )
        PrintLog( "LoadVectorFileUsingPredicateList() -   Predicate Vector Length : " + str( predicate_vector_length ) )

    PrintLog( "LoadVectorFileUsingPredicateList() - Assigning Number Of CUIs/Number Of Predicates Values" )

    number_of_cuis       = len( unique_cui_data )
    number_of_predicates = len( unique_predicate_data )

    PrintLog( "LoadVectorFileUsingPredicateList() -   Number Of CUIs                     : " + str(      number_of_cuis               ) )
    PrintLog( "LoadVectorFileUsingPredicateList() -   Number Of Predicates               : " + str(      number_of_predicates         ) )
    PrintLog( "LoadVectorFileUsingPredicateList() -   Number Of CUI Embedding Data       : " + str( len( cui_embedding_matrix       ) ) )
    PrintLog( "LoadVectorFileUsingPredicateList() -   Number Of Predicate Embedding Data : " + str( len( predicate_embedding_matrix ) ) )

    # Check(s)
    if( number_of_cuis == 0 ):
        PrintLog( "LoadVectorFileUsingPredicateList() - Error: No CUIs Loaded", 1 )
        return -1
    if( number_of_predicates == 0 ):
        PrintLog( "LoadVectorFileUsingPredicateList() - Error: No Predicates Loaded", 1 )
        return -1

    PrintLog( "LoadVectorFileUsingPredicateList() - Loaded " + str( loaded_cui_elements ) + " CUI Elements" )
    PrintLog( "LoadVectorFileUsingPredicateList() - Loaded " + str( loaded_predicate_elements ) + " Predicate Elements" )
    PrintLog( "LoadVectorFileUsingPredicateList() - Complete" )
    return 0

#   Re-Adjusts CUI and Predicate Vector Data Indices For Only Elements Found In The Training File
#   Used To Optimize Training Time Complexity. Set "<AdjustVectors>:1" In Config File.
def AdjustVectorIndexData():
    global steps
    global training_data
    global number_of_cuis
    global unique_cui_data
    global cui_vector_length
    global number_of_predicates
    global cui_dense_input_mode
    global cui_embedding_matrix
    global unique_predicate_data
    global train_file_data_length
    global predicate_vector_length
    global predicate_dense_input_mode
    global predicate_embedding_matrix

    # Check(s)
    if( adjust_for_unidentified_vectors is 0 ):
        PrintLog( "AdjustVectorIndexData() - Vector Adjustment Disabled" )
        return 0

    if( len( training_data ) == 0 ):
        PrintLog( "AdjustVectorIndexData() - Error: No Training Data Loaded In Memory", 1 )
        return -1

    # Variables
    cuis             = []
    predicates       = []
    found_cuis       = {}
    found_predicates = {}
    cui_pattern      = re.compile( "C[0-9]+" )

    PrintLog( "AdjustVectorIndexData() - Vector Adjustment Enabled" )
    PrintLog( "AdjustVectorIndexData() - Gathering All Unique Vectors From Training File" )

    for line in training_data:
        elements = re.split( r"\s+", line )

        # Only Add New Unique Concept Unique Identifier (CUI)
        cui_elements = filter( cui_pattern.match, elements )

        for element in cui_elements:
            if element not in cuis: cuis.append( element )

        # Only Add New Unique Predicates To The List
        #   Ex line: C001 ISA C002   <- elements[1] = Predicate / Relation Type (String)
        if elements[1] not in predicates: predicates.append( elements[1] )

    PrintLog( "AdjustVectorIndexData() - Found " + str( len( cuis ) ) + " Unique CUIs" )
    PrintLog( "AdjustVectorIndexData() - Found " + str( len( predicates ) ) + " Unique Predicates" )

    cuis.sort()
    predicates.sort()

    # Find All Existing Unique CUIs And Predicates In Vector Data From The Training Data
    PrintLog( "AdjustVectorIndexData() - Matching Training Data Unique CUIs/Predicates To Unique Vector CUI/Predicate Data" )

    for cui in cuis:
        if cui in unique_cui_data: found_cuis[cui] = unique_cui_data[cui]

    for predicate in predicates:
        if predicate in unique_predicate_data: found_predicates[predicate] = unique_predicate_data[predicate]


    PrintLog( "AdjustVectorIndexData() -   Original Total Unique CUIs             : " + str( len( unique_cui_data ) ) )
    PrintLog( "AdjustVectorIndexData() -   Original Total Unique Predicates       : " + str( len( unique_predicate_data ) ) )
    PrintLog( "AdjustVectorIndexData() -   New Adjusted Total Unique CUIs         : " + str( len( found_cuis ) ) )
    PrintLog( "AdjustVectorIndexData() -   New Adjusted Total Unique Predicates   : " + str( len( found_predicates ) ) )

    # Adjust Dense Weights From CUI/Predicate W2V Embeddings
    cui_embeddings       = []
    predicate_embeddings = []

    if( len( cui_embedding_matrix ) > 0 ):
        if( cui_dense_input_mode is True ):
            PrintLog( "AdjustVectorIndexData() - Adjusting CUI Dense Weight Vectors" )
            
            cui_embeddings.append( np.zeros( cui_vector_length ) )
            
            for cui in sorted( found_cuis ):
                cui_embeddings.append( cui_embedding_matrix[unique_cui_data[cui]] )
        else:
            PrintLog( "AdjustVectorIndexData() - Adjusting CUI Sparse Vectors" )
            
            cui_embeddings.append( "" )
            
            for found_cui in sorted( found_cuis ):
                vector = cui_embedding_matrix[unique_cui_data[found_cui]]
                
                # Set Hard-Coded CUI Indices To Actual CUI Names
                vector = vector.split()
                for i in range( len( vector ) ):
                    data    = vector[i].split( ":" )
                    
                    for cui, index in unique_cui_data.items():
                        if( index == int( data[0] ) + 1 ):
                            data[0] = cui
                            break
                    
                    separator = ":"
                    vector[i] = separator.join( data )
                    
                separator = " "
                cui_embeddings.append( separator.join( vector ) )

        PrintLog( "AdjustVectorIndexData() -   Original CUI Embedding Matrix Number Of Elements    : " + str( len( cui_embedding_matrix ) ) )
        if( cui_dense_input_mode is True ):
            cui_embedding_matrix = np.asarray( cui_embeddings, dtype = 'float32' )
        else:
            cui_embedding_matrix = cui_embeddings
        PrintLog( "AdjustVectorIndexData() -   New Adjusted CUI Embedding Matrix Number Of Elements: " + str( len( cui_embedding_matrix ) ) )

    if( len( predicate_embedding_matrix ) > 0 ):
        if( predicate_dense_input_mode is True ):
            PrintLog( "AdjustVectorIndexData() - Adjusting Predicate Dense Weight Vectors" )
            
            predicate_embeddings.append( np.zeros( predicate_vector_length ) )
            
            for predicate in sorted( found_predicates ):
                predicate_embeddings.append( predicate_embedding_matrix[unique_predicate_data[predicate]] )
        else:
            PrintLog( "AdjustVectorIndexData() - Adjusting Predicate Sparse Vectors" )
            
            predicate_embeddings.append( "" )
            
            for predicate in sorted( found_predicates ):
                vector = predicate_embedding_matrix[unique_predicate_data[predicate]]
                
                # Set Hard-Coded CUI Indices To Actual CUI Names
                vector = vector.split()
                for i in range( len( vector ) ):
                    data    = vector[i].split( ":" )
                    
                    for cui, index in unique_predicate_data.items():
                        if( index == int( data[0] ) + 1 ):
                            data[0] = cui
                            break
                    
                    separator = ":"
                    vector[i] = separator.join( data )
                    
                separator = " "
                predicate_embeddings.append( separator.join( vector ) )
        
        PrintLog( "AdjustVectorIndexData() -   Original Predicate Embedding Matrix Number Of Elements    : " + str( len( predicate_embedding_matrix ) ) )
        if( predicate_dense_input_mode is True ):
            predicate_embedding_matrix = np.asarray( predicate_embeddings, dtype = 'float32' )
        else:
            predicate_embedding_matrix = predicate_embeddings
        PrintLog( "AdjustVectorIndexData() -   New Adjusted Predicate Embedding Matrix Number Of Elements: " + str( len( predicate_embedding_matrix ) ) )

    PrintLog( "AdjustVectorIndexData() - Adjusting Unique CUI/Predicate Indices" )
    PrintLog( "AdjustVectorIndexData() - Setting Number Of CUIs And Number Of Predicates Variables To Newly Adjusted Values" )

    unique_index = 1

    PrintLog( "AdjustVectorIndexData() - Adjusting CUI Indices" )

    for cui in sorted( found_cuis ):
        PrintLog( "AdjustVectorIndexData() -   CUI: " + str( cui ) + ", Old Index: " + str( found_cuis[cui] ) + ", New Index: " + str( unique_index ) )
        found_cuis[cui] = unique_index
        unique_index += 1

    PrintLog( "AdjustVectorIndexData() - Setting \"found_cuis\" to \"unique_cui_data\"" )
    unique_cui_data = found_cuis
    number_of_cuis  = len( unique_cui_data )
    unique_index = 1
    
    if( cui_dense_input_mode is False ):
        PrintLog( "AdjustVectorIndexData() - Adjusting CUI Sparse Hard-Coded Vector Indices" )
        
        for i in range( len( cui_embeddings ) ):
            embedding = cui_embeddings[i]
            if( embedding == "" ): continue     # Skip First Element Place-Holder Element
            
            elements = embedding.split()
            
            for j in range( len( elements ) ):
                data = elements[j].split( ":" )
                data[0] = str( unique_cui_data[data[0]] - 1 )
                separator = ":"
                elements[j] = separator.join( data )
            
            separator = " "
            cui_embeddings[i] = separator.join( elements )

    PrintLog( "AdjustVectorIndexData() - Adjusting Predicate Indices" )

    for predicate in sorted( found_predicates ):
        PrintLog( "AdjustVectorIndexData() -   Predicate: " + str( predicate ) + ", Old Index: " + str( found_predicates[predicate] ) + ", New Index: " + str( unique_index ) )
        found_predicates[predicate] = unique_index
        unique_index += 1

    PrintLog( "AdjustVectorIndexData() - Setting \"found_predicates\" to \"unique_predicate_data\"" )
    unique_predicate_data = found_predicates
    number_of_predicates  = len( unique_predicate_data )
    
    if( predicate_dense_input_mode is False ):
        PrintLog( "AdjustVectorIndexData() - Adjusting Predicate Sparse Hard-Coded Vector Indices" )
        
        for i in range( len( predicate_embeddings ) ):
            embedding = predicate_embeddings[i]
            if( embedding == "" ): continue     # Skip First Element Place-Holder Element
            
            elements = embedding.split()
            
            for j in range( len( elements ) ):
                data = elements[j].split( ":" )
                data[0] = str( unique_predicate_data[data[0]] - 1 )
                separator = ":"
                elements[j] = separator.join( data )
            
            separator = " "
            predicate_embeddings[i] = separator.join( elements )

    PrintLog( "AdjustVectorIndexData() - Complete" )
    return 0

#   Fetches Concept Unique Identifier Data From The File
#   Adds Unique CUIs and Predicates / Relation Types To Hashes
#   Along With Unique Numeric Index Identification Values
#   Also Sets The Number Of Steps If Not Specified
def GetConceptUniqueIdentifierData():
    global training_data
    global number_of_cuis
    global evaluation_data
    global unique_cui_data
    global number_of_predicates
    global unique_predicate_data
    global train_file_data_length
    global process_eval_metrics_per_epoch

    # Load Training File
    if CheckIfFileExists( train_file ) == False:
        PrintLog( "GetConceptUniqueIdentifierData() - Error: Training Data File \"" + str( train_file ) + "\" Does Not Exist", 1 )
        return -1;

    # Read Concept Unique Identifier-Predicate Occurrence Data From File
    PrintLog( "GetConceptUniqueIdentifierData() - Reading Training Data File: \"" + str( train_file ) + "\"" )
    try:
        with open( train_file, "r" ) as in_file:
            training_data = in_file.readlines()
    except FileNotFoundError:
        PrintLog( "GetConceptUniqueIdentifierData() - Error: Unable To Open Training Data File \"" + str( train_file )+ "\"", 1 )
        return -1
    finally:
        in_file.close()

    PrintLog( "GetConceptUniqueIdentifierData() - Training File Data In Memory" )

    training_data = [ line.strip() for line in training_data ]    # Removes Trailing Space Characters From CUI Data Strings
    training_data.sort()
    
    train_file_data_length = len( training_data )
    
    PrintLog( "GetConceptUniqueIdentifierData() - Training File Data Length: " + str( train_file_data_length ) )
    
    # Load Evaluation File
    if( evaluation_file is not "" ):
        if CheckIfFileExists( evaluation_file ) == False:
            PrintLog( "GetConceptUniqueIdentifierData() - Error: Evaluation File \"" + str( evaluation_file ) + "\" Does Not Exist", 1 )
            return -1;
    
        # Read Concept Unique Identifier-Predicate Occurrence Data From File
        PrintLog( "GetConceptUniqueIdentifierData() - Reading Evaluation File: \"" + str( evaluation_file ) + "\"" )
        try:
            with open( evaluation_file, "r" ) as in_file:
                evaluation_data = in_file.readlines()
        except FileNotFoundError:
            PrintLog( "GetConceptUniqueIdentifierData() - Error: Unable To Open Evaluation File \"" + str( evaluation_file )+ "\"", 1 )
            return -1
        finally:
            in_file.close()
    
        PrintLog( "GetConceptUniqueIdentifierData() - Evaluation File Data In Memory" )
    
        evaluation_data = [ line.strip() for line in evaluation_data ]    # Removes Trailing Space Characters From CUI Data Strings
        evaluation_data.sort()
        
        PrintLog( "GetConceptUniqueIdentifierData() - Evaluation File Data Length: " + str( len( evaluation_data ) ) )
        
        if( len( evaluation_data ) == 0 or testing_stats_file == "" ):
            PrintLog( "GetConceptUniqueIdentifierData() - Warning: Evaluation File Could Not Be Loaded Or Testing Statistics File Is Empty String", 1 )
            PrintLog( "GetConceptUniqueIdentifierData() -          Testing Metrics Will Not Be Computed" )
            process_eval_metrics_per_epoch = 0
        else:
            process_eval_metrics_per_epoch = 1
            
    else:
        PrintLog( "GetConceptUniqueIdentifierData() - Warning: No Evaluation File Specified / Testing Metrics Will Not Be Computed" )
        process_eval_metrics_per_epoch = 0
    

    cui_data_loaded       = False
    predicate_data_loaded = False

    if( len( unique_cui_data       ) > 0 ): cui_data_loaded       = True
    if( len( unique_predicate_data ) > 0 ): predicate_data_loaded = True

    ###################################################
    #                                                 #
    #   Generate The Unique CUI and Predicate Lists   #
    #                                                 #
    ###################################################
    if( cui_data_loaded is True and predicate_data_loaded is True ):
        PrintLog( "GetConceptUniqueIdentifierData() - Unique CUI/Predicate Data Previously Generated" )
        PrintLog( "GetConceptUniqueIdentifierData() - Assigning Number Of CUIs/Number Of Predicates Values" )

        number_of_cuis       = len( unique_cui_data )
        number_of_predicates = len( unique_predicate_data )

        PrintLog( "GetConceptUniqueIdentifierData() - Number Of CUIs       : " + str( number_of_cuis ) )
        PrintLog( "GetConceptUniqueIdentifierData() - Number Of Predicates : " + str( number_of_predicates ) )
        PrintLog( "GetConceptUniqueIdentifierData() - Complete" )
        return 0

    PrintLog( "GetConceptUniqueIdentifierData() - Generating Unique CUI And/Or Predicate Data Lists" )
    
    # Append Nothing To First Element / Index "0"
    if( cui_data_loaded is False ):       cui_embedding_matrix.append( "" )
    if( predicate_data_loaded is False ): predicate_embedding_matrix.append( "" )

    cui_pattern     = re.compile( "C[0-9]+" )

    for line in training_data:
        elements = re.split( r"\s+", line )

        # Only Add New Unique Concept Unique Identifier (CUI)
        cui_elements = filter( cui_pattern.match, elements )

        for element in cui_elements:
            if element not in unique_cui_data and cui_data_loaded is False:
                PrintLog( "GetConceptUniqueIdentifierData() - Found Unique CUI: " + str( element ) )
                unique_cui_data[ element ] = 1

        # Only Add New Unique Predicates To The List
        #   Ex line: C001 ISA C002   <- elements[1] = Predicate / Relation Type (String)
        if elements[1] not in unique_predicate_data and predicate_data_loaded is False:
            PrintLog( "GetConceptUniqueIdentifierData() - Found Unique Predicate: " + str( elements[1] ) )
            unique_predicate_data[ elements[1] ] = 1    # Add Predicate / Relation Type To Unique List

    PrintLog( "GetConceptUniqueIdentifierData() - Assigning Number Of CUIs/Number Of Predicates Values" )

    if( cui_data_loaded is False ):       number_of_cuis       = len( unique_cui_data )
    if( predicate_data_loaded is False ): number_of_predicates = len( unique_predicate_data )

    PrintLog( "GetConceptUniqueIdentifierData() - List(s) Generated" )
    PrintLog( "GetConceptUniqueIdentifierData() - Unique Number Of CUIs: "       + str( number_of_cuis ) )
    PrintLog( "GetConceptUniqueIdentifierData() - Unique Number Of Predicates: " + str( number_of_predicates ) )

    # Sort CUIs/Predicates/Relation Types In Ascending Order And Assign Appropriate Identification Values
    # Generate "Index:Value" Data For GenerateNetworkData()
    PrintLog( "GetConceptUniqueIdentifierData() - Sorting List(s) In Ascending Order" )

    index = 1

    if( cui_data_loaded is False ):
        PrintLog( "GetConceptUniqueIdentifierData() - Sorting CUI List" )

        for cui in sorted( unique_cui_data.keys() ):
            PrintLog( "GetConceptUniqueIdentifierData() -   Assigning CUI: \"" + str( cui ) + "\", Index: " + str( index ) + ", Value: 1" )
            unique_cui_data[ cui ] = index

            PrintLog( "GetConceptUniqueIdentifierData() -   Appending CUI: \"" + str( cui ) + "\" -> \"Index:Value\" To Embedding Matrix" )
            cui_embedding_matrix.append( str( index - 1 ) + ":1" )
            index += 1
    else:
        PrintLog( "GetConceptUniqueIdentifierData() - Warning: CUI Data Already Exists In Memory / Loaded CUI Vector Previously?" )

    index = 1

    if( predicate_data_loaded is False ):
        PrintLog( "GetConceptUniqueIdentifierData() - Sorting Predicate List" )

        for predicate in sorted( unique_predicate_data.keys() ):
            PrintLog( "GetConceptUniqueIdentifierData() -   Assigning Predicate: \"" + str( predicate ) + "\", Index: " + str( index ) + ", Value: 1" )
            unique_predicate_data[ predicate ] = index

            PrintLog( "GetConceptUniqueIdentifierData() -   Appending Predicate: \"" + str( predicate ) + "\" -> \"Index:Value\" To Embedding Matrix" )
            predicate_embedding_matrix.append( str( index - 1 ) + ":1" )
            index += 1
    else:
        PrintLog( "GetConceptUniqueIdentifierData() - Warning: Predicate Data Already Exists In Memory / Loaded Predicate Vector Previously?" )

    PrintLog( "GetConceptUniqueIdentifierData() - Complete" )
    return 0

#   Print CUI and Predicate Key Files
def PrintKeyFiles():
    if( print_key_files is 1 ):
        output_name = ""
        
        # Get Output File Name
        if( output_file_name != "" ):
            PrintLog( "PrintKeyFiles() - Grabbing Output File Output File Name Variable" )
            output_name = output_file_name
        # Get Train File Name From Path
        else:
            PrintLog( "PrintKeyFiles() - Grabbing Output File Output File Name From Train File" )
            output_name = train_file.split( "/" )
            if( len( output_name ) == 1 ): output_name = train_file.split( "\\" )
            output_name = output_name[-1]
        
        PrintLog( "PrintKeyFiles() - Key File Printing Enabled" )
        PrintLog( "PrintKeyFiles() - Printing CUI Key File: " + output_name + ".cui_key" )

        cui_keys       = sorted( unique_cui_data.keys() )
        predicate_keys = sorted( unique_predicate_data.keys() )

        try:
            with open( output_name + ".cui_key", "w" ) as out_file:
                for cui in cui_keys:
                    if( cui_dense_input_mode is True ):
                        out_file.write( str( unique_cui_data[ cui ] ) + " " + str( cui ) + "\n" )
                    # Print Indices Hard-coded By Sparse Vectors (Needs Testing W/ Association Vectors)
                    else:
                        sparse_vtr = cui_embedding_matrix[ unique_cui_data[ cui ] ]
                        sparse_vtr = sparse_vtr.split( " " )
                        vtr_data   = sparse_vtr[0].split( ":" )
                        index      = vtr_data[0]
                        out_file.write( str( index ) + " " + str( cui ) + "\n" )
        except IOError:
            PrintLog( "PrintKeyFiles() - Error: Unable To Create CUI Key File", 1 )
            return -1
        finally:
            out_file.close()

        PrintLog( "PrintKeyFiles() - File Created" )
        PrintLog( "PrintKeyFiles() - Printing CUI Key File: " + output_name + ".predicate_key" )

        try:
            with open( output_name + ".predicate_key", "w" ) as out_file:
                for predicate in predicate_keys:
                    if( predicate_dense_input_mode is True ):
                        out_file.write( str( unique_predicate_data[ predicate ] ) + " " + str( predicate ) + "\n" )
                    # Print Indices Hard-coded By Sparse Vectors (Needs Testing W/ Association Vectors)
                    else:
                        sparse_vtr = predicate_embedding_matrix[ unique_predicate_data[ predicate ] ]
                        sparse_vtr = sparse_vtr.split( " " )
                        vtr_data   = sparse_vtr[0].split( ":" )
                        index      = vtr_data[0]
                        out_file.write( str( index ) + " " + str( predicate ) + "\n" )
        except IOError:
            PrintLog( "PrintKeyFiles() - Error: Unable To Create Predicate Key File", 1 )
            return -1
        finally:
            out_file.close()

        PrintLog( "PrintKeyFiles() - File Created" )
        PrintLog( "PrintKeyFiles() - Complete" )
    else:
        PrintLog( "PrintKeyFiles() - Print Key Files Disabled" )

    return 0

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

def GetSubjectOneHotVector( cui ):
    if( cui not in unique_cui_data ):
        PrintLog( "GetSubjectOneHotVector() - Error: CUI \"" + str( cui ) + "\" Not In Unique CUI Data List", 1 )
        return []
    
    vector = []
    
    if( cui_dense_input_mode is False ):
        vector        = np.zeros( number_of_cuis )
        sparse_data   = cui_embedding_matrix[ unique_cui_data[cui] ]
        sparse_data   = sparse_data.split( " " )
        
        for data in sparse_data:
            cui_data  = data.split( ":" )
            cui_index = int( cui_data[0] )
            cui_value = float( cui_data[1] )
            
            vector[cui_index] = cui_value
    
    return vector

def GetPredicateOneHotVector( predicate ):
    if( predicate not in unique_predicate_data ):
        PrintLog( "GetPredicateOneHotVector() - Error: Predicate \"" + str( predicate ) + "\" Not In Unique Predicate Data List", 1 )
        return []
    
    vector = []
    
    if( predicate_dense_input_mode is False ):
        vector       = np.zeros( number_of_predicates )
        sparse_data  = predicate_embedding_matrix[ unique_predicate_data[predicate] ]
        sparse_data  = sparse_data.split( " " )
        
        for data in sparse_data:
            predicate_data  = data.split( ":" )
            predicate_index = int( predicate_data[0] )
            predicate_value = float( predicate_data[1] )
            
            vector[predicate_index] = predicate_value
    
    return vector

def GetObjectOneHotCUIVector( cui ):
    if( cui not in unique_cui_data ):
        PrintLog( "GetObjectOneHotCUIVector() - Error: CUI \"" + str( cui ) + "\" Not In Unique CUI Data List", 1 )
        return []
    
    vector                   = np.zeros( number_of_cuis )
    object_cui_index         = int( unique_cui_data[cui] ) - 1
    vector[object_cui_index] = 1
    return vector

#   Fixes The Input For A Sparse Matrix
def Batch_Gen( X, Y, batch_size ):
    samples_per_epoch = train_file_data_length
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

#   Parses Through CUI Data And Generates Sparse Matrices For
#   Concept Input, Predicate Input and Concept Output Data
def GenerateNetworkData( passed_data = None, start_index = None, end_index = None ):
    global steps
    global batch_size
    global identified_cuis
    global unidentified_cuis
    global print_network_inputs
    global identified_predicates
    global unidentified_predicates
    global actual_train_data_length
    global curr_training_data_index
    global print_matrix_generation_stats

    # Check(s)
    if( train_file_data_length == 0 ):
        PrintLog( "GenerateNetworkData() - Error: No CUI-Predicate Co-occurrence Data In Memory / Was An Input CUI-Predicate Co-Occurrence File Read Before Calling Method?", 1 )
        return [], [], []
    if( len( unique_cui_data ) == 0 ):
        PrintLog( "GenerateNetworkData() - Error: No Unique CUI Data In Memory / Unique CUI Data Dictionary Empty", 1 )
        return [], [], []
    if( len( unique_predicate_data ) == 0 ):
        PrintLog( "GenerateNetworkData() - Erros: No Unique Predicate Data In Memory / Unique Predicate Data Dictionary Empty", 1 )
        return [], [], []

    PrintLog( "GenerateNetworkData() - Generating Network Matrices" )

    number_of_subject_cui_inputs = 0
    number_of_predicate_inputs   = 0
    number_of_object_cui_outputs = 0
    concept_input_indices        = []
    concept_input_values         = []
    predicate_input_indices      = []
    predicate_input_values       = []
    concept_output_indices       = []
    concept_output_values        = []

    # Parses Each Line Of Raw Data Input And Adds Them To Arrays For Matrix Generation
    PrintLog( "GenerateNetworkData() - Parsing CUI Data / Generating Network Input-Output Data Arrays" )

    temp_data               = []
    index                   = 0
    number_of_skipped_lines = 0
    
    # Use Passed Data By Parameter / Overrides Global "training_data"
    if( passed_data is not None ):
        # Fetch All Data In Passed Data Parameter
        if( start_index is None and end_index is None ):
            PrintLog( "GenerateNetworkData() - Parsing Entire Passed Data Parameter" )
            temp_data = passed_data
        # Fetch Passed Data In Batch Using Start/End Indices
        else:
            PrintLog( "GenerateNetworkData() - Passed Data Start Index: " + str( start_index ) + " - End Index: " + str( end_index ) )
            temp_data = passed_data[start_index:end_index] 
    # Use Training Data (global) Parameter
    else:
        # Fetch All Training Data
        if( start_index is None and end_index is None ):
            PrintLog( "GenerateNetworkData() - Warning: No Start-End Indices Specified / Generating Matrices Using All Training Data" )
            temp_data = training_data
        # Fetch Training Data In Batch Using Start/End Indices
        else:
            PrintLog( "GenerateNetworkData() - Training Data Start Index: " + str( start_index ) + " - End Index: " + str( end_index ) )
            temp_data = training_data[start_index:end_index]
            
            
    for i in range( len( temp_data ) ):
        not_found_flag = False
        line = temp_data[i]
        PrintLog( "GenerateNetworkData() - Parsing Line: " + str( line ) )
        line_elements = re.split( r"\s+", line )

        # Check(s)
        # If Subject CUI, Predicate and Object CUIs Are Not Found Within The Specified Unique CUI and Predicate Lists, Report Error and Skip The Line
        if( line_elements[0] not in unique_cui_data ):
            PrintLog( "GenerateNetworkData() - Warning: Subject CUI \"" + str( line_elements[0] ) + "\" Is Not In Unique CUI Data List / Skipping Line " + str( curr_training_data_index + i ) )
            not_found_flag = True
            if( line_elements[0] not in unidentified_cuis ): unidentified_cuis.append( line_elements[0] )
        else:
            if( line_elements[0] not in identified_cuis ):   identified_cuis.append( line_elements[0] )
        if( line_elements[1] not in unique_predicate_data ):
            PrintLog( "GenerateNetworkData() - Warning: Predicate \"" + str( line_elements[1] ) + "\" Is Not In Unique Predicate Data List / Skipping Line " + str( curr_training_data_index + i ) )
            not_found_flag = True
            if( line_elements[1] not in unidentified_predicates ): unidentified_predicates.append( line_elements[1] )
        else:
            if( line_elements[1] not in identified_predicates ): identified_predicates.append( line_elements[1] )
        for element in line_elements[2:]:
            if( element not in unique_cui_data ):
                PrintLog( "GenerateNetworkData() - Warning: Object CUI \"" + str( element ) + "\" Is Not In Unique CUI Data List / Skipping Line " + str( curr_training_data_index + i ) )
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

        # Add Unique Element Indices To Concept/Predicate Network Input/Value List
        # Add Subject CUI Index/Value
        # CUI Input Sparse Vector Support
        if( cui_dense_input_mode is False ):
            sparse_data  = cui_embedding_matrix[ subject_cui_index ]
            sparse_data  = sparse_data.split( " " )

            # Add Subject CUI Indices and Values
            for index_value_data in sparse_data:
                data = index_value_data.split( ":" )
                cui_vtr_index = int( data[0] )
                cui_vtr_value = float( data[1] )
                concept_input_indices.append( [ index, cui_vtr_index ] )
                concept_input_values.append( cui_vtr_value )

                PrintLog( "GenerateNetworkData() - Adding Index: " + str( index ) + ", Subject CUI Index: " + str( cui_vtr_index ) + ", Value: " + str( cui_vtr_value ) )

            number_of_subject_cui_inputs += 1
        
        # CUI Input Dense Vector Support
        else:
            PrintLog( "GenerateNetworkData() - Adding Index: " + str( index ) + ", Subject CUI Index: " + str( subject_cui_index ) )
            concept_input_indices.append( [subject_cui_index] )
            number_of_subject_cui_inputs += 1

        # Add Predicate Index/Value
        if( predicate_dense_input_mode is False ):
            predicate_data = predicate_embedding_matrix[ predicate_index ]
            predicate_data = predicate_data.split( " " )

            for index_value_element in predicate_data:
                index_value = index_value_element.split( ":" )
                pred_index  = int( index_value[0] )
                pred_value  = float( index_value[1] )
                predicate_input_indices.append( [ index, pred_index ] )
                predicate_input_values.append( pred_value )

                PrintLog( "GenerateNetworkData() - Adding Index: " + str( index ) + ", Predicate Index: " + str( pred_index ) + ", Value: " + str( pred_value ) )

            number_of_predicate_inputs += 1

        # Predicate Dense Vector Support
        else:
            PrintLog( "GenerateNetworkData() - Adding Index: " + str( index ) + ", Predicate Index: " + str( predicate_index ) )
            predicate_input_indices.append( [predicate_index] )
            number_of_predicate_inputs += 1
        
        
        # Adds All Object CUI Indices To The Output CUI Index Array
        for element in line_elements[2:]:
            object_cui_index = unique_cui_data[element] - 1
            PrintLog( "GenerateNetworkData() - Adding Index: " + str( index ) + ", Object CUI Index: " + str( object_cui_index ) + ", Value: 1" )
            concept_output_indices.append( [ index, object_cui_index ] )
            concept_output_values.append( 1 )
            number_of_object_cui_outputs += 1
        
        index += 1

    
    # Set Global Variable
    curr_training_data_index += len( temp_data )

    # Check(s)
    if( len( concept_input_indices ) is 0 ):
        PrintLog( "GenerateNetworkData() - Error: Concept Input Indices List Contains No Data / Specified Subject CUIs In The Input Not File Not Within Unique CUI Data?", 1 )
    if( len( predicate_input_indices ) is 0 ):
        PrintLog( "GenerateNetworkData() - Error: Predicate Input Indices List Contains No Data / Specified Predicates In The Input Not File Not Within Unique CUI Data?", 1 )
        PrintLog( "GenerateNetworkData() -        Note: This May Be Reporting Due To Concept Input Indices List Erroring Out" )
    if( len( concept_output_indices ) is 0 ):
        PrintLog( "GenerateNetworkData() - Error: Concept Output Indices List Contains No Data / Specified Object CUIs In The Input Not File Not Within Unique CUI Data?", 1 )
        PrintLog( "GenerateNetworkData() -        Note: This May Be Reporting Due To Concept Input or Predicate Indices List Erroring Out"                               , 1 )
    if( len( concept_input_indices ) is 0 or len( predicate_input_indices ) is 0 or len( concept_output_indices ) is 0 ): return None, None, None

    # @REMOVEME - Sets The Number Of Steps If Not Specified
    if( steps == 0 ):
        PrintLog( "GenerateNetworkData() - Warning: Number Of Steps Not Specified / Generating Value Based On Data Size" )
        steps = int( actual_train_data_length / batch_size ) + ( 0 if( actual_train_data_length % batch_size == 0 ) else 1 )
        PrintLog( "GenerateNetworkData() - Assigning \"Number Of Steps\" Value: " + str( steps ) )

    # Matrix Generation Stats
    if( print_matrix_generation_stats is 1 ):
        PrintLog( "GenerateNetworkData() - =========================================================" )
        PrintLog( "GenerateNetworkData() - =                Matrix Generation Stats                =" )
        PrintLog( "GenerateNetworkData() - =========================================================" )
        PrintLog( "GenerateNetworkData() -   Number Of Subject CUI Inputs              : " + str( number_of_subject_cui_inputs ) )
        PrintLog( "GenerateNetworkData() -   Number Of Predicate Inputs                : " + str( number_of_predicate_inputs ) )
        PrintLog( "GenerateNetworkData() -   Number Of Object CUI Outputs              : " + str( number_of_object_cui_outputs ) )
        PrintLog( "GenerateNetworkData() -   Number Of Concept Input Index Elements    : " + str( len( concept_input_indices ) ) )
        PrintLog( "GenerateNetworkData() -   Number Of Concept Input Value Elements    : " + str( len( concept_input_values ) ) )
        PrintLog( "GenerateNetworkData() -   Number Of Predicate Input Index Elements  : " + str( len( predicate_input_indices ) ) )
        PrintLog( "GenerateNetworkData() -   Number Of Predicate Input Value Elements  : " + str( len( predicate_input_values ) ) )
        PrintLog( "GenerateNetworkData() -   Number Of Concept Output Index Elements   : " + str( len( concept_output_indices ) ) )
        PrintLog( "GenerateNetworkData() -   Number Of Concept Output Value Elements   : " + str( len( concept_output_values ) ) )
        PrintLog( "GenerateNetworkData() -   Number Of Identified CUI Elements         : " + str( len( identified_cuis ) ) )
        PrintLog( "GenerateNetworkData() -   Number Of Identified Predicate Elements   : " + str( len( identified_predicates ) ) )
        PrintLog( "GenerateNetworkData() -   Number Of Unidentified CUI Elements       : " + str( len( unidentified_cuis ) ) )
        PrintLog( "GenerateNetworkData() -   Number Of Unidentified Predicate Elements : " + str( len( unidentified_predicates ) ) )
        PrintLog( "GenerateNetworkData() -   Total Unique CUIs                         : " + str( len( unique_cui_data ) ) )
        PrintLog( "GenerateNetworkData() -   Total Unique Predicates                   : " + str( len( unique_predicate_data ) ) )
        PrintLog( "GenerateNetworkData() -   Identified Input Data Array Length        : " + str( actual_train_data_length ) )
        PrintLog( "GenerateNetworkData() -   Number Of Skipped Lines In Training Data  : " + str( number_of_skipped_lines ) )
        PrintLog( "GenerateNetworkData() -   Total Input Data Array Length             : " + str( train_file_data_length ) )
        PrintLog( "GenerateNetworkData() - =========================================================" )
        PrintLog( "GenerateNetworkData() - =                                                       =" )
        PrintLog( "GenerateNetworkData() - =========================================================" )

    # Set Up Sparse Matrices To Include All Specified CUI/Predicate Vectors
    matrix_cui_length       = len( unique_cui_data )
    matrix_predicate_length = len( unique_predicate_data )   
    
    # Transpose The Arrays, Then Convert To Rows/Columns
    PrintLog( "GenerateNetworkData() - Transposing Index Data Arrays Into Row/Column Data" )
    if( cui_dense_input_mode       is False ): concept_input_row,   concept_input_column   = zip( *concept_input_indices   )
    if( predicate_dense_input_mode is False ): predicate_input_row, predicate_input_column = zip( *predicate_input_indices )
    concept_output_row,  concept_output_column  = zip( *concept_output_indices  )

    # Convert Row/Column Data Into Sparse Matrices
    PrintLog( "GenerateNetworkData() - Converting Index Data Into Matrices" )
    if( cui_dense_input_mode       is False ):
        concept_input_matrix   = sparse.csr_matrix( ( concept_input_values,           ( concept_input_row,   concept_input_column ) ),   shape = ( number_of_subject_cui_inputs, matrix_cui_length ) )
    else:
        concept_input_matrix   = np.asarray( concept_input_indices )
    
    if( predicate_dense_input_mode is False ):
        predicate_input_matrix = sparse.csr_matrix( ( predicate_input_values,         ( predicate_input_row, predicate_input_column ) ), shape = ( number_of_predicate_inputs,   matrix_predicate_length ) )
    else:
        predicate_input_matrix = np.asarray( predicate_input_indices )
    
    concept_output_matrix  = sparse.csr_matrix( ( [1]*len( concept_output_indices ),  ( concept_output_row,  concept_output_column ) ),  shape = ( number_of_subject_cui_inputs, matrix_cui_length ) )
    
    # Print Sparse Matrices
    if( print_network_inputs is 1 ):
        PrintLog( "GenerateNetworkData() - =========================================================" )
        PrintLog( "GenerateNetworkData() - =        Printing Compressed Row/Sparse Matrices        =" )
        PrintLog( "GenerateNetworkData() - =========================================================" )
        
        if( cui_dense_input_mode is True ):
            PrintLog( "GenerateNetworkData() - Subject CUI Index Sequence Inputs" )
            PrintLog( concept_input_indices )
        else:
            PrintLog( concept_input_matrix )
            PrintLog( "GenerateNetworkData() - Original Dense Formatted Sparse Matrix - Subject CUIs" )
            PrintLog( concept_input_matrix.todense() )
        
        if( predicate_dense_input_mode is True ):
            PrintLog( "GenerateNetworkData() - Predicate Sequence Index Sequence Inputs" )
            PrintLog( predicate_input_indices )
        else:
            PrintLog( "GenerateNetworkData() - Compressed Sparse Matrix - Predicates" )
            PrintLog( predicate_input_matrix )
            PrintLog( "GenerateNetworkData() - Original Dense Formatted Sparse Matrix - Predicates" )
            PrintLog( predicate_input_matrix.todense() )
        
        PrintLog( "GenerateNetworkMatrices() - Compressed Sparse Matrix - Object CUIs" )
        PrintLog( concept_output_matrix )
        PrintLog( "GenerateNetworkMatrices() - Original Dense Formatted Sparse Matrix - Object CUIs" )
        PrintLog( concept_output_matrix.todense() )
        PrintLog( "GenerateNetworkData() - =========================================================" )
        PrintLog( "GenerateNetworkData() - =                                                       =" )
        PrintLog( "GenerateNetworkData() - =========================================================" )
    
    PrintLog( "GenerateNetworkData() - Complete" )
    
    return concept_input_matrix, predicate_input_matrix, concept_output_matrix


"""
    Creates A Keras Neural Network Using Input/Output Sparse Matrices and Returns The Network

    Returns
    --------
     Model
      The neural network model
"""
def GenerateNeuralNetwork():
    # Build The Layers As Designed
    # Build CUI Input Layer / Type Of Model Depends On Sparse or Dense Vector/Matrix Use
    PrintLog( "GenerateNeuralNetwork() - Generating Input And Predicate Layers" )

    cui_input_layer         = None
    pred_input_layer        = None
    concept_layer           = None
    know_in                 = None
    know_layer_input_dim    = 0

    # CUI Input Layer To Concept Layer
    # Dense CUI Input
    if( cui_dense_input_mode is True ):
        PrintLog( "GenerateNeuralNetwork() - Generating \"CUI_OneHot_Input\" Layer, Type: Int32, Shape: ( 1, )" )
        cui_input_layer  = Input( shape = ( 1, ), dtype = 'int32', name = 'CUI_OneHot_Input' )                                        # CUI Input Layer

        PrintLog( "GenerateNeuralNetwork() - Generating \"CUI_Dense_Weights\" Embedding Layer, # Of Elements: " + str( number_of_cuis + 1 ) + ", Vector Length: " + str( cui_vector_length ) + ", Input Length: 1, Trainable: " + str( trainable_dense_embeddings ) )
        cui_word_embedding_layer = Embedding( number_of_cuis + 1, cui_vector_length, weights=[cui_embedding_matrix], input_length = 1, name = "CUI_Dense_Weights", trainable = trainable_dense_embeddings )( cui_input_layer )

        PrintLog( "GenerateNeuralNetwork() - Appending Flatten Layer: \"CUI_Weight_Dimensionality_Reduction\" To Embedding Layer: \"CUI_Dense_Weights\"" )
        cui_word_embedding_layer = Flatten( name = "CUI_Weight_Dimensionality_Reduction" )( cui_word_embedding_layer )

        PrintLog( "GenerateNeuralNetwork() - Generating Concept Layer: \"Concept_Representation\" From Embedding Layer \"CUI_Weight_Dimensionality_Reduction\", Units: " + str( layer_1_size ) + ", Activation: ReLU, Input Dim: " + str( number_of_cuis ) )
        concept_layer            = Dense( units = layer_1_size, activation = 'relu', input_dim = cui_vector_length, name = 'Concept_Representation' )( cui_word_embedding_layer )

    # Sparse CUI Input
    else:
        PrintLog( "GenerateNeuralNetwork() - Generating \"CUI_OneHot_Input\" Layer, Type: Float32, Shape: ( " + str( number_of_cuis ) + ", )" )
        cui_input_layer     = Input( shape = ( number_of_cuis, ), dtype = 'float32', name = 'CUI_OneHot_Input' )                                        # CUI Input Layer

        PrintLog( "GenerateNeuralNetwork() - Appending Concept Layer: \"Concept_Representation\" To Input Layer \"CUI_OneHot_Input\", Units: " + str( layer_1_size ) + ", Activation: ReLU, Input Dim: " + str( number_of_cuis ) )
        concept_layer       = Dense( units = layer_1_size, activation = 'relu', input_dim = number_of_cuis, name = 'Concept_Representation' )( cui_input_layer )

    # Adjust "know_layer" Input Dim
    know_layer_input_dim = layer_1_size

    # Predicate Layer
    # Dense Predicate Input
    if( predicate_dense_input_mode is True ):
        PrintLog( "GenerateNeuralNetwork() - Generating \"Predicate_OneHot_Input\" Layer, Type: Int32, Shape: ( 1, )" )
        pred_input_layer          = Input( shape = ( 1, ), dtype = 'int32', name = 'Predicate_OneHot_Input' )                                  # Predicate Input Layer

        PrintLog( "GenerateNeuralNetwork() - Generating \"Predicate_Dense_Weights\" Embedding Layer, # Of Elements: " + str( number_of_predicates + 1 ) + ", Vector Length: " + str( predicate_vector_length ) + ", Input Length: 1, Trainable: " + str( trainable_dense_embeddings ) )
        pred_word_embedding_layer = Embedding( number_of_predicates + 1, predicate_vector_length, weights=[predicate_embedding_matrix], input_length = 1, name = "Predicate_Dense_Weights", trainable = trainable_dense_embeddings )( pred_input_layer )

        PrintLog( "GenerateNeuralNetwork() - Appending Flatten Layer: \"Predicate_Weight_Dimensionality_Reduction\" To Embedding Layer: \"Predicate_Dense_Weights\"" )
        pred_word_embedding_layer = Flatten( name = "Predicate_Weight_Dimensionality_Reduction" )( pred_word_embedding_layer )

        PrintLog( "GenerateNeuralNetwork() - Generating Concatenate Layer: \"Knowledge_Input_Layer\" From Concatenating Concept Layer: \"Concept_Representation\" And Embedding Layer \"Predicate_Weight_Dimensionality_Reduction\"" )
        know_in                   = concatenate( [concept_layer, pred_word_embedding_layer], name = "Knowledge_Input_Layer" )                                                        # Concatenate The Predicate Layer To The CUI Layer

        # Adjust "know_layer" Input Dim
        know_layer_input_dim += predicate_vector_length

    # Sparse Predicate Input
    else:
        PrintLog( "GenerateNeuralNetwork() - Generating \"Predicate_OneHot_Input\" Layer, Type: Float32, Shape: ( " + str( number_of_predicates ) + ", )" )
        pred_input_layer     = Input( shape = ( number_of_predicates, ), dtype = 'float32', name = 'Predicate_OneHot_Input' )                                  # Predicate Input Layer

        PrintLog( "GenerateNeuralNetwork() - Generating Concatenate Layer: \"Knowledge_Input_Layer\" From Concept Layer: \"Concept_Representation\" And Predicate Input Layer \"Predicate_OneHot_Input\"" )
        know_in              = concatenate( [concept_layer, pred_input_layer], name = "Knowledge_Input_Layer" )                                                                 # Concatenate The Predicate Layer To The CUI Layer

        # Adjust "know_layer" Input Dim
        know_layer_input_dim += number_of_predicates

    # Build The Rest Of The Network Using The Above Layers
    PrintLog( "GenerateNeuralNetwork() - Generating Know_Layer: \"Knowledge_Representation_Layer\" From Know_In Layer: \"Knowledge_Input_Layer\", Units: " + str( layer_2_size ) + ", Activation: ReLU, Input Dim: " + str( know_layer_input_dim ) )
    know_layer       = Dense( units = layer_2_size, activation = 'relu', input_dim = know_layer_input_dim, name = "Knowledge_Representation_Layer" )( know_in )             # Knowledge Representation Layer

    PrintLog( "GenerateNeuralNetwork() - Generating Dropout Layer: \"Dropout_Layer\" From Know_Layer: \"Knowledge_Representation_Layer\"" )
    dropper          = Dropout( dropout_amt, name = "Dropout_Layer" )( know_layer )                                                                                         # Define The Dropout

    PrintLog( "GenerateNeuralNetwork() - Generating Output Layer: \"CUI_Output_Layer\" From Dropout Layer: \"Dropout_Layer\", Units: " + str( number_of_cuis ) + ", Activation: Sigmoid" )
    cui_output_layer = Dense( units = number_of_cuis, activation = 'sigmoid', name = 'CUI_Output_Layer' )( dropper )                                                        # Define The Output

    # Compile The Network Model Using Input and Output Layers
    PrintLog( "GenerateNeuralNetwork() - Generating Model From Input Layers: \"CUI_OneHot_Input\", \"Predicate_OneHot_Input\" And Output Layer: \"CUI_Output_Layer\"" )
    model = Model( inputs = [cui_input_layer, pred_input_layer], outputs = [cui_output_layer] )

    # Create The Optimizers And Metrics For The Output
    PrintLog( "GenerateNeuralNetwork() - Setting Optimzers: SGD, Learning Rate: " + str( learning_rate ) + ", Momentum: " + str( momentum ) )
    sgd = optimizers.SGD( lr = learning_rate, momentum = momentum )

    PrintLog( "GenerateNeuralNetwork() - Compiling Model Using, Loss: BCE, Optimizer: SGD, Metrics: Accuracy, Precision, Recall, Matthews Correlation" )
    # " loss = 'binary_crossentropy' " Is Also A Keras Built-In Option
    model.compile( loss = BCE, optimizer = sgd, metrics = ['accuracy', Precision, Recall, Matthews_Correlation] )

    # Print Model Summary
    PrintLog( "GenerateNeuralNetwork() - =========================================================" )
    PrintLog( "GenerateNeuralNetwork() - =                     Model Summary                     =" )
    PrintLog( "GenerateNeuralNetwork() - =========================================================" )

    model.summary( print_fn = lambda x:  PrintLog( "GenerateNeuralNetwork() - " + str( x ) ) )      # Capture Model.Summary()'s Print Output As A Function And Store In Variable 'x'

    PrintLog( "GenerateNeuralNetwork() - =========================================================" )
    PrintLog( "GenerateNeuralNetwork() - =                                                       =" )
    PrintLog( "GenerateNeuralNetwork() - =========================================================" )

    return model

#   Train The Neural Network
def TrainNeuralNetwork( neural_network_model = None, initial_epoch = None, concept_input = None, predicate_input =  None, concept_output = None ):
    
    # Check(s)
    if( neural_network_model is None ):
        PrintLog( "TrainNeuralNetwork() - Error: Model Has Not Been Instantiated / Model == None", 1 )
    if( concept_input is None ):
        PrintLog( "TrainNeuralNetwork() - Error: Concept Input Contains No Data", 1 )
    if( predicate_input is None ):
        PrintLog( "TrainNeuralNetwork() - Error: Predicate Input Contains No Data", 1 )
    if( concept_output is None ):
        PrintLog( "TrainNeuralNetwork() - Error: Concept Output Contains No Data", 1 )
    if( neural_network_model is None or concept_input is None or predicate_input is None or concept_output is None ):
        PrintLog( "TrainNeuralNetwork() - Empty Input(s) or Output Matrices / No Training Completed" )
        return -1
    
    if( initial_epoch is None ):
        PrintLog( "TrainNeuralNetwork() - Warning: Initial Epoch Value is None / Setting To \"0\"" )
        initial_epoch = 0
    if( type( initial_epoch ) is not int ):
        PrintLog( "TrainNeuralNetwork() - Error: \"initial_epoch\" Variable Is Not Type \"Int\"", 1 )
        return -1

    # Train The Model On The Inputs/Output
    PrintLog( "TrainNeuralNetwork() - Begin Model Training" )
    
    if( training_stats_file == "" ):
        csv_logger = CSVLogger( output_file_name + "_training_stats.txt", append = True, separator = '\t')
        neural_network_model.fit( [concept_input, predicate_input], concept_output, callbacks = [csv_logger], shuffle = shuffle_input_data, initial_epoch = initial_epoch, epochs = initial_epoch + 1 )
    else:
        neural_network_model.fit( [concept_input, predicate_input], concept_output, shuffle = shuffle_input_data, initial_epoch = initial_epoch, epochs = initial_epoch + 1 )

    PrintLog( "TrainNeuralNetwork() - Finished Model Training" )
    
    return 0

#   Given CUI Input, Predicate Input and CUI Output
#   Evaluate The Metrics Given The Data And Output Results
#       Return -> Loss, Accuracy, Precision, Recall, Matthews Correlation
def EvaluateNetworkData( model, cui_input, predicate_input, cui_output ):
    # Check(s)
    if( model is None ):
        PrintLog( "EvaluateNetworkData() - Error: Model Has Not Been Instantiated / Model == None", 1 )
    if( cui_input == "" ):
        PrintLog( "EvaluateNetworkData() - Error: Subject CUI Input Is Empty String", 1 )
    if( predicate_input == "" ):
        PrintLog( "EvaluateNetworkData() - Error: Predicate Input Is Empty String", 1 )
    if( cui_output == "" ):
        PrintLog( "EvaluateNetworkData() - Error: Object CUI Output Is Empty String", 1 )
    if( model is None or cui_input == "" or predicate_input == "" or cui_output == "" ):
        return -1, -1, -1, -1, -1
    
    cui_input_vectors       = []
    predicate_input_vectors = []
    cui_output_vectors      = []
    
    cui_input               = cui_input.split( " " )
    predicate_input         = predicate_input.split( " " )
    cui_output              = cui_output.split( " " )
    
    # Subject CUI Inputs
    for subject_cui in cui_input:
        # Sparse Vector Mode - Fetch Hard-Coded Vector Indices And Values
        #                      Create A One-Hot Vector And Assign Values
        if( cui_dense_input_mode is False ):
            vector = GetSubjectOneHotVector( subject_cui )
            if( len( vector ) > 0 ):
                cui_input_vectors.append( vector )
            else:
                PrintLog( "EvaluateNetworkData() - Warning: Subject CUI \"" + str( subject_cui ) + "\" Not In Unique CUI Data List / Skipping Element" )
        
        # Dense Vector Mode - Fetch CUI Index As Sequence Number And
        #                     Store As Array Of Sequence Value Per Input
        else:
            if( subject_cui in unique_cui_data ):
                cui_input_vectors.append( [unique_cui_data[subject_cui]] )
            else:
                PrintLog( "EvaluateNetworkData() - Warning: Subject CUI \"" + str( subject_cui ) + "\" Not In Unique CUI Data List / Skipping Element" )
        
    # Predicate Inputs
    for predicate in predicate_input:
        # Sparse Vector Mode - Fetch Hard-Coded Vector Indices And Values
        #                      Create A One-Hot Vector And Assign Values
        if( predicate_dense_input_mode is False ):
            vector = GetPredicateOneHotVector( predicate )
            if( len( vector ) > 0 ):
                predicate_input_vectors.append( vector )
            else:
                PrintLog( "EvaluateNetworkData() - Warning: Predicate \"" + str( predicate ) + "\" Not In Unique Predicate Data List / Skipping Element" )
        
        # Dense Vector Mode - Fetch CUI Index As Sequence Number And
        #                     Store As Array Of Sequence Value Per Input
        else:
            if( predicate in unique_predicate_data ):
                predicate_input_vectors.append( [unique_predicate_data[predicate]] )
            else:
                PrintLog( "EvaluateNetworkData() - Warning: Predicate \"" + str( predicate ) + "\" Not In Unique Predicate Data List / Skipping Element" )
        
    # Object CUI Outputs
    # Fetch Hard-Coded Vector Indices And Values
    # Create A One-Hot Vector And Assign Values
    for object_cui in cui_output:
        vector = GetObjectOneHotCUIVector( object_cui )
        
        if( len( vector ) > 0 ):
            cui_output_vectors.append( [vector] )
        else:
            PrintLog( "EvaluateNetworkData() - Warning: Object CUI \"" + str( subject_cui ) + "\" Not In Unique CUI Data List / Skipping Element" )
    
    # Check(s)
    if( len( cui_input_vectors ) == 0 or len( predicate_input_vectors ) == 0 or len( cui_output_vectors ) == 0 ):
        PrintLog( "EvaluateNetworkData() - Error: One Or More Network Input/Output Vectors == 0", 1 )
        return -1, -1, -1, -1, -1
    if( len( cui_input_vectors ) != len( predicate_input_vectors ) and len( cui_input_vectors ) != len( cui_output_vectors ) ):
        PrintLog( "EvaluateNetworkData() - Error: Subject CUI, Predicate and/or Object Vector Lengths Not Equal", 1 )
        return -1, -1, -1, -1, -1
    
    return model.evaluate( [cui_input_vectors, predicate_input_vectors], cui_output_vectors, verbose = 0 )

#   Given CUI Input And Predicate Input, Predict Output
#   Return -> Array Of Output Predictions
def NetworkPredictOutput( model, cui_input, predicate_input ):
    # Check(s)
    if( model is None ):
        PrintLog( "NetworkPredictOutput() - Error: Model Has Not Been Instantiated / Model == None", 1 )
    if( cui_input == "" ):
        PrintLog( "NetworkPredictOutput() - Error: Subject CUI Input Is Empty String", 1 )
    if( predicate_input == "" ):
        PrintLog( "NetworkPredictOutput() - Error: Predicate Input Is Empty String", 1 )
    if( model is None or cui_input == "" or predicate_input == "" ):
        return -1
    
    cui_input_vectors       = []
    predicate_input_vectors = []
    
    cui_input               = cui_input.split( " " )
    predicate_input         = predicate_input.split( " " )
    
    # Subject CUI Inputs
    for subject_cui in cui_input:
        # Sparse Vector Mode - Fetch Hard-Coded Vector Indices And Values
        #                      Create A One-Hot Vector And Assign Values
        if( cui_dense_input_mode is False ):
            vector = GetSubjectOneHotVector( subject_cui )
            if( len( vector ) > 0 ):
                cui_input_vectors.append( vector )
            else:
                PrintLog( "NetworkPredictOutput() - Warning: CUI \"" + str( subject_cui ) + "\" Not In Unique CUI Data List / Skipping Element" )
        
        # Dense Vector Mode - Fetch CUI Index As Sequence Number And
        #                     Store As Array Of Sequence Value Per Input
        else:
            if( subject_cui in unique_cui_data ):
                cui_input_vectors.append( [unique_cui_data[subject_cui]] )
            else:
                PrintLog( "NetworkPredictOutput() - Warning: CUI \"" + str( subject_cui ) + "\" Not In Unique CUI Data List / Skipping Element" )
        
    # Predicate Inputs
    for predicate in predicate_input:
        # Sparse Vector Mode - Fetch Hard-Coded Vector Indices And Values
        #                      Create A One-Hot Vector And Assign Values
        if( predicate_dense_input_mode is False ):
            vector = GetPredicateOneHotVector( predicate )
            if( len( vector ) > 0 ):
                predicate_input_vectors.append( vector )
            else:
                PrintLog( "NetworkPredictOutput() - Warning: Predicate \"" + str( predicate ) + "\" Not In Unique Predicate Data List / Skipping Element" )
        
        # Dense Vector Mode - Fetch CUI Index As Sequence Number And
        #                     Store As Array Of Sequence Value Per Input
        else:
            if( predicate in unique_predicate_data ):
                predicate_input_vectors.append( [unique_predicate_data[predicate]] )
            else:
                PrintLog( "NetworkPredictOutput() - Warning: Predicate \"" + str( predicate ) + "\" Not In Unique Predicate Data List / Skipping Element" )
        
    # Check(s)
    if( len( cui_input_vectors ) == 0 or len( predicate_input_vectors ) == 0 ):
        PrintLog( "NetworkPredictOutput() - Error: One Or More Network Input Vectors == 0", 1 )
        return -1
    if( len( cui_input_vectors ) != len( predicate_input_vectors ) ):
        PrintLog( "NetworkPredictOutput() - Error: CUI and Predicate Vector Lengths Not Equal", 1 )
        return -1
    
    return model.predict( [cui_input_vectors, predicate_input_vectors], verbose = 0 )

#   Trains The Neural Network Using Specified Concept/Predicate Input/Output Matrix Parameters
def ProcessNeuralNetwork():
    global curr_training_data_index
    
    eval_cui_input_matrix        = None
    eval_cui_output_matrix       = None
    train_cui_input_matrix       = None
    train_cui_output_matrix      = None
    training_stats_file_handle   = None
    eval_predicate_input_matrix  = None
    train_predicate_input_matrix = None
    
    # Train The Neural Network Using Specified Input/Output Matrices
    PrintLog( "ProcessNeuralNetwork() - Generating Actual Neural Network" )
    model = GenerateNeuralNetwork()
    PrintLog( "ProcessNeuralNetwork() - Finished Generating Neural Network" )
    
    # Generating File Handle and Matrices For Complete Training Metrics Per Epoch
    if( training_stats_file != "" ):
        PrintLog( "ProcessNeuralNetwork() - Creating Training Statistics File" )
        OpenTrainingStatsFileHandle()
        
        WriteStringToTrainingStatsFile( "Epoch\tLoss\tAccuracy\tPrecision\tRecall\tMatthews_Correlation" )
        
        PrintLog( "ProcessNeuralNetwork() - Generating Training CUI Input, Predicate Input and CUI Output Matrices/Sequences" )
        train_cui_input_matrix, train_predicate_input_matrix, train_cui_output_matrix = GenerateNetworkData()
    
    # Generating File Handle and Matrices For Complete Testing Metrics Per Epoch
    if( process_eval_metrics_per_epoch is 1 ):
        PrintLog( "ProcessNeuralNetwork() - Creating Testing Statistics File" )
        OpenTestingStatsFileHandle()
        
        WriteStringToTestingStatsFile( "Epoch\tLoss\tAccuracy\tPrecision\tRecall\tMatthews_Correlation" )
        
        PrintLog( "ProcessNeuralNetwork() - Generating Evaluation CUI Input, Predicate Input and CUI Output Matrices/Sequences" )
        eval_cui_input_matrix, eval_predicate_input_matrix, eval_cui_output_matrix = GenerateNetworkData( evaluation_data )
    
    # Network Runtime Variables
    start_time                   = time.time()
    weight_dump_counter          = weight_dump_interval
    curr_training_data_index     = 0
    number_of_remaining_elements = train_file_data_length
    
    # Train The Network Using Training Data
    for curr_epoch in range( number_of_epochs ):
        while( number_of_remaining_elements > 0 ):
            PrintLog( "ProcessNeuralNetwork() - Current Epoch: " + str( curr_epoch ) + "/" + str( number_of_epochs ) )
            cui_input, predicate_input, cui_output = GenerateNetworkData( None, curr_training_data_index, curr_training_data_index + batch_size )
            
            PrintLog( "ProcessNeuralNetwork() - Passing Input/Output Sequences/Matrices To Network" )
            TrainNeuralNetwork( model, curr_epoch, cui_input, predicate_input, cui_output )
            
            number_of_remaining_elements -= batch_size
            PrintLog( "ProcessNeuralNetwork() - Remaining Elements In Epoch: " + str( number_of_remaining_elements ) )
            PrintLog( "ProcessNeuralNetwork() - Finished Iteration" )
            
            if( debug_log is 0 ): print( "\n" )
        
        PrintLog( "ProcessNeuralNetwork() - Finished Epoch: " + str( curr_epoch ) + " Training" )
        
        weight_dump_counter          += 1
        curr_training_data_index     = 0
        number_of_remaining_elements = train_file_data_length
        
        # Neural Network Weight Dumping Per Epoch
        if( weight_dump_interval > 0 ):
            PrintLog( "ProcessNeuralNetwork() - Current Weight Dump Counter: " + str( weight_dump_counter ) )
            
            # Save The Weights Of The Current Network To A File Based Upon "weight_dump_interval" Variable
            if( weight_dump_counter >= weight_dump_interval ):
                    PrintLog( "ProcessNeuralNetwork() - Weight Dump Interval Met Or Exceeded / Saving Neural Network Weights For Epoch: " + str( curr_epoch ) )
                    PrintLog( "ProcessNeuralNetwork() -     Saving Current Epoch Network Weights To File: \"" + output_file_name + "_epoch_" + str( curr_epoch ) + "_model_weights.h5" + "\"" )
                    model.save_weights( output_file_name + "_epoch_" + str( curr_epoch ) + "_model_weights.h5" )
                    weight_dump_counter = 0
        
        # Compute Training Statistics Based On The Entire Training Dataset
        if( training_stats_file != "" ):
            PrintLog( "ProcessNeuralNetwork() - Evaluating Current Network Using Training Evaluation Data" )
            loss, accuracy, precision, recall, matthews_correlation = model.evaluate( [ train_cui_input_matrix, train_predicate_input_matrix ], train_cui_output_matrix )
            
            PrintLog( "ProcessNeuralNetwork() - Epoch " + str( curr_epoch ) + " - Complete Metrics: Loss: " + str( loss )
                     + "\tAccuracy: " + str( accuracy ) + "\tPrecision: " + str( precision ) + "\tRecall: " + str( recall )
                     + "\tMatthews_Correlation: " + str( matthews_correlation ) )
            WriteStringToTrainingStatsFile( str( curr_epoch ) + "\t" + str( loss ) + "\t" + str( accuracy ) + "\t"
                                           + str( precision ) + "\t" + str( recall ) + "\t" + str( matthews_correlation ) )
    
        # Compute Testing Statistics Based On Evaluation File
        if( process_eval_metrics_per_epoch is 1 ):
            PrintLog( "ProcessNeuralNetwork() - Evaluating Current Network Using Testing Evaluation Data" )
            loss, accuracy, precision, recall, matthews_correlation = model.evaluate( [ eval_cui_input_matrix, eval_predicate_input_matrix ], eval_cui_output_matrix )
            
            PrintLog( "ProcessNeuralNetwork() - Epoch " + str( curr_epoch ) + " - Complete Metrics: Loss: " + str( loss )
                     + "\tAccuracy: " + str( accuracy ) + "\tPrecision: " + str( precision ) + "\tRecall: " + str( recall )
                     + "\tMatthews_Correlation: " + str( matthews_correlation ) )
            WriteStringToTestingStatsFile( str( curr_epoch ) + "\t" + str( loss ) + "\t" + str( accuracy ) + "\t"
                                           + str( precision ) + "\t" + str( recall ) + "\t" + str( matthews_correlation ) )
    
    # Clean Up Training Statistics Variables
    if( training_stats_file != "" ):
        PrintLog( "ProcessNeuralNetwork() - Cleaning Up Training Statistics Variables" )
        CloseTrainingStatsFileHandle()
        train_cui_input_matrix       = None
        train_predicate_input_matrix = None
        train_cui_ouput_matrix       = None
        
    # Clean Up Testing Statistics Variables
    if( process_eval_metrics_per_epoch is 1 ):
        PrintLog( "ProcessNeuralNetwork() - Cleaning Up Testing Statistics Variables" )
        CloseTestingStatsFileHandle()
        eval_cui_input_matrix       = None
        eval_predicate_input_matrix = None
        eval_cui_output_matrix      = None

    PrintLog( "ProcessNeuralNetwork() - Training Time: %s secs" % ( time.time() - start_time ), 1 )

    # Save Network Model
    PrintLog( "ProcessNeuralNetwork() - Saving Model: \"" + output_file_name + "_model.h5\"", 1 )
    model.save( output_file_name + "_model.h5" )

    # Save Model Architechture in JSON Format
    PrintLog( "ProcessNeuralNetwork() - Saving Model Architecture: \"" + output_file_name + "_model_architecture.json\"", 1 )
    with open( output_file_name + '_model_architecture.json', 'w' ) as out_file:
        out_file.write( model.to_json() )          # Same as trained_nn.get_config()
    out_file.close()

    # Save Trained Network Weights
    PrintLog( "ProcessNeuralNetwork() - Saving Model Weights: \"" + output_file_name + "_model_weights.h5\"", 1 )
    model.save_weights( output_file_name + "_model_weights.h5" )

    # Print Model Depiction To File
    PrintLog( "ProcessNeuralNetwork() - Generating Visual Model: \"" + output_file_name + "_model_visual.png\"", 1 )
    plot_model( model, to_file = output_file_name + '_model_visual.png' )
    
    # Model Evaluation Test
    # Get First CUI, First Predicate And First CUI As Input, Input and Output (Although It May Not Be Correct), Forward Propagate Through The Network To Predict Accuracy
    if( test_input_cui != "" and test_input_predicate != "" and test_output_cui != "" ):
        PrintLog( "ProcessNeuralNetwork() - Evaluating Model", 1 )
        PrintLog( "ProcessNeuralNetwork() - Evaluating Inputs - CUI: \"" + str( test_input_cui ) + "\" and Predicate: \"" + str( test_input_predicate ) + "\" Versus Output CUI: \"" + str( test_output_cui ) + "\"", 1 )
        loss, accuracy, precision, recall, matthews_correlation = EvaluateNetworkData( model, test_input_cui, test_input_predicate, test_output_cui )
        PrintLog( "ProcessNeuralNetwork() - Loss: " + str( loss ) + " - Accuracy: " + str( accuracy ) + " - Precision: " + str( precision ) + " - Matthews Correlation: " + str( matthews_correlation ), 1 )
    
    if( test_input_cui != "" and test_input_predicate != "" ):
        PrintLog( "ProcessNeuralNetwork() - Predicting Output Given Inputs -> CUI: \"" + str( test_input_cui ) + "\" and Predicate: \"" + str( test_input_predicate ) + "\"", 1 )
        predicted_cui_indices = NetworkPredictOutput( model, test_input_cui, test_input_predicate )
        PrintLog( "ProcessNeuralNetwork() - Predicted CUI Indices: " + str( predicted_cui_indices ), 1 )
    
    return 0

#   Removes Unused Data From Memory
def CleanUp():
    global training_data
    global evaluation_data
    global identified_cuis
    global unique_cui_data
    global unidentified_cuis
    global cui_embedding_matrix
    global eval_cui_input_matrix
    global identified_predicates
    global unique_predicate_data
    global eval_cui_output_matrix
    global unidentified_predicates
    global predicate_embedding_matrix
    global eval_predicate_input_matrix

    PrintLog( "CleanUp() - Removing Data From Memory" )

    unique_cui_data              = None
    training_data                = None
    cui_embedding_matrix         = None
    unique_predicate_data        = None
    predicate_embedding_matrix   = None

    identified_cuis              = None
    identified_predicates        = None
    unidentified_cuis            = None
    unidentified_predicates      = None
    
    evaluation_data              = None
    eval_cui_input_matrix        = None
    eval_predicate_input_matrix  = None
    eval_cui_output_matrix       = None

    gc.collect()

    PrintLog( "CleanUp() - Complete" )


############################################################################################
#                                                                                          #
#    Main                                                                                  #
#                                                                                          #
############################################################################################

# Check(s)
if( len( sys.argv ) < 2 ):
    print( "Main() - Error: No Configuration File Argument Specified" )
    exit()
    
def Main():
    if( ReadConfigFile( sys.argv[1] ) == -1 ): exit()
    
    # Read CUIs and Predicates From The Same Vector Data File
    if( concept_vector_file is not "" and concept_vector_file == predicate_vector_file ):
        if( LoadVectorFileUsingPredicateList( concept_vector_file, predicate_list_file ) == -1 ):
            exit()
    
    # Read CUIs and Predicates From Differing Vector Data Files
    else:
        cui_file_loaded       = LoadVectorFile( concept_vector_file,   True )
        predicate_file_loaded = LoadVectorFile( predicate_vector_file, False )
    
        if( cui_file_loaded == -1 or predicate_file_loaded == -1 ):
            PrintLog( "Main() - Warning: Unable To Load Vector File(s) / Auto-Generating One-Hot Vectors Using Co-Occurrence Data (CNDA Mode)", 1 )
    
    # Set Numpy Print Options/Length To "MAXSIZE" ( Used To Debug GenerateNetworkData() Function )    @REMOVEME
    # np.set_printoptions( threshold = sys.maxsize )
    
    if( GetConceptUniqueIdentifierData() == -1 ):
        PrintLog( "Main() - Error: Failed To Auto-Generate Sparse CUI/Predicate Data" )
        exit()
    
    if( AdjustVectorIndexData() == -1 ):
        PrintLog( "Main() - Error: Failed To Adjust CUI/Predicate Indexing Data" )
        exit()
    
    if( PrintKeyFiles() == -1 ):
        PrintLog( "Main() - Error: Failed To Print CUI/Predicate Data Key Files" )
        exit()
    
    ProcessNeuralNetwork()
    
    # Garbage Collection / Free Unused Memory
    CleanUp()
    
    CloseDebugFileHandle()
    
    print( "~Fin" )


# Main Function Call To Run TrainNN
if __name__ == "__main__":
    Main()
