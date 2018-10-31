############################################################################################
#                                                                                          #
#    Neural Network - Literature Based Discovery                                           #
#    -------------------------------------------                                           #
#                                                                                          #
#    Date:    10/02/2018                                                                   #
#    Revised: 10/27/2018                                                                   #
#                                                                                          #
#    Generates A Neural Network Using A Configuration File.                                #
#      - Supports Dense and Sparse Input Vectors In All Combinations Of CUI and            #
#        Predicate Input                                                                   #
#      - Outputs The Trained Network Model, Model Weights, Architecture and Visual         #
#        Model Depiction                                                                   #
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
import keras
import keras.backend as K
#from tensorflow.python import keras
from keras.models import Model, model_from_json, load_model
from keras import optimizers
from keras.utils.vis_utils import plot_model
from keras.layers import Dense, Activation, Input, concatenate, Dropout, Embedding, Flatten
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
version                         = 0.31
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
show_prediction                 = 0
train_file                      = ""
concept_output_file             = ""
test_input_cui                  = ""
test_output_cui                 = ""
concept_vector_file             = ""
predicate_vector_file           = ""
predicate_list_file             = ""
print_input_matrices            = 0
print_matrix_generation_stats   = 0
adjust_for_unidentified_vectors = 0
cui_dense_input_mode            = True
predicate_dense_input_mode      = False
train_file_data_length          = 0
actual_train_data_length        = 0
cui_vector_length               = 0
predicate_vector_length         = 0

# CUI/Predicate Data
training_data                 = []
unique_cui_data               = {}
unique_predicate_data         = {}
cui_embedding_matrix          = []
predicate_embedding_matrix    = []

# Stats Variables
identified_cuis               = []             # CUIs Found In Unique CUI List During Matrix Generation
identified_predicates         = []             # Predicates Found In Unique CUI List During Matrix Generation
unidentified_cuis             = []             # CUIs Not Found In Unique CUI List During Matrix Generation
unidentified_predicates       = []             # Predicates Not Found In Unique CUI List During Matrix Generation

# Debug Log Variables
debug_log                        = True
write_log                        = False
debug_file_name                  = "nnlbd_log.txt"
debug_log_file                   = None


############################################################################################
#                                                                                          #
#    Sub-Routines                                                                          #
#                                                                                          #
############################################################################################

def testMe():
    print("TRAINN --- YEEEEE!")

def testMe2():
    return "s'mores poptart"

#   Print Statements To Console, Debug Log File Or Both
def PrintLog( print_str ):
    if( debug_log is 1 ): print( str( print_str ) )
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
    global test_output_cui
    global print_key_files
    global show_prediction
    global number_of_epochs
    global concept_output_file
    global concept_vector_file
    global predicate_list_file
    global negative_sample_rate
    global print_input_matrices
    global predicate_vector_file
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
        line = re.sub( r'<|>|\n', '', line )
        data = line.split( ":" )

        if data[0] == "DebugLog"            : debug_log                       = int( data[1] )
        if data[0] == "WriteLog"            : write_log                       = int( data[1] )
        if data[0] == "Momentum"            : momentum                        = float( data[1] )
        if data[0] == "BatchSize"           : batch_size                      = int( data[1] )
        if data[0] == "DropoutAMT"          : dropout_amt                     = float( data[1] )
        if data[0] == "Layer1Size"          : layer_1_size                    = int( data[1] )
        if data[0] == "Layer2Size"          : layer_2_size                    = int( data[1] )
        if data[0] == "LearningRate"        : learning_rate                   = float( data[1] )
        if data[0] == "TestInputCUI"        : test_input_cui                  = str( data[1] )
        if data[0] == "TestOutputCUI"       : test_output_cui                 = str( data[1] )
        if data[0] == "NumberOfSteps"       : steps                           = int( data[1] )
        if data[0] == "NumberOfEpochs"      : number_of_epochs                = int( data[1] )
        if data[0] == "TrainFile"           : train_file                      = str( data[1] )
        if data[0] == "NegativeSampleRate"  : negative_sample_rate            = int( data[1] )
        if data[0] == "PrintKeyFiles"       : print_key_files                 = int( data[1] )
        if data[0] == "ShowPrediction"      : show_prediction                 = int( data[1] )
        if data[0] == "ConceptVectorFile"   : concept_vector_file             = str( data[1] )
        if data[0] == "PredicateVectorFile" : predicate_vector_file           = str( data[1] )
        if data[0] == "PredicateListFile"   : predicate_list_file             = str( data[1] )
        if data[0] == "PrintInputMatrices"  : print_input_matrices            = int( data[1] )
        if data[0] == "PrintMatrixStats"    : print_matrix_generation_stats   = int( data[1] )
        if data[0] == "AdjustVectors"       : adjust_for_unidentified_vectors = int( data[1] )

    f.close()

    OpenDebugFileHandle()

    # Check(s)
    if( train_file is "" ):
        PrintLog( "ReadConfigFile() - Error: \"TrainFile\" Not Specified" )
    if( CheckIfFileExists( train_file ) is False ):
        PrintLog( "ReadConfigFile() - Error: \"" + str( train_file ) + "\" Does Not Exist" )
        exit()
    if( batch_size is 0 ):
        PrintLog( "ReadConfigFile() - Error: Batch_Size Variable Cannot Be <= \"0\" / Exiting Program" )
        exit()
    if( concept_vector_file != "" and concept_vector_file == predicate_vector_file and ( predicate_list_file is "" or predicate_list_file is None ) ):
        PrintLog( "ReadConfigFile() - Error: When \"ConceptVectorFile\" == \"PredicateVectorFile\"" )
        PrintLog( "ReadConfigFile() -        A Valid Predicate List Must Be Specified" )
        exit()

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

    PrintLog( "    Train File              : " + str( train_file ) )
    PrintLog( "    Concept Vector File     : " + str( concept_vector_file ) )
    PrintLog( "    Predicate Vector File   : " + str( predicate_vector_file ) )
    PrintLog( "    Predicate List File     : " + str( predicate_list_file ) )
    PrintLog( "    Batch Size              : " + str( batch_size ) )
    PrintLog( "    Learning Rate           : " + str( learning_rate ) )
    PrintLog( "    Number Of Epochs        : " + str( number_of_epochs ) )
    PrintLog( "    Number Of Steps         : " + str( steps ) )
    PrintLog( "    Momentum                : " + str( momentum ) )
    PrintLog( "    Dropout AMT             : " + str( dropout_amt ) )
    PrintLog( "    Layer 1 Size            : " + str( layer_1_size ) )
    PrintLog( "    Layer 2 Size            : " + str( layer_2_size ) )
    PrintLog( "    Negative Sample Rate    : " + str( negative_sample_rate ) )
    PrintLog( "    Print Key Files         : " + str( print_key_files ) )
    PrintLog( "    Print Input Matrices    : " + str( print_input_matrices ) )
    PrintLog( "    Print Matrix Stats      : " + str( print_matrix_generation_stats ) )
    PrintLog( "    Test Input CUI          : " + str( test_input_cui ) )
    PrintLog( "    Test Output CUI         : " + str( test_output_cui ) )
    PrintLog( "    Adjust Vectors          : " + str( adjust_for_unidentified_vectors ) )

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
        PrintLog( "LoadVectorFile() - Warning: No Vector File Specified" )
        return -1

    if( CheckIfFileExists( vector_file_path ) is False ):
        PrintLog( "LoadVectorFile() - Error: Specified File \"" + str( vector_file_path ) + "\" Does Not Exist" )
        return -1

    PrintLog( "LoadVectorFile() - Loading Vector File: \"" + vector_file_path +   "\"" )
    vector_data = None

    try:
        with open( vector_file_path, "r" ) as in_file:
            vector_data = in_file.readlines()
            vector_data.sort()
    except FileNotFoundError:
        PrintLog( "LoadVectorFile() - Error: Unable To Open File \"" + str( vector_file_path ) + "\"" )
        return -1
    finally:
        in_file.close()

    # Check(s)
    if( vector_data is None ):
        PrintLog( "LoadVectorFile() - Error: Failed To Load Vector Data" )
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
        unique_index = 0

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
                
            # Parse CUI/Predicate Data
            else:
                label = vector[0]
                data  = vector[1:]

                # Add Unique CUI Index And Data
                if( is_cui_vectors is True and label not in unique_cui_data ):
                    PrintLog( "LoadVectorFile() -   Found Unique CUI: \"" + str( label ) + "\", Assigning Index: " + str( unique_index ) )
                    unique_cui_data[ label ] = unique_index
                    
                    PrintLog( "LoadVectorFile() -   Appending CUI: \"" + str( label ) + "\" Vector Data To Embedding Matrix" )
                    cui_embedding_matrix.append( data )
                    loaded_elements += 1

                # Add Unique Predicate Index And Data
                if( is_cui_vectors is False and label not in unique_predicate_data ):
                    PrintLog( "LoadVectorFile() -   Found Unique Predicate: \"" + str( label ) + "\", Assigning Index: " + str( unique_index ) )
                    unique_predicate_data[ label ] = unique_index
                    
                    PrintLog( "LoadVectorFile() -   Appending Predicate: \"" + str( label ) + "\" Vector Data To Embedding Matrix" )
                    predicate_embedding_matrix.append( data )
                    loaded_elements += 1

                unique_index += 1

        PrintLog( "LoadVectorFile() - Assigning Number Of CUIs/Number Of Predicates Values" )
        if( is_cui_vectors is True  ): number_of_cuis       = len( unique_cui_data )
        if( is_cui_vectors is False ): number_of_predicates = len( unique_predicate_data )
        
        PrintLog( "LoadVectorFile() -   Number Of CUIs       : " + str( number_of_cuis ) )
        PrintLog( "LoadVectorFile() -   Number Of Predicates : " + str( number_of_predicates ) )

    # Read Sparse Formatted Vector File
    else:
        unique_index = 0
        
        PrintLog( "LoadVectorFile() - Parsing Sparse Vector Data" )

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
            number_of_cuis          = loaded_elements
            cui_vector_length       = loaded_elements
        if( is_cui_vectors is False ):
            number_of_predicates    = loaded_elements
            predicate_vector_length = loaded_elements
        
        PrintLog( "LoadVectorFile() -   CUI Vector Length                  : " + str(      cui_vector_length       ) )
        PrintLog( "LoadVectorFile() -   Predicate Vector Length            : " + str(      predicate_vector_length ) )
        PrintLog( "LoadVectorFile() -   Number Of CUIs                     : " + str(      number_of_cuis          ) )
        PrintLog( "LoadVectorFile() -   Number Of Predicates               : " + str(      number_of_predicates    ) )
        PrintLog( "LoadVectorFile() -   Number Of CUI Embedding Data       : " + str( len( cui_embedding_matrix )  ) )
        PrintLog( "LoadVectorFile() -   Number Of Predicate Embedding Data : " + str( len( cui_embedding_matrix )  ) )

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
        PrintLog( "LoadVectorFileUsingPredicateList() - Error: No Vector File Specified" )
        return -1

    if CheckIfFileExists( vector_file_path ) is False:
        PrintLog( "LoadVectorFileUsingPredicateList() - Error: Specified File \"" + str( vector_file_path ) + "\" Does Not Exist" )
        return -1

    if( predicate_list_path is None or predicate_list_path is "" ):
        PrintLog( "LoadVectorFileUsingPredicateList() - Warning: No Predicate List File Specified" )
        return -1

    if CheckIfFileExists( predicate_list_path ) is False:
        PrintLog( "LoadVectorFileUsingPredicateList() - Error: Specified File \"" + str( predicate_list_path ) + "\" Does Not Exist" )
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
            PrintLog( "LoadVectorFileUsingPredicateList() - Error: Unable To Open File \"" + str( predicate_list_file ) + "\"" )
            return -1
        finally:
            in_file.close()

    # Check(s)
    if( predicate_list is None ):
        PrintLog( "LoadVectorFileUsingPredicateList() - Error: Failed To Predicate List Data" )
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
        PrintLog( "LoadVectorFileUsingPredicateList() - Error: Unable To Open File \"" + str( vector_file_path ) + "\"" )
        return -1
    finally:
        in_file.close()

    # Check(s)
    if( vector_data is None ):
        PrintLog( "LoadVectorFileUsingPredicateList() - Error: Failed To Load Vector Data" )
        return -1
    else:
        PrintLog( "LoadVectorFileUsingPredicateList() - Loaded " + str( len( vector_data ) ) + " Vector Elements" )

    # Choose The Second Element In The Vector Data And Check Vector Format (CUI/Predicate)
    if( len( vector_data ) > 1 ):
        cui_dense_input_mode       = IsDenseVectorFormat( vector_data[2].strip() )
        predicate_dense_input_mode = cui_dense_input_mode
        if( cui_dense_input_mode == True ):  PrintLog( "LoadVectorFileUsingPredicateList() - Detected Dense CUI/Predicate Vector Format" )
        if( cui_dense_input_mode == False ): PrintLog( "LoadVectorFileUsingPredicateList() - Detected Sparse CUI/Predicate Vector Format" )

    loaded_cui_elements       = 0
    loaded_predicate_elements = 0
    
    PrintLog( "LoadVectorFileUsingPredicateList() - Parsing Vector Data" )
    
    # Read Dense Vector Formatted File
    if( cui_dense_input_mode == True or predicate_dense_input_mode == True ):
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
                
            # Parse CUI/Predicate Data
            else:
                label = vector[0]
                data  = vector[1:]

                # Add Unique CUI Index And Data
                if( label not in predicate_list and label not in unique_cui_data ):
                    PrintLog( "LoadVectorFileUsingPredicateList() -   Found Unique CUI: \"" + str( label ) + "\", Assigning Index: " + str( loaded_cui_elements ) )
                    unique_cui_data[ label ] = loaded_cui_elements
                    
                    PrintLog( "LoadVectorFileUsingPredicateList() -   Appending CUI: \"" + str( label ) + "\" Vector Data To Embedding Matrix" )
                    cui_embedding_matrix.append( data )
                    loaded_cui_elements += 1

                # Add Unique Predicate Index And Data
                if( label in predicate_list and label not in unique_predicate_data ):
                    PrintLog( "LoadVectorFileUsingPredicateList() -   Found Unique Predicate: \"" + str( label ) + "\", Assigning Index: " + str( loaded_predicate_elements ) )
                    unique_predicate_data[ label ] = loaded_predicate_elements
                    
                    PrintLog( "LoadVectorFileUsingPredicateList() -   Appending Predicate: \"" + str( label ) + "\" Vector Data To Embedding Matrix" )
                    predicate_embedding_matrix.append( data )
                    loaded_predicate_elements += 1

    # Read Sparse Formatted Vector File
    else:
        PrintLog( "LoadVectorFileUsingPredicateList() - Parsing Sparse Vector Data" )
        
        unique_index          =  0
        first_predicate_index = -1

        for vector in vector_data:
            vector = vector.strip()
            vector = vector.split( "<>" )
            label  = vector[0]
            data    = " ".join( vector[1:] )
            data    = re.sub( r',', ':', data )

            if( label not in predicate_list and label not in unique_cui_data ):
                PrintLog( "LoadVectorFileUsingPredicateList() -   Found Unique CUI: \"" + str( label ) + "\", Assigning Index: " + str( loaded_cui_elements ) )
                unique_cui_data[ label ] = unique_index
                
                PrintLog( "LoadVectorFileUsingPredicateList() -   Appending CUI: \"" + str( label ) + "\" Vector Data To Embedding Matrix" )
                cui_embedding_matrix.append( data )
                loaded_cui_elements += len( data.split( " " ) )
                unique_index        += 1

            if( label in predicate_list and label not in unique_predicate_data ):
                if( first_predicate_index == -1 ): first_predicate_index = unique_index
                PrintLog( "LoadVectorFileUsingPredicateList() -   Found Unique Predicate: \"" + str( label ) + "\", Assigning Index: " + str( unique_index - first_predicate_index ) )
                unique_predicate_data[ label ] = unique_index - first_predicate_index

                PrintLog( "LoadVectorFileUsingPredicateList() -   Adjusting Remaining Indices In Predicate" )
                
                # Account For Specified Vector Index In Vector "Index,Value" Data Vs Relative Index
                # The First Element Of The Predicate Data Is Subtracted From The First Known Element
                # To Generate A Relative Index Vs The Actual Specified Index
                #    Ex: 10,1 -> 0,1 If The First Known Predicate Starts At Index 10
                data = data.split( " " )
                
                for i in range( len( data ) ):
                    temp = data[i].split( ":" )
                    PrintLog( "LoadVectorFileUsingPredicateList() -   Old Index: " + str( temp[0] ) + ", New Index: " + str( int( temp[0] ) + first_predicate_index ) )
                    temp[0] = str( int( temp[0] ) - first_predicate_index )
                    data[i] = ":".join( temp )

                data = " ".join( data )

                PrintLog( "LoadVectorFileUsingPredicateList() -   Appending Predicate: \"" + str( label ) + "\" Vector Data To Embedding Matrix" )
                predicate_embedding_matrix.append( data )
                loaded_predicate_elements += len( data.split( " " ) )
                unique_index              += 1
        
        PrintLog( "LoadVectorFileUsingPredicateList() - Assigning CUI Vector Length/Predicate Vector Length Values" )
        
        cui_vector_length       = loaded_cui_elements
        predicate_vector_length = loaded_predicate_elements
        
        PrintLog( "LoadVectorFileUsingPredicateList() -   CUI Vector Length       : " + str( cui_vector_length ) )
        PrintLog( "LoadVectorFileUsingPredicateList() -   Predicate Vector Length : " + str( predicate_vector_length ) )

    PrintLog( "LoadVectorFileUsingPredicateList() - Assigning Number Of CUIs/Number Of Predicates Values" )

    number_of_cuis       = len( unique_cui_data )
    number_of_predicates = len( unique_predicate_data )
    
    PrintLog( "LoadVectorFileUsingPredicateList() -   Number Of CUIs                     : " + str(      number_of_cuis         ) )
    PrintLog( "LoadVectorFileUsingPredicateList() -   Number Of Predicates               : " + str(      number_of_predicates   ) )
    PrintLog( "LoadVectorFileUsingPredicateList() -   Number Of CUI Embedding Data       : " + str( len( cui_embedding_matrix ) ) )
    PrintLog( "LoadVectorFileUsingPredicateList() -   Number Of Predicate Embedding Data : " + str( len( cui_embedding_matrix ) ) )

    # Check(s)
    if( number_of_cuis == 0 ):
        PrintLog( "LoadVectorFileUsingPredicateList() - Error: No CUIs Loaded" )
        return -1
    if( number_of_predicates == 0 ):
        PrintLog( "LoadVectorFileUsingPredicateList() - Error: No Predicates Loaded" )
        return -1

    PrintLog( "LoadVectorFileUsingPredicateList() - Loaded " + str( loaded_cui_elements ) + " CUI Elements" )
    PrintLog( "LoadVectorFileUsingPredicateList() - Loaded " + str( loaded_predicate_elements ) + " Predicate Elements" )
    PrintLog( "LoadVectorFileUsingPredicateList() - Complete" )
    return 0

#   Re-Adjusts CUI and Predicate Data For All Indices Not Found In The Training File
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
        PrintLog( "AdjustVectorIndexData() - Error: No Training Data Loaded In Memory" )
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
    PrintLog( "AdjustVectorIndexData() -   New Adjusted Total Unique Predicates   : " + str( len( found_cuis ) ) )
    PrintLog( "AdjustVectorIndexData() -   New Adjusted Total Unique Predicates   : " + str( len( found_predicates ) ) )

    # Adjust Dense Weights From CUI/Predicate W2V Embeddings
    cui_embeddings       = []
    predicate_embeddings = []

    if( len( cui_embedding_matrix ) > 0 ):
        PrintLog( "AdjustVectorIndexData() - Adjusting CUI Sparse Input Vectors/Dense Weight Vectors" )
        for cui in sorted( found_cuis ):
            cui_embeddings.append( cui_embedding_matrix[unique_cui_data[cui]] )

        PrintLog( "AdjustVectorIndexData() -   Original CUI Embedding Matrix Number Of Elements    : " + str( len( cui_embedding_matrix ) ) )
        cui_embedding_matrix = cui_embeddings
        PrintLog( "AdjustVectorIndexData() -   New Adjusted CUI Embedding Matrix Number Of Elements: " + str( len( cui_embedding_matrix ) ) )

    if( len( predicate_embedding_matrix ) > 0 ):
        PrintLog( "AdjustVectorIndexData() - Adjusting Predicate Sparse Input Vectors/Dense Weight Vectors" )
        for predicate in sorted( found_predicates ):
            predicate_embeddings.append( predicate_embedding_matrix[unique_predicate_data[predicate]] )

        PrintLog( "AdjustVectorIndexData() -   Original Predicate Embedding Matrix Number Of Elements    : " + str( len( predicate_embedding_matrix ) ) )
        predicate_embedding_matrix = predicate_embeddings
        PrintLog( "AdjustVectorIndexData() -   New Adjusted Predicate Embedding Matrix Number Of Elements: " + str( len( predicate_embedding_matrix ) ) )

    PrintLog( "AdjustVectorIndexData() - Adjusting Unique CUI/Predicate Indices" )
    PrintLog( "AdjustVectorIndexData() - Setting Number Of CUIs And Number Of Predicates Variables To Newly Adjusted Values" )

    unique_index = 0

    PrintLog( "AdjustVectorIndexData() - Adjusting CUI Indices" )

    for cui in sorted( found_cuis ):
        PrintLog( "AdjustVectorIndexData() -   CUI: " + str( cui ) + ", Old Index: " + str( found_cuis[cui] ) + ", New Index: " + str( unique_index ) )
        found_cuis[cui] = unique_index
        unique_index += 1

    PrintLog( "AdjustVectorIndexData() - Setting \"found_cuis\" to \"unique_cui_data\"" )
    unique_cui_data = found_cuis
    number_of_cuis  = len( unique_cui_data )
    unique_index = 0

    PrintLog( "AdjustVectorIndexData() - Adjusting Predicate Indices" )

    for predicate in sorted( found_predicates ):
        PrintLog( "AdjustVectorIndexData() -   Predicate: " + str( predicate ) + ", Old Index: " + str( found_predicates[predicate] ) + ", New Index: " + str( unique_index ) )
        found_predicates[predicate] = unique_index
        unique_index += 1

    PrintLog( "AdjustVectorIndexData() - Setting \"found_predicates\" to \"unique_predicate_data\"" )
    unique_predicate_data = found_predicates
    number_of_predicates  = len( unique_predicate_data )

    PrintLog( "AdjustVectorIndexData() - Complete" )
    return 0

#   Fetches Concept Unique Identifier Data From The File
#   Adds Unique CUIs and Predicates / Relation Types To Hashes
#   Along With Unique Numeric Index Identification Values
#   Also Sets The Number Of Steps If Not Specified
def GetConceptUniqueIdentifierData():
    global training_data
    global number_of_cuis
    global unique_cui_data
    global number_of_predicates
    global unique_predicate_data
    global train_file_data_length

    if CheckIfFileExists( train_file ) == False:
        PrintLog( "GetConceptUniqueIdentifierData() - Error: CUI Data File \"" + str( train_file ) + "\" Does Not Exist" )
        return -1;

    # Read Concept Unique Identifier-Predicate Occurrence Data From File
    PrintLog( "GetConceptUniqueIdentifierData() - Reading Concept Input File: \"" + str( train_file ) + "\"" )
    try:
        with open( train_file, "r" ) as in_file:
            training_data = in_file.readlines()
    except FileNotFoundError:
        PrintLog( "GetConceptUniqueIdentifierData() - Error: Unable To Open File \"" + str( train_file )+ "\"" )
        return -1
    finally:
        in_file.close()

    PrintLog( "GetConceptUniqueIdentifierData() - File Data In Memory" )

    training_data = [ line.strip() for line in training_data ]    # Removes Trailing Space Characters From CUI Data Strings
    training_data.sort()

    train_file_data_length = len( training_data )


    ###################################################
    #                                                 #
    #   Generate The Unique CUI and Predicate Lists   #
    #                                                 #
    ###################################################
    if( len( unique_cui_data ) > 0 and len( unique_predicate_data ) > 0 ):
        PrintLog( "GetConceptUniqueIdentifierData() - Unique CUI/Predicate Data Previously Generated" )
        PrintLog( "GetConceptUniqueIdentifierData() - Assigning Number Of CUIs/Number Of Predicates Values" )
        
        number_of_cuis       = len( unique_cui_data )
        number_of_predicates = len( unique_predicate_data )
        
        PrintLog( "GetConceptUniqueIdentifierData() - Number Of CUIs       : " + str( number_of_cuis ) )
        PrintLog( "GetConceptUniqueIdentifierData() - Number Of Predicates : " + str( number_of_predicates ) )
        PrintLog( "GetConceptUniqueIdentifierData() - Complete" )
        return 0

    PrintLog( "GetConceptUniqueIdentifierData() - Generating Unique CUI and Predicate Data Lists" )

    cui_pattern     = re.compile( "C[0-9]+" )

    for line in training_data:
        elements = re.split( r"\s+", line )

        # Only Add New Unique Concept Unique Identifier (CUI)
        cui_elements = filter( cui_pattern.match, elements )

        for element in cui_elements:
            if element not in unique_cui_data:
                PrintLog( "GetConceptUniqueIdentifierData() - Found Unique CUI: " + str( element ) )
                unique_cui_data[ element ] = 1

        # Only Add New Unique Predicates To The List
        #   Ex line: C001 ISA C002   <- elements[1] = Predicate / Relation Type (String)
        if elements[1] not in unique_predicate_data:
            PrintLog( "GetConceptUniqueIdentifierData() - Found Unique Predicate: " + str( elements[1] ) )
            unique_predicate_data[ elements[1] ] = 1    # Add Predicate / Relation Type To Unique List

    PrintLog( "GetConceptUniqueIdentifierData() - Assigning Number Of CUIs/Number Of Predicates Values" )
    
    number_of_cuis       = len( unique_cui_data )
    number_of_predicates = len( unique_predicate_data )

    PrintLog( "GetConceptUniqueIdentifierData() - Lists Generated" )
    PrintLog( "GetConceptUniqueIdentifierData() - Unique Number Of CUIs: " + str( number_of_cuis ) )
    PrintLog( "GetConceptUniqueIdentifierData() - Unique Number Of Predicates: " + str( number_of_predicates ) )

    # Sort CUIs/Predicates/Relation Types In Ascending Order And Assign Appropriate Identification Values
    # Generate "Index:Value" Data For GenerateNetworkMatrices()
    PrintLog( "GetConceptUniqueIdentifierData() - Sorting Lists In Ascending Order" )

    PrintLog( "GetConceptUniqueIdentifierData() - Sorting CUI List" )
    index = 0
    
    for cui in sorted( unique_cui_data.keys() ):
        PrintLog( "GetConceptUniqueIdentifierData() -   Assigning CUI: \"" + str( cui ) + "\", Index: " + str( index ) + ", Value: 1" )
        unique_cui_data[ cui ] = index
        
        PrintLog( "GetConceptUniqueIdentifierData() -   Appending CUI: \"" + str( cui ) + "\" -> \"Index:Value\" To Embedding Matrix" )
        cui_embedding_matrix.append( str( index ) + ":1" )
        index += 1

    PrintLog( "GetConceptUniqueIdentifierData() - Sorting Predicate List" )
    index = 0
    
    for predicate in sorted( unique_predicate_data.keys() ):
        PrintLog( "GetConceptUniqueIdentifierData() -   Assigning Predicate: \"" + str( predicate ) + "\", Index: " + str( index ) + ", Value: 1" )
        unique_predicate_data[ predicate ] = index
        
        PrintLog( "GetConceptUniqueIdentifierData() -   Appending Predicate: \"" + str( predicate ) + "\" -> \"Index:Value\" To Embedding Matrix" )
        predicate_embedding_matrix.append( str( index ) + ":1" )
        index += 1

    PrintLog( "GetConceptUniqueIdentifierData() - Complete" )
    return 0

#   Print CUI and Predicate Key Files
def PrintKeyFiles():
    if( print_key_files is 1 ):
        PrintLog( "PrintKeyFiles() - Key File Printing Enabled" )
        PrintLog( "PrintKeyFiles() - Printing CUI Key File: " + train_file + ".cui_key" )

        cui_keys       = sorted( unique_cui_data.keys() )
        predicate_keys = sorted( unique_predicate_data.keys() )

        try:
            with open( train_file + ".cui_key", "w" ) as out_file:
                for cui in cui_keys:
                    out_file.write( str( unique_cui_data[ cui ] ) + " " + str( cui ) + "\n" )
        except IOError:
            PrintLog( "PrintKeyFiles() - Error: Unable To Create CUI Key File" )
            return -1
        finally:
            out_file.close()

        PrintLog( "PrintKeyFiles() - File Created" )
        PrintLog( "PrintKeyFiles() - Printing CUI Key File: " + train_file + ".predicate_key" )

        try:
            with open( train_file + ".predicate_key", "w" ) as out_file:
                for predicate in predicate_keys:
                    out_file.write( str( unique_predicate_data[ predicate ] ) + " " + str( predicate ) + "\n" )
        except IOError:
            PrintLog( "PrintKeyFiles() - Error: Unable To Create Predicate Key File" )
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

#   Turns An Input CUI Into A One-Hot Vector
def CUI_To_One_Hot( cui ):
    one_hot_cui = np.array( [0.0] * number_of_cuis )
    one_hot_cui[unique_cui_data[cui]] = 1.0
    return one_hot_cui

#   Turns The Predicate Into A One-Hot Vector Based On The Dictionary
def Predicate_To_One_Hot( predicate ):
    one_hot_predicate = np.array( [0.0] * number_of_predicates )
    one_hot_predicate[ unique_predicate_data[ predicate ] ] = 1.0
    return one_hot_predicate

#   Fixes The Input For A Sparse Matrix
def Batch_Gen( X, Y, batch_size ):
    samples_per_epoch = actual_train_data_length
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
def GenerateNetworkMatrices():
    global steps
    global identified_cuis
    global unidentified_cuis
    global print_input_matrices
    global identified_predicates
    global unidentified_predicates
    global actual_train_data_length
    global print_matrix_generation_stats

    # Check(s)
    if( train_file_data_length == 0 ):
        PrintLog( "GenerateNetworkMatrices() - Error: No CUI Data In Memory / Was An Input CUI File Read Before Calling Method?" )
        return None, None, None

    PrintLog( "GenerateNetworkMatrices() - Generating Network Matrices" )

    number_of_unique_cui_inputs       = 0
    number_of_unique_predicate_inputs = 0
    concept_input_indices             = []
    concept_input_values              = []
    predicate_input_indices           = []
    predicate_input_values            = []
    concept_output_indices            = []
    concept_output_values             = []

    # Parses Each Line Of Raw Data Input And Adds Them To Arrays For Matrix Generation
    PrintLog( "GenerateNetworkMatrices() - Parsing CUI Data / Generating Network Input-Output Data Arrays" )

    index = 0
    number_of_skipped_lines = 0

    for i in range( train_file_data_length ):
        not_found_flag = False
        line = training_data[i]
        line_elements = re.split( r"\s+", line )

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
                
                PrintLog( "GenerateNetworkMatrices() - Adding Index: " + str( index ) + ", Subject CUI Index: " + str( cui_vtr_index ) + ", Value: " + str( cui_vtr_value ) )

            number_of_unique_cui_inputs += 1

            # Add Predicate Input Indices and Values
            predicate_data = predicate_embedding_matrix[ predicate_index ]
            predicate_data = predicate_data.split( " " )

            for index_value_element in predicate_data:
                index_value = index_value_element.split( ":" )
                pred_index  = int( index_value[0] )
                pred_value  = float( index_value[1] )
                predicate_input_indices.append( [ index, pred_index ] )
                predicate_input_values.append( pred_value )
                
                PrintLog( "GenerateNetworkMatrices() - Adding Index: " + str( index ) + ", Predicate Index: " + str( cui_vtr_index ) + ", Value: " + str( cui_vtr_value ) )

            number_of_unique_predicate_inputs += 1

        # Dense Vector Support
        else:
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

    # Note: "number_of_unique_cui_inputs/number_of_unique_predicate_input" should equal "index" After Parsing. Otherwise There Will Be Consequences

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

    # Sets The Number Of Steps If Not Specified
    if( steps == 0 ):
        PrintLog( "GenerateNetworkMatrices() - Warning: Number Of Steps Not Specified / Generating Value Based On Data Size" )
        steps = int( actual_train_data_length / batch_size ) + ( 0 if( actual_train_data_length % batch_size == 0 ) else 1 )
        PrintLog( "GenerateNetworkMatrices() - Assigning \"Number Of Steps\" Value: " + str( steps ) )

    # Matrix Generation Stats
    if( print_matrix_generation_stats is 1 ):
        PrintLog( "GenerateNetworkMatrices() - =========================================================" )
        PrintLog( "GenerateNetworkMatrices() - =                Matrix Generation Stats                =" )
        PrintLog( "GenerateNetworkMatrices() - =========================================================" )
        PrintLog( "GenerateNetworkMatrices() -   Number Of Unique CUI Inputs               : " + str( number_of_unique_cui_inputs ) )
        PrintLog( "GenerateNetworkMatrices() -   Number Of Unique Predicate Inputs         : " + str( number_of_unique_predicate_inputs ) )
        PrintLog( "GenerateNetworkMatrices() -   Number Of Concept Input Index Elements    : " + str( len( concept_input_indices ) ) )
        PrintLog( "GenerateNetworkMatrices() -   Number Of Concept Input Value Elements    : " + str( len( concept_input_values ) ) )
        PrintLog( "GenerateNetworkMatrices() -   Number Of Predicate Input Index Elements  : " + str( len( predicate_input_indices ) ) )
        PrintLog( "GenerateNetworkMatrices() -   Number Of Predicate Input Value Elements  : " + str( len( predicate_input_values ) ) )
        PrintLog( "GenerateNetworkMatrices() -   Number Of Concept Output Index Elements   : " + str( len( concept_output_indices ) ) )
        PrintLog( "GenerateNetworkMatrices() -   Number Of Concept Output Value Elements   : " + str( len( concept_output_values ) ) )
        PrintLog( "GenerateNetworkMatrices() -   Number Of Identified CUI Elements         : " + str( len( identified_cuis ) ) )
        PrintLog( "GenerateNetworkMatrices() -   Number Of Identified Predicate Elements   : " + str( len( identified_predicates ) ) )
        PrintLog( "GenerateNetworkMatrices() -   Number Of Unidentified CUI Elements       : " + str( len( unidentified_cuis ) ) )
        PrintLog( "GenerateNetworkMatrices() -   Number Of Unidentified Predicate Elements : " + str( len( unidentified_predicates ) ) )
        PrintLog( "GenerateNetworkMatrices() -   Total Unique CUIs                         : " + str( len( unique_cui_data ) ) )
        PrintLog( "GenerateNetworkMatrices() -   Total Unique Predicates                   : " + str( len( unique_predicate_data ) ) )
        PrintLog( "GenerateNetworkMatrices() -   Identified Input Data Array Length        : " + str( actual_train_data_length ) )
        PrintLog( "GenerateNetworkMatrices() -   Number Of Skipped Lines In Training Data  : " + str( number_of_skipped_lines ) )
        PrintLog( "GenerateNetworkMatrices() -   Total Input Data Array Length             : " + str( train_file_data_length ) )
        PrintLog( "GenerateNetworkMatrices() - =========================================================" )
        PrintLog( "GenerateNetworkMatrices() - =                                                       =" )
        PrintLog( "GenerateNetworkMatrices() - =========================================================" )

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

    # Print Sparse Matrices
    if( print_input_matrices is 1 ):
        PrintLog( "GenerateNetworkMatrices() - =========================================================" )
        PrintLog( "GenerateNetworkMatrices() - =        Printing Compressed Row/Sparse Matrices        =" )
        PrintLog( "GenerateNetworkMatrices() - =========================================================" )
        PrintLog( "GenerateNetworkMatrices() - Compressed Sparse Matrix - Subject CUIs" )
        PrintLog( concept_input_matrix )
        PrintLog( "GenerateNetworkMatrices() - Original Dense Formatted Sparse Matrix - Subject CUIs" )
        PrintLog( concept_input_matrix.todense() )
        PrintLog( "GenerateNetworkMatrices() - Compressed Sparse Matrix - Predicates" )
        PrintLog( predicate_input_matrix )
        PrintLog( "GenerateNetworkMatrices() - Original Dense Formatted Sparse Matrix - Predicates" )
        PrintLog( predicate_input_matrix.todense() )
        PrintLog( "GenerateNetworkMatrices() - Compressed Sparse Matrix - Object CUIs" )
        PrintLog( concept_output_matrix )
        PrintLog( "GenerateNetworkMatrices() - Original Dense Formatted Sparse Matrix - Object CUIs" )
        PrintLog( concept_output_matrix.todense() )
        PrintLog( "GenerateNetworkMatrices() - =========================================================" )
        PrintLog( "GenerateNetworkMatrices() - =                                                       =" )
        PrintLog( "GenerateNetworkMatrices() - =========================================================" )

    PrintLog( "GenerateNetworkMatrices() - Complete" )

    return concept_input_matrix, predicate_input_matrix, concept_input_matrix

"""
  Creates A Keras Neural Network Using Input/Output Sparse Matrices and Returns The Network

  Parameters
  -----------
  concept_input   : scipy.sparse matrix
    Sparse matrix of all of the one hot vector sets for the input cuis
  predicate_input : scipy.sparse matrix
    Sparse matrix of all of the one hot vector sets for the input predicates
  concept_output  : scipy.sparse matrix
    Sparse matrix of all of the one hot vector sets for the output cuis
  train_network   : boolean
    Train the neural network or return it immediately after it is created

  Returns
  --------
  Model
    The neural network model
"""
def GenerateNeuralNetwork( concept_input, predicate_input, concept_output, train_network ):
    # Check(s)
    if( concept_input is None ):
        PrintLog( "GenerateNeuralNetwork() - Error: Concept Input Contains No Data" )
    if( predicate_input is None ):
        PrintLog( "GenerateNeuralNetwork() - Error: Predicate Input Contains No Data" )
    if( concept_output is None ):
        PrintLog( "GenerateNeuralNetwork() - Error: Concept Output Contains No Data" )
    if( concept_input is None or predicate_input is None or concept_output is None ):
        return None

    # Build The Layers As Designed
    # Build CUI Input Layer / Type Of Model Depends On Sparse or Dense Vector/Matrix Use
    PrintLog( "GenerateNeuralNetwork() - Generating Input And Predicate Layers" )
    
    PrintLog( "GenerateNeuralNetwork() - Generating \"CUI_OneHot_Input\" Layer,       Type: Float32, Shape: (" + str( number_of_cuis ) + ", )" )
    cui_input_layer  = Input( shape = ( number_of_cuis,       ), dtype = 'float32', name = 'CUI_OneHot_Input' )                                        # CUI Input Layer
    
    PrintLog( "GenerateNeuralNetwork() - Generating \"Predicate_OneHot_Input\" Layer, Type: Float32, Shape: (" + str( number_of_predicates ) + ", )" )
    pred_input_layer = Input( shape = ( number_of_predicates, ), dtype = 'float32', name = 'Predicate_OneHot_Input' )                                  # Predicate Input Layer
    concept_layer    = None
    know_in          = None

    # CUI Input Layer To Concept Layer
    # Dense CUI Input
    if( cui_dense_input_mode is True ):
        PrintLog( "GenerateNeuralNetwork() - Generating \"CUI_Dense_Weights\" Embedding Layer, # Of Elements: " + str( number_of_cuis ) + ", Vector Length: " + str( cui_vector_length ) + ", Input Length: " + str( number_of_cuis ) +  ", Trainable: False" )
        cui_word_embedding_layer = Embedding( number_of_cuis, cui_vector_length, weights=[np.asarray(cui_embedding_matrix)], input_length = number_of_cuis, name = "CUI_Dense_Weights", trainable = False )( cui_input_layer )
        
        PrintLog( "GenerateNeuralNetwork() - Appending Flatten Layer: \"CUI_Weight_Dimensionality_Reduction\" To Embedding Layer: \"CUI_Dense_Weights\"" )
        cui_word_embedding_layer = Flatten( name = "CUI_Weight_Dimensionality_Reduction" )( cui_word_embedding_layer )
        
        PrintLog( "GenerateNeuralNetwork() - Generating Concept Layer: \"Concept_Representation\" From Embedding Layer \"CUI_Weight_Dimensionality_Reduction\", Units: " + str( layer_1_size ) + ", Activation: ReLU, Input Dim: " + str( number_of_cuis ) )
        concept_layer            = Dense( units = layer_1_size, activation = 'relu', input_dim = number_of_cuis, name = 'Concept_Representation' )( cui_word_embedding_layer )
        
    # Sparse CUI Input
    else:
        PrintLog( "GenerateNeuralNetwork() - Appending Concept Layer: \"Concept_Representation\" To Input Layer \"CUI_OneHot_Input\", Units: " + str( layer_1_size ) + ", Activation: ReLU, Input Dim: " + str( number_of_cuis ) )
        concept_layer            = Dense( units = layer_1_size, activation = 'relu', input_dim = number_of_cuis, name = 'Concept_Representation' )( cui_input_layer )

    # Predicate Layer
    # Dense Predicate Input
    if( predicate_dense_input_mode is True ):
        PrintLog( "GenerateNeuralNetwork() - Generating \"Predicate_Dense_Weights\" Embedding Layer, # Of Elements: " + str( number_of_predicates ) + ", Vector Length: " + str( predicate_vector_length ) + ", Input Length: " + str( number_of_predicates ) +  ", Trainable: False" )
        pred_word_embedding_layer = Embedding( number_of_predicates, predicate_vector_length, weights=[np.asarray(predicate_embedding_matrix)], input_length = number_of_predicates, name = "Predicate_Dense_Weights", trainable = False )( pred_input_layer )
        
        PrintLog( "GenerateNeuralNetwork() - Appending Flatten Layer: \"Predicate_Weight_Dimensionality_Reduction\" To Embedding Layer: \"Predicate_Dense_Weights\"" )
        pred_word_embedding_layer = Flatten( name = "Predicate_Weight_Dimensionality_Reduction" )( pred_word_embedding_layer )
        
        PrintLog( "GenerateNeuralNetwork() - Generating Concatenate Layer: \"Knowledge_Input_Layer\" From Concatenating Concept Layer: \"Concept_Representation\" And Embedding Layer \"Predicate_Weight_Dimensionality_Reduction\"" )
        know_in                   = concatenate( [concept_layer, pred_word_embedding_layer], name = "Knowledge_Input_Layer" )                                                        # Concatenate The Predicate Layer To The CUI Layer
    
    # Sparse Predicate Input
    else:
        PrintLog( "GenerateNeuralNetwork() - Generating Concatenate Layer: \"Knowledge_Input_Layer\" From Concept Layer: \"Concept_Representation\" And Predicate Input Layer \"Predicate_OneHot_Input\"" )
        know_in = concatenate( [concept_layer, pred_input_layer], name = "Knowledge_Input_Layer" )                                                                 # Concatenate The Predicate Layer To The CUI Layer

    # Build The Rest Of The Network Using The Above Layers
    PrintLog( "GenerateNeuralNetwork() - Generating Know_Layer: \"Knowledge_Representation_Layer\" From Know_In Layer: \"Knowledge_Input_Layer\", Units: " + str( layer_2_size ) + ", Activation: ReLU, Input Dim: " + str( number_of_predicates ) )
    know_layer       = Dense( units = layer_2_size, activation = 'relu', input_dim = number_of_predicates, name = "Knowledge_Representation_Layer" )( know_in )             # Knowledge Representation Layer
    
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
    model.compile( loss = BCE, optimizer = sgd, metrics = ['accuracy', Precision, Recall, Matthews_Correlation] )

    # Print Model Summary
    PrintLog( "GenerateNeuralNetwork() - =========================================================" )
    PrintLog( "GenerateNeuralNetwork() - =                     Model Summary                     =" )
    PrintLog( "GenerateNeuralNetwork() - =========================================================" )
    
    model.summary( print_fn = lambda x:  PrintLog( "GenerateNeuralNetwork() - " + str( x ) ) )      # Capture Model.Summary()'s Print Output As A Function And Store In Variable 'x'
    
    PrintLog( "GenerateNeuralNetwork() - =========================================================" )
    PrintLog( "GenerateNeuralNetwork() - =                                                       =" )
    PrintLog( "GenerateNeuralNetwork() - =========================================================" )

    # Train The Model On The Inputs/Output
    if( train_network ):
        PrintLog( "GenerateNeuralNetwork() - Begin Model Training" )
        PrintLog( "GenerateNeuralNetwork() - Using Batch_Gen To Randomize Input In Batches, Batch Size: " + str( batch_size ) + ", Steps Per Epoch: " + str( steps ) + ", Epochs: " + str( number_of_epochs ) + ", Shuffle: False"  )
        
        model.fit_generator( generator = Batch_Gen( [concept_input, predicate_input], concept_output, batch_size ), steps_per_epoch = steps, epochs = number_of_epochs, shuffle = False )
        
        PrintLog( "GenerateNeuralNetwork() - Finished Model Training" )

    return model

#   Trains The Neural Network Using Specified Concept/Predicate Input/Output Matrix Parameters
def ProcessNeuralNetwork( concept_input, predicate_input, concept_output ):
    # Check(s)
    if( concept_input is None ):
        PrintLog( "ProcessNeuralNetwork() - Error: Concept Input Contains No Data" )
    if( predicate_input is None ):
        PrintLog( "ProcessNeuralNetwork() - Error: Predicate Input Contains No Data" )
    if( concept_output is None ):
        PrintLog( "ProcessNeuralNetwork() - Error: Concept Output Contains No Data" )
    if( concept_input is None or predicate_input is None or concept_output is None ):
        Print( "ProcessNeuralNetwork() - Error: The Script Completed With Error(s)" )
        exit()

    # Create An Untrained Network As A Baseline
    PrintLog( "ProcessNeuralNetwork() - Creating A Baseline Untrained Network" )
    untrained_nn = GenerateNeuralNetwork( concept_input, predicate_input, concept_output, False )

    # Forward Propagate Through The Base Network, Generating Baseline Results
    PrintLog( "ProcessNeuralNetwork() - Generate Baseline Neural Network Metrics" )
    baseline_accuracy = untrained_nn.evaluate_generator( generator = Batch_Gen( [concept_input, predicate_input], concept_output, batch_size = batch_size ), steps = steps )
    PrintLog( "ProcessNeuralNetwork() - Baseline Network [Loss, Accuracy, Precision, Recall]: %s" % baseline_accuracy )

    # Train The Neural Network Using Specified Input/Output Matrices
    PrintLog( "ProcessNeuralNetwork() - Generating Actual Neural Network" )
    start_time = time.time()
    
    PrintLog( "ProcessNeuralNetwork() - Passing Input/Output Matrices To Network" )
    trained_nn = GenerateNeuralNetwork( concept_input, predicate_input, concept_output, True )

    PrintLog( "ProcessNeuralNetwork() - Training Time: %s secs" % ( time.time() - start_time ) )
    PrintLog( "ProcessNeuralNetwork() - Predicting: %s v. %s" % ( test_input_cui, test_output_cui ) )


    # @REMOVE ME - Weight Data Testing
    # weights = trained_nn.get_weights()
    # with open( 'weights.txt', 'w' ) as out_file:
    #     for element in weights:
    #         out_file.write( "%s \n" % element )
    # out_file.close()
    #
    # f = h5py.File( "nn_weights.h5", "r" )
    # for key in f.keys():
    #     data = f.get( key )
    #     print( "Key: " + key )
    #
    #     for g_key in data.keys():
    #         group = data.get( g_key )
    #         print( "G Group Key: " + g_key )
    #
    #         for h_key in group.keys():
    #             print( "Data Key: " + h_key  )
    #             print( "Shape: " + str( group.get( h_key ).shape ) )
    #             print( "Size: " + str( group.get( h_key ).size ) )
    # f.close()

    # Save Network Model
    PrintLog( "ProcessNeuralNetwork() - Saving Model: \"trained_nn_model.h5\"" )
    trained_nn.save( "trained_nn_model.h5" )

    # Save Model Architechture in JSON Format
    PrintLog( "ProcessNeuralNetwork() - Saving Model Architecture: \"model_architecture.json\"" )
    with open( 'model_architecture.json', 'w' ) as out_file:
        out_file.write( trained_nn.to_json() )          # Same as trained_nn.get_config()
    out_file.close()

    # Save Trained Network Weights
    PrintLog( "ProcessNeuralNetwork() - Saving Model Weights: \"trained_nn_model_weights.h5\"" )
    trained_nn.save_weights( "trained_nn_model_weights.h5" )

    # Print Model Depiction To File
    PrintLog( "ProcessNeuralNetwork() - Generating Visual Model: \"model_visual.png\"" )
    plot_model( trained_nn, to_file = 'model_visual.png' )

    # Format The Prediction Data
    if( test_input_cui in unique_cui_data and test_output_cui in unique_predicate_data ):
        predict_cui_df = pd.DataFrame( columns = ( sorted( unique_cui_data ) ) )
        for i in range( number_of_predicates ):
            predict_cui_df.loc[i] = CUI_To_One_Hot( test_input_cui )

        predicate_list  = list( unique_predicate_data.keys() )
        predict_pred_df = pd.DataFrame( columns = predicate_list )
        for i in range( number_of_predicates ):
            predict_pred_df.loc[i] = Predicate_To_One_Hot( predicate_list[i] )

        # Make Predictions
        predictions = trained_nn.predict( x = { "CUI_OneHot_Input": predict_cui_df, "Predicate_OneHot_Input":predict_pred_df  }, verbose = 1 )
        for i in range( len( predictions ) ):
            if( show_prediction is 1 ):
                PrintLog( "%s: %s" % ( test_input_cui + " " + predicate_list[i] + " " + test_output_cui, predictions[i] ) )
                #output_predictions( i, predictions[i] )
            #rankPredictions( i )

#   Removes Unused Data From Memory
def CleanUp():
    global training_data
    global identified_cuis
    global unique_cui_data
    global unidentified_cuis
    global cui_embedding_matrix
    global identified_predicates
    global unique_predicate_data
    global unidentified_predicates
    global predicate_embedding_matrix
    
    PrintLog( "CleanUp() - Removing Data From Memory" )

    unique_cui_data            = None
    training_data              = None
    cui_embedding_matrix       = None
    unique_predicate_data      = None
    predicate_embedding_matrix = None

    identified_cuis            = None
    identified_predicates      = None
    unidentified_cuis          = None
    unidentified_predicates    = None

    gc.collect()
    
    PrintLog( "CleanUp() - Complete" )


############################################################################################
#                                                                                          #
#    Main                                                                                  #
#                                                                                          #
############################################################################################

def main():

    # Check(s)
    if( len( sys.argv ) < 2 ):
        print( "Main() - Error: No Configuration File Argument Specified" )
        exit()

    result = ReadConfigFile( sys.argv[1] )

    # Read CUIs and Predicates From The Same Vector Data File
    if( concept_vector_file is not "" and concept_vector_file == predicate_vector_file ):
        result = LoadVectorFileUsingPredicateList( concept_vector_file, predicate_list_file )

        if( result == -1 ): exit()

    # Read CUIs and Predicates From Differing Vector Data Files
    else:
        cui_file_loaded       = LoadVectorFile( concept_vector_file,   True )
        predicate_file_loaded = LoadVectorFile( predicate_vector_file, False )

        if( cui_file_loaded == -1 or predicate_file_loaded == -1 ):
            PrintLog( "Main() - Error: Unable To Load Vector File(s) / Auto-Generating One-Hot Vectors Using Co-Occurrence Data" )

    # Set Numpy Print Options/Length To "MAXSIZE" ( Used To Debug GenerateNetworkMatrices() Function )    @REMOVEME
    # np.set_printoptions( threshold = sys.maxsize )

    result = GetConceptUniqueIdentifierData()

    if( result == -1 ):
        PrintLog( "Main() - Error: Failed To Auto-Generate Sparse CUI/Predicate Data" )
        exit()

    AdjustVectorIndexData()

    if( result == -1 ):
        PrintLog( "Main() - Error: Failed To Adjust CUI/Predicate Indexing Data" )
        exit()
        
    PrintKeyFiles()

    if( result == -1 ):
        PrintLog( "Main() - Error: Failed To Print CUI/Predicate Data Key Files" )
        exit()

    CUI_OneHot_Input, predicate_input, cui_output = GenerateNetworkMatrices()

    if( CUI_OneHot_Input != None and predicate_input != None and cui_output != None ):
        ProcessNeuralNetwork( CUI_OneHot_Input, predicate_input, cui_output )
    else:
        PrintLog( "Main() - Error: One Or More Input/Output Matrices Failed Generation / One Or More Matrices == None" )
        CUI_OneHot_Input = None
        predicate_input  = None
        cui_output       = None

    # Garbage Collection / Free Unused Memory
    CleanUp()

    CloseDebugFileHandle()

    print( "~Fin" )

if __name__ == "__main__":
    main()
