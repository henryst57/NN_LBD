############################################################################################
#                                                                                          #
#    Neural Network - Literature Based Discovery                                           #
#    -------------------------------------------                                           #
#                                                                                          #
#    Date:    10/02/2018                                                                   #
#    Revised: 10/17/2018                                                                   #
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
from fractions import gcd


############################################################################################
#                                                                                          #
#    Global Variables / Neural Network Parameters (Default Values)                         #
#                                                                                          #
############################################################################################
version                    = 0.2
number_of_predicates       = 0
number_of_cuis             = 0
layer_1_size               = 200
layer_2_size               = 400
learning_rate              = 0.1
number_of_epochs           = 5
steps                      = 1
batch_size                 = 10
momentum                   = 0.9
dropout_amt                = 0.25
negative_sample_rate       = 5
print_key_files            = 0
show_prediction            = 0
concept_input_file         = ""
concept_output_file        = ""
test_input_cui             = ""
test_output_cui            = ""
concept_vector_file        = ""
predicate_vector_file      = ""
predicate_list_file        = ""
print_input_matrices       = 0
cui_dense_input_mode       = False
predicate_dense_input_mode = False
cui_occurence_data_length  = 0
cui_vector_length          = 0
predicate_vector_length    = 0

cui_occurence_data         = []
unique_cui_data            = {}
unique_predicate_data      = {}
cui_embedding_matrix       = []
predicate_embedding_matrix = []

# Debug Log Variables
debug_log                  = False
write_log                  = False
debug_file_name            = "nnlb_log.txt"
debug_log_file             = None


############################################################################################
#                                                                                          #
#    Sub-Routines                                                                          #
#                                                                                          #
############################################################################################

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
    # Split Based On "<>" Characters, Only Present In Parse Format
    number_of_elements = vector.split( '<>' )
    if ( len( number_of_elements ) == 1 ): return True
    if ( len( number_of_elements ) >  1 ): return False

#   Reads The Specified Configuration File Parameters Into Memory
#   and Sets The Appropriate Variable Data
def ReadConfigFile( config_file_path ):
    global debug_log
    global write_log
    global layer_1_size
    global layer_2_size
    global learning_rate
    global number_of_epochs
    global steps
    global batch_size
    global momentum
    global dropout_amt
    global negative_sample_rate
    global print_key_files
    global show_prediction
    global concept_input_file
    global concept_output_file
    global debug_log_file
    global test_input_cui
    global test_output_cui
    global concept_vector_file
    global predicate_vector_file
    global predicate_list_file
    global print_input_matrices

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

        if data[0] == "DebugLog"            : debug_log             = int( data[1] )
        if data[0] == "WriteLog"            : write_log             = int( data[1] )
        if data[0] == "Momentum"            : momentum              = float( data[1] )
        if data[0] == "BatchSize"           : batch_size            = int( data[1] )
        if data[0] == "DropoutAMT"          : dropout_amt           = float( data[1] )
        if data[0] == "Layer1Size"          : layer_1_size          = int( data[1] )
        if data[0] == "Layer2Size"          : layer_2_size          = int( data[1] )
        if data[0] == "LearningRate"        : learning_rate         = float( data[1] )
        if data[0] == "TestInputCUI"        : test_input_cui        = str( data[1] )
        if data[0] == "TestOutputCUI"       : test_output_cui       = str( data[1] )
        if data[0] == "NumberOfSteps"       : steps                 = int( data[1] )
        if data[0] == "NumberOfEpochs"      : number_of_epochs      = int( data[1] )
        if data[0] == "ConceptInputFile"    : concept_input_file    = str( data[1] )
        if data[0] == "NegativeSampleRate"  : negative_sample_rate  = int( data[1] )
        if data[0] == "PrintKeyFiles"       : print_key_files       = int( data[1] )
        if data[0] == "ShowPrediction"      : show_prediction       = int( data[1] )
        if data[0] == "ConceptVectorFile"   : concept_vector_file   = str( data[1] )
        if data[0] == "PredicateVectorFile" : predicate_vector_file = str( data[1] )
        if data[0] == "PredicateListFile"   : predicate_list_file   = str( data[1] )
        if data[0] == "PrintInputMatrices"  : print_input_matrices  = int( data[1] )

    f.close()

    OpenDebugFileHandle()

    # Check(s)
    if( concept_input_file is "" ):
        PrintLog( "ReadConfigFile() - Error: \"ConceptInputFile\" Not Specified" )
        exit()
    if( batch_size is 0 ):
        PrintLog( "ReadConfigFile() - Error: Batch_Size Variable Cannot Be <= \"0\" / Exiting Program" )
        exit()
    if( concept_vector_file == predicate_vector_file and ( predicate_list_file is "" or predicate_list_file is None ) ):
        PrintLog( "ReadConfigFile() - Error: When \"ConceptVectorFile\" == \"PredicateVectorFile\"" )
        PrintLog( "ReadConfigFile() -        A Valid Predicate List Must Be Specified" )
        exit()

    PrintLog( "=========================================================" )
    PrintLog( "~      Neural Network - Literature Based Discovery      ~" )
    PrintLog( "~          Version " + str( version ) + " (Based on CaNaDA v0.8)           ~" )
    PrintLog( "=========================================================\n" )

    PrintLog( "  Built on Tensorflow Version: 1.8.0" )
    PrintLog( "  Built on Keras Version: 2.1.6" )
    PrintLog( "  Installed TensorFlow Version: " + str( tf.__version__ ) )
    PrintLog( "  Installed Keras Version: "      + str( keras.__version__ )  + "\n" )

    # Print Settings To Console
    PrintLog( "=========================================================" )
    PrintLog( "-   Configuration File Settings                         -" )
    PrintLog( "=========================================================" )

    PrintLog( "    Concept Data Input File: " + str( concept_input_file ) )
    PrintLog( "    Concept Vector File    : " + str( concept_vector_file ) )
    PrintLog( "    Predicate Vector File  : " + str( predicate_vector_file ) )
    PrintLog( "    Predicate List File    : " + str( predicate_list_file ) )
    PrintLog( "    Batch Size             : " + str( batch_size ) )
    PrintLog( "    Learning Rate          : " + str( learning_rate ) )
    PrintLog( "    Number Of Epochs       : " + str( number_of_epochs ) )
    PrintLog( "    Number Of Steps        : " + str( steps ) )
    PrintLog( "    Momentum               : " + str( momentum ) )
    PrintLog( "    Dropout AMT            : " + str( dropout_amt ) )
    PrintLog( "    Layer 1 Size           : " + str( layer_1_size ) )
    PrintLog( "    Layer 2 Size           : " + str( layer_2_size ) )
    PrintLog( "    Number Of CUIs         : " + str( number_of_cuis ) )
    PrintLog( "    Number Of Predicates   : " + str( number_of_predicates ) )
    PrintLog( "    Negative Sample Rate   : " + str( negative_sample_rate ) )
    PrintLog( "    Print Key Files        : " + str( print_key_files ) )
    PrintLog( "    Print Input Matrices   : " + str( print_input_matrices ) )
    PrintLog( "    Test Input CUI         : " + str( test_input_cui ) )
    PrintLog( "    Test Output CUI        : " + str( test_output_cui ) )

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
    global unique_predicate_date
    global predicate_vector_length
    global predicate_dense_input_mode
    global predicate_embedding_matrix

    # Check(s)
    if( vector_file_path is None or vector_file_path is "" ):
        PrintLog( "LoadVectorFile() - Warning: No Vector File Specified" )
        return -1

    if CheckIfFileExists( vector_file_path ) is False:
        PrintLog( "LoadVectorFile() - Error: Specified File \"" + str( vector_file_path ) + "\" Does Not Exist" )
        return -1

    PrintLog( "LoadVectorFile() - Loading Vector File: \"" + vector_file_path +   "\"" )
    vector_data      = None

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

    # Choose The Second Element In The Vector Data And Check Vector Format (Predicate)
    if( len( vector_data ) > 1 and is_cui_vectors is False ):
        predicate_dense_input_mode = IsDenseVectorFormat( vector_data[2].strip() )
        if( predicate_dense_input_mode == True ):  PrintLog( "LoadVectorFile() - Detected Dense Predicate Vector Format" )
        if( predicate_dense_input_mode == False ): PrintLog( "LoadVectorFile() - Detected Sparse Predicate Vector Format" )

    loaded_elements  = 0

    # Read Dense Vector Formatted File
    if( ( is_cui_vectors == True and cui_dense_input_mode == True ) or ( is_cui_vectors == False and predicate_dense_input_mode == True ) ):
        unique_index     = 0

        for vector in vector_data:
            vector = vector.strip()
            vector = vector.split()

            # Parse Header Information
            if( len( vector ) == 2 ):
                number_of_vectors = int( vector[0] )
                if( is_cui_vectors is True  ): cui_vector_length       = int( vector[1] )
                if( is_cui_vectors is False ): predicate_vector_length = int( vector[1] )
            # Parse CUI/Predicate Data
            else:
                label = vector[0]
                data  = vector[1:]

                # Add Unique CUI Index And Data
                if( is_cui_vectors is True and label not in unique_cui_data ):
                    unique_cui_data[ label ] = unique_index
                    cui_embedding_matrix.append( data )
                    loaded_elements += 1

                # Add Unique Predicate Index And Data
                if( is_cui_vectors is False and label not in unique_predicate_data ):
                    unique_predicate_data[ label ] = unique_index
                    predicate_embedding_matrix.append( data )
                    loaded_elements += 1

                unique_index += 1

        if( is_cui_vectors is True  ): number_of_cuis       = len( unique_cui_data )
        if( is_cui_vectors is False ): number_of_predicates = len( unique_predicate_data )

    # Read Sparse Formatted Vector File
    else:
        for vector in vector_data:
            vector = vector.strip()
            vector = vector.split( "<>" )
            label  = vector[0]
            data   = vector[1:]

            for data_element in data:
                index_data = data_element.split( ',' )

                if( is_cui_vectors == True and label not in unique_cui_data ):
                    unique_cui_data[ label ] = int( index_data[0] )
                if( is_cui_vectors == False and label not in unique_predicate_data ):
                    unique_predicate_data[ label ] = int( index_data[0] )

                loaded_elements += 1

        if( is_cui_vectors is True  ):
            number_of_cuis          = loaded_elements
            cui_vector_length       = loaded_elements
        if( is_cui_vectors is False ):
            number_of_predicates    = loaded_elements
            predicate_vector_length = loaded_elements

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
    global unique_predicate_date
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

    PrintLog( "LoadVectorFileUsingPredicateList() - Loading Vector File: \"" + predicate_list_file +   "\"" )
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

    # Read Dense Vector Formatted File
    if( cui_dense_input_mode == True or predicate_dense_input_mode == True ):
        for vector in vector_data:
            vector = vector.strip()
            vector = vector.split()

            # Parse Header Information
            if( len( vector ) == 2 ):
                number_of_vectors       = int( vector[0] )
                cui_vector_length       = int( vector[1] )
                predicate_vector_length = int( vector[1] )
            # Parse CUI/Predicate Data
            else:
                label = vector[0]
                data  = vector[1:]

                # Add Unique CUI Index And Data
                if( label not in predicate_list and label not in unique_cui_data ):
                    unique_cui_data[ label ] = loaded_cui_elements
                    cui_embedding_matrix.append( data )
                    loaded_cui_elements += 1

                # Add Unique Predicate Index And Data
                if( label in predicate_list and label not in unique_predicate_data ):
                    unique_predicate_data[ label ] = loaded_predicate_elements
                    predicate_embedding_matrix.append( data )
                    loaded_predicate_elements += 1

    # Read Sparse Formatted Vector File
    else:
        first_predicate_index = -1

        for vector in vector_data:
            vector = vector.strip()
            vector = vector.split( "<>" )
            label  = vector[0]
            data   = vector[1:]

            for data_element in data:
                index_data = data_element.split( ',' )

                if( label not in predicate_list and label not in unique_cui_data ):
                    unique_cui_data[ label ] = int( index_data[0] )
                    loaded_cui_elements += 1
                if( label in predicate_list and label not in unique_predicate_data ):
                    if( first_predicate_index == -1 ): first_predicate_index = int( index_data[0] )
                    unique_predicate_data[ label ] = int( index_data[0] ) - int( first_predicate_index )
                    loaded_predicate_elements += 1

        cui_vector_length       = loaded_cui_elements
        predicate_vector_length = loaded_predicate_elements

    number_of_cuis       = len( unique_cui_data )
    number_of_predicates = len( unique_predicate_data )

    PrintLog( "LoadVectorFileUsingPredicateList() - Loaded " + str( loaded_cui_elements ) + " CUI Elements" )
    PrintLog( "LoadVectorFileUsingPredicateList() - Loaded " + str( loaded_predicate_elements ) + " Predicate Elements" )
    PrintLog( "LoadVectorFileUsingPredicateList() - Complete" )
    return 0

#   Fetches Concept Unique Identifier Data From The File
#   Adds Unique CUIs and Predicates / Relation Types To Hashes
#   Along With Unique Numeric Index Identification Values
#   Also Sets The Number Of Steps If Not Specified
def GetConceptUniqueIdentifierData():
    global steps
    global number_of_cuis
    global unique_cui_data
    global cui_occurence_data
    global unique_predicate_data
    global cui_occurence_data_length

    if CheckIfFileExists( concept_input_file ) == False:
        PrintLog( "GetConceptUniqueIdentifierData() - Error: CUI Data File \"" + str( concept_input_file ) + "\" Does Not Exist" )
        return -1;

    # Read Concept Unique Identifier Data From File
    PrintLog( "GetConceptUniqueIdentifierData() - Reading Concept Input File: \"" + str( concept_input_file ) + "\"" )
    try:
        with open( concept_input_file, "r" ) as in_file:
            cui_occurence_data = in_file.readlines()
    except FileNotFoundError:
        PrintLog( "GetConceptUniqueIdentifierData() - Error: Unable To Open File \"" + str( concept_input_file )+ "\"" )
        return -1
    finally:
        in_file.close()

    PrintLog( "GetConceptUniqueIdentifierData() - File Data In Memory" )

    cui_occurence_data = [ line.strip() for line in cui_occurence_data ]    # Removes Trailing Space Characters From CUI Data Strings
    cui_occurence_data.sort()

    cui_occurence_data_length = len( cui_occurence_data )

    # Sets The Number Of Steps If Not Specified
    if( steps == 0 ):
        PrintLog( "GetConceptUniqueIdentifierData() - Warning: Number Of Steps Not Specified / Generating Value Based On Data Size" )
        steps = int( cui_occurence_data_length / batch_size ) + ( 0 if( cui_occurence_data_length % batch_size == 0 ) else 1 )
        PrintLog( "GetConceptUniqueIdentifierData() - Number Of Steps: " + str( steps ) )

    ###################################################
    #                                                 #
    #   Generate The Unique CUI and Predicate Lists   #
    #                                                 #
    ###################################################
    if( len( unique_cui_data ) > 0 and len( unique_predicate_data ) > 0 ):
        PrintLog( "GetConceptUniqueIdentifierData() - Unique CUI/Predicate Data Previously Generated" )
        number_of_cuis       = len( unique_cui_data )
        number_of_predicates = len( unique_predicate_data )
        PrintLog( "GetConceptUniqueIdentifierData() - Complete" )
        return 0

    PrintLog( "GetConceptUniqueIdentifierData() - Generating Unique CUI and Predicate Data Lists" )

    cui_pattern     = re.compile( "C[0-9]+" )

    for line in cui_occurence_data:
        elements = re.split( r"\s+", line )

        # Only Add New Unique Concept Unique Identifier (CUI)
        cui_elements = filter( cui_pattern.match, elements )

        for element in cui_elements:
            if element not in unique_cui_data: unique_cui_data[ element ] = 1

        # Only Add New Unique Predicates To The List
        #   Ex line: C001 ISA C002   <- elements[1] = Predicate / Relation Type (String)
        if elements[1] not in unique_predicate_data: unique_predicate_data[ elements[1] ] = 1    # Add Predicate / Relation Type To Unique List

    number_of_cuis       = len( unique_cui_data )
    number_of_predicates = len( unique_predicate_data )

    PrintLog( "GetConceptUniqueIdentifierData() - Lists Generated" )
    PrintLog( "GetConceptUniqueIdentifierData() - Unique Number Of CUIs: " + str( number_of_cuis ) )
    PrintLog( "GetConceptUniqueIdentifierData() - Unique Number Of Predicates: " + str( number_of_predicates ) )

    # Sort CUIs/Predicates/Relation Types In Ascending Order And Assign Appropriate Identification Values
    PrintLog( "GetConceptUniqueIdentifierData() - Sorting Lists In Ascending Order" )

    index = 0
    for cui in sorted( unique_cui_data.keys() ):
        unique_cui_data[ cui ] = index
        index += 1

    index = 0
    for predicate in sorted( unique_predicate_data.keys() ):
        unique_predicate_data[ predicate ] = index
        index += 1

    PrintLog( "GetConceptUniqueIdentifierData() - Complete" )
    return 0

#   Print CUI and Predicate Key Files
def PrintKeyFiles():
    if( print_key_files is 1 ):
        PrintLog( "PrintKeyFiles() - Key File Printing Enabled" )
        PrintLog( "PrintKeyFiles() - Printing CUI Key File: " + concept_input_file + ".cui_key" )

        cui_keys       = sorted( unique_cui_data.keys() )
        predicate_keys = sorted( unique_predicate_data.keys() )

        try:
            with open( concept_input_file + ".cui_key", "w" ) as out_file:
                for cui in cui_keys:
                    out_file.write( str( unique_cui_data[ cui ] ) + " " + str( cui ) + "\n" )
        except IOError:
            PrintLog( "PrintKeyFiles() - Error: Unable To Create CUI Key File" )
            return -1
        finally:
            out_file.close()

        PrintLog( "PrintKeyFiles() - File Read" )
        PrintLog( "PrintKeyFiles() - Printing CUI Key File: " + concept_input_file + ".predicate_key" )

        try:
            with open( concept_input_file + ".predicate_key", "w" ) as out_file:
                for predicate in predicate_keys:
                    out_file.write( str( unique_predicate_data[ predicate ] ) + " " + str( predicate ) + "\n" )
        except IOError:
            PrintLog( "PrintKeyFiles() - Error: Unable To Create Predicate Key File" )
            return -1
        finally:
            out_file.close()

        PrintLog( "PrintKeyFiles() - File Read" )

    PrintLog( "PrintKeyFiles() - Complete" )
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

#   Parses Through CUI Data And Generates Sparse Matrices For
#   Concept Input, Predicate Input and Concept Output Data
def GenerateNetworkMatrices():
    global print_input_matrices

    # Check(s)
    if( cui_occurence_data_length == 0 ):
        PrintLog( "GenerateNetworkMatrices() - Error: No CUI Data In Memory / Was An Input CUI File Read Before Calling Method?" )
        return None, None, None

    PrintLog( "GenerateNetworkMatrices() - Generating Network Matrices" )

    concept_input_indices   = []
    predicate_input_indices = []
    concept_output_indices  = []
    concept_output_values   = []
    output_cui_count        = 0

    # Parses each line of raw data input and adds them to arrays for matrix generation
    PrintLog( "GenerateNetworkMatrices() - Parsing CUI Data / Generating Network Input-Output Data Arrays" )
    for i in range( cui_occurence_data_length ):
        not_found_flag = False
        line = cui_occurence_data[i]
        line_elements = re.split( r"\s+", line )

        # Check(s)
        # If Subject CUI, Predicate and Object CUIs Are Not Found Within The Specified Unique CUI and Predicate Lists, Report Error and Skip The Line
        if( line_elements[0] not in unique_cui_data ):
            PrintLog( "GenerateNetworkMatrices() - Error: Subject CUI \"" + str( line_elements[0] ) + "\" Is Not In Unique CUI Data List" )
            not_found_flag = True
        if( line_elements[1] not in unique_predicate_data ):
            PrintLog( "GenerateNetworkMatrices() - Error: Predicate \"" + str( line_elements[1] ) + "\" Is Not In Unique Predicate Data List" )
            not_found_flag = True
        for element in line_elements[2:]:
            if( element not in unique_cui_data ):
                PrintLog( "GenerateNetworkMatrices() - Error: Object CUI \"" + str( element ) + "\" Is Not In Unique CUI Data List" )
                not_found_flag = True

        if( not_found_flag is True ): continue

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
        PrintLog( "GenerateNetworkMatrices() - Error: Concept Input Indices List Contains No Data / Specified Subject CUIs In The Input Not File Not Within Unique CUI Data?" )
    if( len( predicate_input_indices ) is 0 ):
        PrintLog( "GenerateNetworkMatrices() - Error: Predicate Input Indices List Contains No Data / Specified Predicates In The Input Not File Not Within Unique CUI Data?" )
        PrintLog( "GenerateNetworkMatrices() -        Note: This May Be Reporting Due To Concept Input Indices List Erroring Out" )
    if( len( concept_output_indices ) is 0 ):
        PrintLog( "GenerateNetworkMatrices() - Error: Concept Output Indices List Contains No Data / Specified Object CUIs In The Input Not File Not Within Unique CUI Data?" )
        PrintLog( "GenerateNetworkMatrices() -        Note: This May Be Reporting Due To Concept Input or Predicate Indices List Erroring Out" )
    if( len( concept_input_indices ) is 0 or len( predicate_input_indices ) is 0 or len( concept_output_indices ) is 0 ): return None, None, None

    # Transpose the arrays, then convert to rows/columns
    PrintLog( "GenerateNetworkMatrices() - Transposing Index Data Arrays Into Row/Column Data" )
    concept_input_row,   concept_input_column   = zip( *concept_input_indices   )
    predicate_input_row, predicate_input_column = zip( *predicate_input_indices )
    concept_output_row,  concept_output_column  = zip( *concept_output_indices  )

    # Convert row/column data into sparse matrices
    PrintLog( "GenerateNetworkMatrices() - Converting Index Data Into Matrices" )
    concept_input_matrix   = sparse.csr_matrix( ( [1]*cui_occurence_data_length,    ( concept_input_row,   concept_input_column ) ),   shape = ( cui_occurence_data_length, number_of_cuis ) )
    predicate_input_matrix = sparse.csr_matrix( ( [1]*cui_occurence_data_length,    ( predicate_input_row, predicate_input_column ) ), shape = ( cui_occurence_data_length, number_of_predicates ) )
    concept_output_matrix  = sparse.csr_matrix( ( [1]*output_cui_count, ( concept_output_row,  concept_output_column ) ),  shape = ( cui_occurence_data_length, number_of_cuis ) )

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
    cui_input_layer  = Input( shape = ( number_of_cuis, ), dtype = 'float32', name = 'cui_input' )                                              # CUI Input Layer
    pred_input_layer = Input( shape = ( number_of_predicates, ), dtype = 'float32', name = 'pred_input' )                                       # Predicate Input Layer
    concept_layer    = None
    know_in          = None

    # CUI Input Layer To Concept Layer
    # Dense CUI Input
    if( cui_dense_input_mode is True ):
        cui_word_embedding_layer = Embedding( number_of_cuis, cui_vector_length, weights=[np.asarray(cui_embedding_matrix)], input_length = number_of_cuis, trainable = False )( cui_input_layer )
        cui_word_embedding_layer = Flatten()( cui_word_embedding_layer )
        concept_layer            = Dense( units = layer_1_size, activation = 'relu', input_dim = number_of_cuis, name = 'concept_rep' )( cui_word_embedding_layer )
    # Sparse CUI Input
    else:
        concept_layer    = Dense( units = layer_1_size, activation = 'relu', input_dim = number_of_cuis, name = 'concept_rep' )( cui_input_layer )

    # Predicate Layer
    # Dense Predicate Input
    if( predicate_dense_input_mode is True ):
        pred_word_embedding_layer = Embedding( number_of_predicates, predicate_vector_length, weights=[np.asarray(predicate_embedding_matrix)], input_length = number_of_predicates, trainable = False )( pred_input_layer )
        pred_word_embedding_layer = Flatten()( pred_word_embedding_layer )
        know_in          = concatenate( [concept_layer, pred_word_embedding_layer] )                                                                # Concatenate The Predicate Layer To The CUI Layer
    # Sparse Predicate Input
    else:
        know_in          = concatenate( [concept_layer, pred_input_layer] )                                                                         # Concatenate The Predicate Layer To The CUI Layer

    # Build The Rest Of The Network Using The Above Layers
    know_layer       = Dense( units = layer_2_size, activation = 'relu', input_dim = number_of_predicates )( know_in )                              # Knowledge Representation Layer
    dropper          = Dropout( dropout_amt )( know_layer )                                                                                         # Define The Dropout
    cui_output_layer = Dense( units = number_of_cuis, activation = 'sigmoid', name = 'cui_output' )( dropper )                                      # Define The Output

    # Compile The Network Model Using Input and Output Layers
    model = Model( inputs = [cui_input_layer, pred_input_layer], outputs = [cui_output_layer] )

    # Create The Optimizers And Metrics For The Output
    sgd = optimizers.SGD( lr = learning_rate, momentum = momentum )
    model.compile( loss = BCE, optimizer = sgd, metrics = ['accuracy', Precision, Recall, Matthews_Correlation] )

    # Print Model Summary
    PrintLog( "\nGenerateNeuralNetwork() - Model Summary" )
    model.summary( print_fn = lambda x:  PrintLog( "GenerateNeuralNetwork() - " + str( x ) ) )      # Capture Model.Summary()'s Print Output As A Function And Store In Variable 'x'
    PrintLog( "GenerateNeuralNetwork() - End Model Summary\n" )

    # Train The Model On The Inputs/Output
    if( train_network ):
        model.fit_generator( generator = Batch_Gen( [concept_input, predicate_input], concept_output, batch_size ), steps_per_epoch = steps, epochs = number_of_epochs, shuffle = False )

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
    untrained_nn = GenerateNeuralNetwork( concept_input, predicate_input, concept_output, False )

    # Forward Propagate Through The Base Network, Generating Baseline Results
    baseline_accuracy = untrained_nn.evaluate_generator( generator = Batch_Gen( [concept_input, predicate_input], concept_output, batch_size = batch_size ), steps = steps )
    PrintLog( "ProcessNeuralNetwork() - Baseline Network [Loss, Accuracy, Precision, Recall]: %s" % baseline_accuracy )

    # Train The Neural Network Using Specified Input/Output Matrices
    start_time = time.time()
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
    predict_cui_df = pd.DataFrame( columns = ( sorted( unique_cui_data ) ) )
    for i in range( number_of_predicates ):
        predict_cui_df.loc[i] = CUI_To_One_Hot( test_input_cui )

    predicate_list  = list( unique_predicate_data.keys() )
    predict_pred_df = pd.DataFrame( columns = predicate_list )
    for i in range( number_of_predicates ):
        predict_pred_df.loc[i] = Predicate_To_One_Hot( predicate_list[i] )

    # Make Predictions
    predictions = trained_nn.predict( x = { "cui_input": predict_cui_df, "pred_input":predict_pred_df  }, verbose = 1 )
    for i in range( len( predictions ) ):
        if( show_prediction is 1 ):
            PrintLog( "%s: %s" % ( test_input_cui + " " + predicate_list[i] + " " + test_output_cui, predictions[i] ) )
            #output_predictions( i, predictions[i] )
        #rankPredictions( i )

#   Removes Unused Data From Memory
def CleanUp():
    global unique_cui_data
    global cui_occurence_data
    global cui_embedding_matrix
    global unique_predicate_data
    global predicate_embedding_matrix

    unique_cui_data            = None
    cui_occurence_data         = None
    cui_embedding_matrix       = None
    unique_predicate_data      = None
    predicate_embedding_matrix = None

    gc.collect()
    

############################################################################################
#                                                                                          #
#    Main                                                                                  #
#                                                                                          #
############################################################################################

# Check(s)
if( len( sys.argv ) < 2 ):
    print( "Main() - Error: No Configuration File Argument Specified" )
    exit()

result = ReadConfigFile( sys.argv[1] )

# Read CUIs and Predicates From The Same Vector Data File
if( concept_vector_file == predicate_vector_file ):
    LoadVectorFileUsingPredicateList( concept_vector_file, predicate_list_file )
# Read CUIs and Predicates From Differing Vector Data Files
else:
    LoadVectorFile( concept_vector_file, True )
    LoadVectorFile( predicate_vector_file, False )

GetConceptUniqueIdentifierData()
PrintKeyFiles()

cui_input, predicate_input, cui_output = GenerateNetworkMatrices()

if( cui_input != None and predicate_input != None and cui_output != None ):
    ProcessNeuralNetwork( cui_input, predicate_input, cui_output )

# Garbage Collection / Free Unused Memory
CleanUp()

CloseDebugFileHandle()

print( "~Fin" )
