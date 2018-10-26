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
from keras import optimizers


########################
#                      #
#   GLOBAL VARIABLES   #
#                      #
########################

model                      = null
model_file                 = 'trained_nn_model.h5'      #this may be set by the config later 

# config stuff
learning_rate              = 0.1
momentum                   = 0.9
steps                      = 1
batch_size                 = 10

concept_input_file         = ""
concept_vector_file        = ""
predicate_vector_file      = ""
predicate_list_file        = ""
eval_file                  = ""

# Debug Log Variables
debug_log                  = False
write_log                  = False
debug_file_name            = "nnlb_log.txt"
debug_log_file             = None

#   Reads The Specified Configuration File Parameters Into Memory
#   and Sets The Appropriate Variable Data
def ReadConfigFile( config_file_path ):
    global debug_log
    global write_log
    global learning_rate
    global momentum
    global batch_size
    global steps
    global concept_input_file
    global concept_vector_file
    global predicate_vector_file
    global predicate_list_file
    global eval_file

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
        if data[0] == "BatchSize"           : batch_size            = int( data[1] )       
        if data[0] == "NumberOfSteps"       : steps                 = int( data[1] )
        if data[0] == "Momentum"            : momentum              = float( data[1] )
        if data[0] == "LearningRate"        : learning_rate         = float( data[1] )
        if data[0] == "ConceptInputFile"    : concept_input_file    = str( data[1] )
        if data[0] == "ConceptVectorFile"   : concept_vector_file   = str( data[1] )
        if data[0] == "PredicateVectorFile" : predicate_vector_file = str( data[1] )
        if data[0] == "PredicateListFile"   : predicate_list_file   = str( data[1] )
        if data[0] == "EvaluateFile"        : eval_file             = str( data[1] )
        
    f.close()

    OpenDebugFileHandle()
    
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
    PrintLog( "    Evaluation File        : " + str( eval_file ) )
    PrintLog( "    Batch Size             : " + str( batch_size ) )
    PrintLog( "    Number Of Steps        : " + str( steps ) )
    PrintLog( "    Learning Rate          : " + str( learning_rate ) )
    PrintLog( "    Momentum               : " + str( momentum ) )
    PrintLog( "    Number Of CUIs         : " + str( number_of_cuis ) )
    PrintLog( "    Number Of Predicates   : " + str( number_of_predicates ) )
    
    PrintLog( "=========================================================" )
    PrintLog( "-                                                       -" )
    PrintLog( "=========================================================\n" )

    PrintLog( "ReadConfigFile() - Complete" )

    return 0


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


###########################       NEURAL NETWORK FUNCTIONS        #######################



#loads the model
def LoadModel():
    model = load_model(eval_file)
    printLog("Loaded the model from %s" % eval_file)

#evaluates the data on the loaded model
def Evaluate(concept_input, predicate_input, concept_output, metricSet):

     # Check(s)
    if( concept_input is None ):
        PrintLog( "GenerateNeuralNetwork() - Error: Concept Input Contains No Data" )
    if( predicate_input is None ):
        PrintLog( "GenerateNeuralNetwork() - Error: Predicate Input Contains No Data" )
    if( concept_output is None ):
        PrintLog( "GenerateNeuralNetwork() - Error: Concept Output Contains No Data" )
    if( concept_input is None or predicate_input is None or concept_output is None ):
        return None

    #get the model together
    sgd = optimizers.SGD( lr = learning_rate, momentum = momentum )
    model.compile(loss = BCE, optimizer = sgd, metrics = metricSet)
    eval_metrics = model.evaluate_generator(generator = Batch_Gen([concept_input, predicate_input], concept_output, batch_size ), steps = steps, verbose=1)




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

# Read CUIs and Predicates From the Key Files
GetConceptUniqueIdentifierData()
PrintKeyFiles()

cui_input, predicate_input, cui_output = GenerateNetworkMatrices()

if( cui_input != None and predicate_input != None and cui_output != None ):
    Evaluate( cui_input, predicate_input, cui_output, ['accuracy', Precision, Recall, Matthews_Correlation])

# Garbage Collection / Free Unused Memory
CleanUp()

CloseDebugFileHandle()

print( "~Fin Eval" )
