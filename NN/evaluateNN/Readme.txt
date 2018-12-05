############################################################################
#                                                                          #
#    NNLBD - Neural Network Literature Based Discovery v0.36               #
#                                                                          #
############################################################################

  ~ Based on CNDA (CUI-Neural-net-Discovery-Approximator) v0.8
    Pronounced 'Canada' - A neural network that takes SemRep
    relationships to predict new CUI relationships. 


############################################################################
#                                                                          #
#    Getting Started                                                       #
#                                                                          #
############################################################################

    NNLBD's EvaluateNN script has python dependencies that must be installed prior
    to running.

    ~ Prerequisites and Libraries ~
      - Python 3.4.3 or higher -> https://www.python.org/downloads/
      - Numpy and Scipy        -> https://scipy.org/install.html
      - Tensorflow 1.3.0       -> https://www.tensorflow.org/install/install_linux
      - Keras 2.0.9            -> https://keras.io/#installation
      - Pydot                  -> Install via "pip install pydot" or "conda install pydot"
         * Windows Installations Will Require Installing "GraphViz" Program *

    This script uses functions defined in trainNN.py. In order for the script to access these functions, the 
    environment variable $NN_PATH must be set to the absolute path of the modules (NN directory.) 
    To automatically set this environment variable, run the "setNNPathVar.sh" script found in the same directory.
    This will set up the environment variable and export it for global use on the user's system.
   
    In order to run this script, trainNN.py must be run and a model file with weights must be saved
    This must be a ".h5" file type to be used with the built in libraries
    These files are automatically generated if trainNN.py is run prior
        Example Model File:         trained_nn_model.h5
        Example Model Weights File: trained_nn_model_weights.h5
    
    Key files containing the indices of the unique predicates and cuis used in the training data must also be exported
    The extension of the cui key file must be ".cui_key" and the extension of the predicate key file must be ".predicate_key"
    These files are generated if trainNN.py is run prior with the parameters "<PrintKeyFiles>:1" in the config file
        Example CUI Key File:       known_sample.cui_key
        Example Predicate Key File: known_sample.predicate_key
   
    The evaluation file must be in the same format as the training input file used for trainNN.py
        <See trainNN/Readme.txt for an example format>
        
        

############################################################################
#                                                                          #
#    Running The Script                                                    #
#                                                                          #
############################################################################

    The configuration file used by evaluateNN.py is similar to the configuration file needed for trainNN.py
    However, additional parameters are needed by the program to run the evaluation
          * <EvaluateFile>:evaluationFile
          * <ModelFile>:trained_nn_model.h5
          * <ModelWeights>:train_nn_model_weights_epoch4.h5
            <CUIKeyFile>sample.cui_key
            <PredKeyFile>:sample.predicate_key
            <EvalOut>:evaluation_output

          * - required to run the program
    <See "Configuration File Parameters" for more detailed explanation>
    For the purpose of this README documentation, we will use "config2.cfg" as the update 
    configuation file needed for evaluation

    To execute evaluation with the parameters in this file use the command:
        "python evaluateNN.py config2.cfg" or "python3 evaluateNN.py config2.cfg"
    
    The same parameters set for training in trainNN.py will be included and 
    used for evaluation in evaluateNN.py (i.e.  NumberOfEpochs, LearningRate, BatchSize, Momentum, etc.)
    
    The evaluation file generates matrices similar to trainNN.py. It takes in 
    the set in the same format as the training set used on the model. However, instead of re-creating 
    the unique CUI and predicate list sets, it uses the original set (extracted from the key files)
    to create new matrices based on the training set's CUIs and predicates.

    The program will output evaluation metrics based on the function defined in trainNN.py
        Accuracy                - a built in Keras function
        Precision()             - TP / (TP+FP) ; re-defined in trainNN.py
        Recall()                - TP / (TP+FN) ; re-defined in trainNN.py
        Matthews_Correlation()  - balance of TP and FP; re-defined in trainNN.py
    

############################################################################
#                                                                          #
#    Output                                                                #
#                                                                          #
############################################################################

    After running the evaluation matrices on the model, the script will output a single 
    evaluation file depicting the values associated to each metric specified in the 
    following format:

        METRIC_TYPE1 : VALUE1
        METRIC_TYPE2 : VALUE2
        METRIC_TYPE3 : VALUE3
        ...
         
############################################################################
#                                                                          #
#    Configuration File Parameters                                         #
#                                                                          #
############################################################################

    <ModelFile>             -> the file with the model architechture
    <ModelWeights>          -> the file with the model weights
    <EvaluateFile>          -> the file to evaluate the network 
    <CUIKeyFile>            -> the unique cui key file (defaults to <TRAINFILE>.cui_key if not specified)
    <PredKeyFile>           -> the unique predicate key file (defaults to <TRAINFILE>.predicate_key if not specified)
    <EvalOut>               -> the file to output the evaluation results to (defaults to evaluation.out if not specified)

############################################################################
#                                                                          #
#    Authors                                                               #
#                                                                          #
############################################################################

  Megan Charity     - Virginia Commonwealth University (charityml@vcu.edu)
  Clint Cuffy       - Virginia Commonwealth University (cuffyca@vcu.edu)
  Sam Henry         - Virginia Commonwealth University (henryst@vcu.edu)
  Bridget T McInnes - Virginia Commonwealth University (btmcinnes@vcu.edu)


############################################################################
#                                                                          #
#    License                                                               #
#                                                                          #
############################################################################

    Copyright (C) 2018 Clint Cuffy, Megan Charity, Sam Henry & Bridget T. McInnes

    Permission is granted to copy, distribute and/or modify this document
    under the terms of the GNU Free Documentation License, Version 1.2 or
    any later version published by the Free Software Foundation; with no
    Invariant Sections, no Front-Cover Texts, and no Back-Cover Texts.

    Note: a copy of the GNU Free Documentation License is available on the
    web at:

    <http://www.gnu.org/copyleft/fdl.html>

    and is included in this distribution as FDL.txt.


############################################################################
#                                                                          #
#    Acknowledgments                                                       #
#                                                                          #
############################################################################

- Network as proposed by Rumelhart, et al. 1993 adapted for use with SemRep
