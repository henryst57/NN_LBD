############################################################################
#                                                                          #
#    NNLBD - Neural Network Literature Based Discovery v0.35               #
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

    NNLBD's TrainNN script has python dependencies that must be installed prior
    to running.

    ~ Prerequisites and Libraries ~
      - Python 3.4.3 or higher -> https://www.python.org/downloads/
      - Numpy and Scipy        -> https://scipy.org/install.html
      - Tensorflow 1.3.0       -> https://www.tensorflow.org/install/install_linux
      - Keras 2.0.9            -> https://keras.io/#installation
      - Pydot                  -> Install via "pip install pydot" or "conda install pydot"
         * Windows Installations Will Require Installing "GraphViz" Program *

    This script is heavily dependent on having a training file of the form:
        
        C001	TREATS	C002	C004	C009
        C001	ISA	C003
        C002	TREATS	C005	C006	C010
        C002	AFFECTS	C003	C004	C007 	C008
        C003	TREATS	C002
        C003	AFFECTS	C001	C005 	C007
        C003	ISA	C008
        C004	ISA	C010
        C005	TREATS	C003	C004	C006
        C005	AFFECTS	C001	C009	C010
        ```
        (tab separated and line-by-line)
        
        ^ This represents the "cui_mini" test/debug dataset mentioned in the segments below.

############################################################################
#                                                                          #
#    Running The Script                                                    #
#                                                                          #
############################################################################

    Running the script also depends on a configuration file of parameters.
    This is a plain text file with parameters listed line-by-line along
    with their associated values.
    
    An example configuration file contents:
        <DebugLog>:1
        <TrainFile>:data\cui_mini
    
    This demonstrates a configuration file of bare-minimum required parameters.
    *** Note: All parameters not specified will use default parameters. ***
    
    Here is a typical configuration file content example, let's call it "config.cfg":
    
        <DebugLog>:1
        <WriteLog>:0
        <Layer1Size>:200
        <Layer2Size>:400
        <NumberOfEpochs>:50
        <LearningRate>:0.025
        <BatchSize>:10
        <Momentum>:0.9
        <DropoutAMT>:0.25
        <PrintKeyFiles>:1
        <TrainFile>:data\cui_mini
        <TestInputCUI>:C001
        <TestInputPredicate>:ISA
        <TestOutputCUI>:C003
        <ConceptVectorFile>:samplevectors\testSparseCUIVectors.bin
        <PredicateVectorFile>:samplevectors\testSparsePredicateVectors.bin
        <TrainingStatsFile>:training_stats_file.txt
    
    To execute training with the parameters in this file use the command:
        "python trainNN.py config.cfg" or "python3 trainNN.py config.cfg"
    
    This instructs the TrainNN script to execute training based on these parameters.
    Also prints debug statements to the console (DebugLog=1). Prints the CUI and Predicate
    Key Files (PrintKeyFiles=1) and report metrics after each epoch to a file named
    "training_stats_file.txt"; Using the network parameters such Layer1Size, Layer2Size,
    NumberOfEpochs, LearningRate, BatchSize, Momentum and DropoutAMT values over the
    Training File "cui_mini" given the Concept and Predicate vector files. After
    training has completed, the script will use the parameters "TestInputCUI,
    TestInputPredicate and TestOutputCUI" for validation which will print metrics to
    the console/file and network prediction. This will also print data to the screen/file.
    
    Clarification Note: With the example above, the script will iterate over the training file,
                        fetching 10 lines at a time (batch size), transform the data into input/
                        output for the network and train. It will continue to fetch the next 10
                        lines and train until the end of the file. This marks one epoch. It will
                        continue doing this over 5 epochs, which is the specified number of epochs
                        in the configuration example above.
    
    The network also supports sparse, association and dense (word2vec) vectors as inputs for CUIs
    and Predicates. It will detect and build the network accordingly depending on what type of vectors
    the user specifies. It also handles dense/sparse/association vectors independently between CUI
    and Predicate inputs.
    
    *** To Use TrainNN in CNDA mode, leave concept and predicate vectors as empty strings.
        TrainNN will build sparse representation automatically given the training file and
        train the network accordingly, similar to CNDA v0.8 with MCC. 

############################################################################
#                                                                          #
#    Output                                                                #
#                                                                          #
############################################################################

    The script will output messages to the console during training. These messages
    include debug statements if the parameter is set and metrics after each training
    epoch. These reported metrics include: Loss, Accuracy, Precision, Recall and
    Matthews Correlation. Example console output as shown below:
    
        10/10 [==============================] - 0s 27ms/step - loss: 0.6861 - acc: 0.5700 - Precision: 0.2917 - Recall: 0.6087 - Matthews_Correlation: 0.1408
    
    After training has completed, it will save the following files:
    
        model_visual.png            <- A Visual Depiction Of The Neural Network Model
        model_architecture.json     <- The Keras Model Architecture In JSON Format
        trained_nn_model.h5         <- The Keras Trained Neural Network Model (Includes Weights)
        trained_nn_model_weights.h5 <- The Keras Trained Neural Network Model Weights
    
    These can be utilized to re-construct the neural network and do as you please with it in python.
    
    *** See Parameters Below For More Options and A Description Of Their Purpose Below ***
        

############################################################################
#                                                                          #
#    Configuration File Parameters                                         #
#                                                                          #
############################################################################

    <DebugLog>              -> Prints Debug Statements To The Console                                                                      [ Type Int: 1 = True / 0 = False (Default) ]
    <WriteLog>              -> Writes Debug Statements To A File                                                                           [ Type Int: 1 = True / 0 = False (Default) ]
    <Momentum>              -> Momentum Value                                                                                              [ Type Float - Default: 0.9                ]
    <BatchSize>             -> Number Of Inputs In A Batch Per Epoch                                                                       [ Type Int   - Default: 10                 ]
    <DropoutAMT>            -> Dropout AMT Value                                                                                           [ Type Float - Default: 0.25               ]
    <Layer1Size>            -> Concept Layer Size                                                                                          [ Type Int   - Default: 200                ]
    <Layer2Size>            -> Knowledge Representation Layer Size                                                                         [ Type Int   - Default: 400                ]
    <Learning Rate>         -> Learning Rate / % Of Error                                                                                  [ Type Float - Default: 0.1                ]
    <TestInputCUI>          -> CUI Input To Test In Trained Network                                                                        [ Type String                              ]
    <TestInputPredicate>    -> Predicate Input To Test In Trained Network                                                                  [ Type String                              ]
    <TestOutputCUI>         -> CUI Output To Test In Trained Network                                                                       [ Type String                              ]
    <NumberOfSteps>         -> Number Of Steps Value                                                                                       [ Type Int                                 ] (Not Used)
    <NumberOfEpochs>        -> Number Of Times To Iterate Over Training Data During Network Learning                                       [ Type Int   - Default: 5                  ]
    <TrainFile>             -> File To Train The Network                                                                                   [ Type String                              ]
    <NegativeSampleRate>    -> Negative Sample Rate                                                                                        [ Type Int: Default: 5                     ] (Not Used/Reserved For Future)
    <PrintKeyFile>          -> Prints Key Files As Used By The Network                                                                     [ Type Int: 1 = True / 0 = False (Default) ]
    <ConceptVectorFile>     -> Subject CUI Vector File Used To Train The Network                                                           [ Type String                              ]
    <PredicateVectorFile>   -> Predicate Vector File Used To Train The Network                                                             [ Type String                              ]
    <PredicateListFile>     -> Predicate List File                                                                                         [ Type String                              ]
    <TrainingStatsFile>     -> File To Print Training Statistics To After Each Epoch                                                       [ Type String                              ]
    <PrintNetworkInputs>    -> Prints Raw Network Input/Output Data (Matrices) Prior To Sending To The Network As Input/Expected Output    [ Type Int: 1 = True / 0 = False (Default) ]
    <PrintMatrixStats>      -> Prints Statistics On Network Input/Output Matrix/Sequence Generation                                        [ Type Int: 1 = True / 0 = False (Default) ]
    <AdjustVectors>         -> Optimizes CUI/Predicate Vectors, Removing Vector Data Not In Training File And Re-organizes Vector Indices  [ Type Int: 1 = True / 0 = False (Default) ]
    <TrainableDenseWeights> -> Sets CUI/Predicate Dense Weights As Trainable                                                               [ Type Int: 1 = True / 0 = False (Default) ]
    <ShuffleInputData>      -> Shuffles Input/Output Data As Entered Into The Network During Training                                      [ Type Int: 1 = True / 0 = False (Default) ]

    Note: Leaving ConceptVectorFile and PredicateVectorFile as empty string (blank) will enable
          TrainNN to operate in CNDA mode which generates sparse vectors given the training file,
          then trains the network accordingly and similarly to CNDA v0.8 with MCC.

############################################################################
#                                                                          #
#    Debug Configuration File Parameters (Detailed)                        #
#                                                                          #
############################################################################

    <TestInputCUI>          -> CUI Input To Test In Trained Network
    <TestInputPredicate>    -> Predicate Input To Test In Trained Network 
    <TestOutputCUI>         -> CUI Output To Test In Trained Network
    
    Using these three parameters, the script will attempt to forward propagate the inputs versus
    the expected network output and evaluate the given results, printing network metrics to the screen.
    The script will also print the network's predictions to the screen as shown below:
    
    Example Console Output:
        ProcessNeuralNetwork() - Evaluating Model
        ProcessNeuralNetwork() - Evaluating Inputs - CUI: "C001" and Predicate: "ISA" Versus Output CUI: "C003"
        ProcessNeuralNetwork() - Loss: 0.4185728132724762 - Accuracy: 0.8999999761581421 - Precision: 0.0 - Matthews Correlation: 0.0
        ProcessNeuralNetwork() - Predicting Output Given Inputs -> CUI: "C001" and Predicate: "ISA"
        ProcessNeuralNetwork() - Predicted CUI Indices: [[0.29122147 0.3247243  0.40073794 0.38012606 0.25465256 0.24141827
          0.27007547 0.32670346 0.30688864 0.33571053]]
    
    
    <TrainingStatsFile>     -> File To Print Training Statistics To After Each Epoch
    
    This parameter can be specified as any string/filename. It will print calculated neural network
    metrics after each epoch. An example can be seen below:
    
    Example File Output:
        epoch	Matthews_Correlation	Precision	Recall	acc	loss
        0	0.10790298134088516	0.27450981736183167	0.6086956262588501	0.5400000214576721	0.6931508779525757
        1	0.09702833741903305	0.26923078298568726	0.6086956262588501	0.5300000309944153	0.6928690671920776
        2	0.03856460377573967	0.24528302252292633	0.5652173757553101	0.5	0.6959676742553711
        3	...
        n   0.2028554230928421	0.37037035822868347	0.43478259444236755	0.699999988079071	0.6662446856498718
    
    
    <PrintNetworkInputs>    -> Prints Raw Network Input/Output Data (Matrices) Prior To Sending To The Network As Input/Expected Output
    
    This parameters will show the matrices/sequences as generated by the script before sending as input/output to the
    neural network. Example console output is shown below:
    
    Example Console Output:
        GenerateNetworkData() - =========================================================
        GenerateNetworkData() - =        Printing Compressed Row/Sparse Matrices        =
        GenerateNetworkData() - =========================================================
          (0, 0)	1.0
          (1, 0)	1.0
          (2, 1)	1.0
          (3, 1)	1.0
          (4, 2)	1.0
          (5, 2)	1.0
          (6, 2)	1.0
          (7, 3)	1.0
          (8, 4)	1.0
          (9, 4)	1.0
        GenerateNetworkData() - Original Dense Formatted Sparse Matrix - Subject CUIs
        [[1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
         [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
         [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
         [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
         [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
         [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
         [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
         [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
         [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
         [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]]
        GenerateNetworkData() - Compressed Sparse Matrix - Predicates
          (0, 1)	1.0
          (1, 0)	1.0
          (2, 2)	1.0
          (3, 0)	1.0
          (4, 2)	1.0
          (5, 1)	1.0
          (6, 0)	1.0
          (7, 1)	1.0
          (8, 2)	1.0
          (9, 0)	1.0
        GenerateNetworkData() - Original Dense Formatted Sparse Matrix - Predicates
        [[0. 1. 0.]
         [1. 0. 0.]
         [0. 0. 1.]
         [1. 0. 0.]
         [0. 0. 1.]
         [0. 1. 0.]
         [1. 0. 0.]
         [0. 1. 0.]
         [0. 0. 1.]
         [1. 0. 0.]]
        GenerateNetworkMatrices() - Compressed Sparse Matrix - Object CUIs
          (0, 2)	1
          (1, 1)	1
          (1, 3)	1
          (1, 8)	1
          (2, 2)	1
          (2, 3)	1
          (2, 6)	1
          (2, 7)	1
          (3, 4)	1
          (3, 5)	1
          (3, 9)	1
          (4, 0)	1
          (4, 4)	1
          (4, 6)	1
          (5, 7)	1
          (6, 1)	1
          (7, 9)	1
          (8, 0)	1
          (8, 8)	1
          (8, 9)	1
          (9, 2)	1
          (9, 3)	1
          (9, 5)	1
        GenerateNetworkMatrices() - Original Dense Formatted Sparse Matrix - Object CUIs
        [[0 0 1 0 0 0 0 0 0 0]
         [0 1 0 1 0 0 0 0 1 0]
         [0 0 1 1 0 0 1 1 0 0]
         [0 0 0 0 1 1 0 0 0 1]
         [1 0 0 0 1 0 1 0 0 0]
         [0 0 0 0 0 0 0 1 0 0]
         [0 1 0 0 0 0 0 0 0 0]
         [0 0 0 0 0 0 0 0 0 1]
         [1 0 0 0 0 0 0 0 1 1]
         [0 0 1 1 0 1 0 0 0 0]]
        GenerateNetworkData() - =========================================================
        GenerateNetworkData() - =                                                       =
        GenerateNetworkData() - =========================================================
    
    
    <PrintMatrixStats>      -> Prints Statistics On Network Input/Output Matrix/Sequence Generation
    
    This will print statistics on the matrices/sequence arrays used as input/output for the neural network.
    
    Example Console Output:
        GenerateNetworkData() - =========================================================
        GenerateNetworkData() - =                Matrix Generation Stats                =
        GenerateNetworkData() - =========================================================
        GenerateNetworkData() -   Number Of Subject CUI Inputs              : 10
        GenerateNetworkData() -   Number Of Predicate Inputs                : 10
        GenerateNetworkData() -   Number Of Object CUI Outputs              : 23
        GenerateNetworkData() -   Number Of Concept Input Index Elements    : 10
        GenerateNetworkData() -   Number Of Concept Input Value Elements    : 10
        GenerateNetworkData() -   Number Of Predicate Input Index Elements  : 10
        GenerateNetworkData() -   Number Of Predicate Input Value Elements  : 10
        GenerateNetworkData() -   Number Of Concept Output Index Elements   : 23
        GenerateNetworkData() -   Number Of Concept Output Value Elements   : 23
        GenerateNetworkData() -   Number Of Identified CUI Elements         : 10
        GenerateNetworkData() -   Number Of Identified Predicate Elements   : 3
        GenerateNetworkData() -   Number Of Unidentified CUI Elements       : 0
        GenerateNetworkData() -   Number Of Unidentified Predicate Elements : 0
        GenerateNetworkData() -   Total Unique CUIs                         : 10
        GenerateNetworkData() -   Total Unique Predicates                   : 3
        GenerateNetworkData() -   Identified Input Data Array Length        : 500
        GenerateNetworkData() -   Number Of Skipped Lines In Training Data  : 0
        GenerateNetworkData() -   Total Input Data Array Length             : 10
        GenerateNetworkData() - =========================================================
        GenerateNetworkData() - =                                                       =
        GenerateNetworkData() - =========================================================

    Note: These parameters are used for debugging purposes. While they're amusing to look at during
          runtime, computational time complexity will suffer as a result of using these parameters.
        
    Cliffnotes: Enabling these functions will result in a loss in performance but have no effect on
                training the neural network.


############################################################################
#                                                                          #
#    Test Script Findings:                                                 #
#                                                                          #
############################################################################

    Using "cui_mini" dataset with dense vectors (CUI and predicates). The
    network is able to successfully predict correct object CUIs given their
    respective subject CUI and predicate. (These dense vectors can either be
    randomized vectors or word2vec vectors. Both achieve good results.)
    
    The "cui_mini" dataset has a line that reads:
    
        C001 TREATS C002 C004 C009
    
    The network is trained on this along with the other lines as input versus
    expected output. This was trained using these settings in the configuration
    file:
        
        <DebugLog>:1
        <WriteLog>:0
        <Layer1Size>:200
        <Layer2Size>:400
        <NumberOfEpochs>:50
        <LearningRate>:0.025
        <BatchSize>:10
        <Momentum>:0.9
        <DropoutAMT>:0.25
        <TrainFile>:data\cui_mini
        <TestInputCUI>:C001
        <TestInputPredicate>:TREATS
        <TestOutputCUI>:C002
        <ConceptVectorFile>:samplevectors\testDenseCUIVectors.bin
        <PredicateVectorFile>:samplevectors\testDensePredicateVectors.bin
    
    After using dense representations of CUIs and Predicates (word2vec or randomized vectors)
    as input for the network, the network was given input CUI: "C001" and input
    predicate "TREATS". Here is sample output post training:
    
        ProcessNeuralNetwork() - Evaluating Model
        ProcessNeuralNetwork() - Evaluating Inputs - CUI: "C001" and Predicate: "TREATS" Versus Output CUI: "C002"
        ProcessNeuralNetwork() - Loss: 0.34693118929862976 - Accuracy: 0.800000011920929 - Precision: 0.3333333432674408 - Matthews Correlation: 0.5091750621795654
        ProcessNeuralNetwork() - Predicting Output Given Inputs -> CUI: "C001" and Predicate: "TREATS"
        ProcessNeuralNetwork() - Predicted CUI Indices: [[0.03838203 0.8059531  0.2969185  0.7409016  0.06411877 0.12303055
          0.02417022 0.01284351 0.6929042  0.09160434]]
          
    Notice in the line "Predicated CUI Indices" the indices 1, 3 and 8 have the highest values.
    These correspond to the network predicting "C002 C004 and C009" given the inputs "C001 and TREATS",
    as demonstrated in the "cui_mini" network input/output training data.
    

############################################################################
#                                                                          #
#    Authors                                                               #
#                                                                          #
############################################################################

  Clint Cuffy       - Virginia Commonwealth University (cuffyca@vcu.edu)
  Megan Charity     - Virginia Commonwealth University (charityml@vcu.edu)
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