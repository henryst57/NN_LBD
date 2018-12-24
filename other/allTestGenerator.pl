#!usr/bin/perl
##########################################################################
#                                                                        #
#    NNLBD Test Config Generator And Executor For All Testing Scenarios  #
#                                                                        #
#     Computes TraiNN Testing With Base Testing Parameters Under The     #
#     Following Scenarios:                                               #
#                  OneHot v OneHot, OneHot v Random, OneHot v W2V        #
#                  Random v OneHot, Random v Random, Random v W2V        #
#                  W2V    v OneHot, W2V    v Random, W2V    v W2V        #
#                                                                        #
#     Format: "CUI Vector File" v "Predicate Vector File"                #
#                                                                        #
#     Usage: Set Vector File Paths, then "perl alltestgenerator.pl"      #
#            It will create folders that categorizes all testing         #
#            scenarios and execute TrainNN on the generated config       #
#            files in each path.                                         #
#                                                                        #
#                              ~ Statler                                 #
#                                                                        #
##########################################################################

use strict;
use warnings;


##########################################################################
#                                                                        #
#    Configuration Constants                                             #
#                                                                        #
##########################################################################
#  ( Used In GenerateConfigurationFile() Function )

# Default Training Parameter Values
my $debug_log             = 0;
my $write_log             = 0;
my $learning_rate         = 0.7;
my $momentum              = 0.9;
my $dropout_amt           = 0.25;
my $batch_size            = 200;
my $layer_1_size          = 200;
my $layer_2_size          = 400;
my $print_key_files       = 1;
my $number_of_epochs      = 100;
my $python_exec_path      = "python";           # python or python3
my $trainNN_path          = "trainNN.py";       # If TrainNN.py is in a different path than working directory
my $training_file         = "/home/share/NN_LBD/data/testTrain/known_10000";
my $evaluation_file       = "/home/share/NN_LBD/data/testTrain/true_10000";

# Vector File Paths (Edit Me)
my $onehot_concept_vector_file   = "/home/share/NN_LBD/data/vectors/vectors_onehot_cuis_new";
my $onehot_predicate_vector_file = "/home/share/NN_LBD/data/vectors/vectors_onehot_predicates_new";
my $random_concept_vector_file   = "/home/share/NN_LBD/data/vectors/vectors_random_cuis";
my $random_predicate_vector_file = "/home/share/NN_LBD/data/vectors/vectors_random_predicates";
my $w2v_concept_vector_file      = "/home/share/NN_LBD/data/vectors/vectors_word2vec_1975_2009_abstractCuis_window8_size200_min-count0_cbow";
my $w2v_predicate_vector_file    = "/home/share/NN_LBD/data/vectors/smdb.allpredicate.s1.mc0.bin_new"; 


$SIG{'INT'} = sub { PrintLog( "EXITING PROGRAM" ); exit; }; # catch kill to terminate for good (not just python)

PrintLog( "Generating Configuration File(s)" );

my $concept_vector_file   = "";                 # Leave Me Blank
my $predicate_vector_file = "";                 # Leave Me Blank

# Generating Comparison Configurations
# Directories Created By The Script Are Named: "CUI Vector File v Predicate Vector File"
my %configurations = ();
$configurations{ "OneHot v OneHot" } = "$onehot_concept_vector_file<>$onehot_predicate_vector_file";
$configurations{ "Random v OneHot" } = "$random_concept_vector_file<>$onehot_predicate_vector_file";
$configurations{ "W2V v OneHot"    } = "$w2v_concept_vector_file<>$onehot_predicate_vector_file";
$configurations{ "OneHot v Random" } = "$onehot_concept_vector_file<>$random_predicate_vector_file";
$configurations{ "Random v Random" } = "$random_concept_vector_file<>$random_predicate_vector_file";
$configurations{ "W2V v Random"    } = "$w2v_concept_vector_file<>$random_predicate_vector_file";
$configurations{ "OneHot v W2V"    } = "$onehot_concept_vector_file<>$w2v_predicate_vector_file";
$configurations{ "Random v W2V"    } = "$random_concept_vector_file<>$w2v_predicate_vector_file";
$configurations{ "W2V v W2V"       } = "$w2v_concept_vector_file<>$w2v_predicate_vector_file";

for my $root_folder_name ( keys %configurations )
{
    my @file_paths = split( "<>", $configurations{ $root_folder_name } );
    $concept_vector_file   = $file_paths[0];
    $predicate_vector_file = $file_paths[1];
    
    if( !defined( $concept_vector_file ) || !defined( $predicate_vector_file ) )
    {
        PrintLog( "$root_folder_name - Concept Vector File Is Empty String"   ) if !defined( $concept_vector_file   );
        PrintLog( "$root_folder_name - Predicate Vector File Is Empty String" ) if !defined( $predicate_vector_file );
        next;
    }
    
    GenerateConfigurationFile( $root_folder_name, "LEARNRATE", "LR_0.1",       0.1,            $momentum, $dropout_amt, $batch_size );
    GenerateConfigurationFile( $root_folder_name, "LEARNRATE", "LR_0.3",       0.3,            $momentum, $dropout_amt, $batch_size );
    GenerateConfigurationFile( $root_folder_name, "LEARNRATE", "LR_0.5",       0.5,            $momentum, $dropout_amt, $batch_size );
    GenerateConfigurationFile( $root_folder_name, "LEARNRATE", "LR_0.7",       0.7,            $momentum, $dropout_amt, $batch_size );
    GenerateConfigurationFile( $root_folder_name, "LEARNRATE", "LR_0.9",       0.9,            $momentum, $dropout_amt, $batch_size );
    GenerateConfigurationFile( $root_folder_name, "BATCHSIZE", "BS_1",         $learning_rate, $momentum, $dropout_amt, 1           );
    GenerateConfigurationFile( $root_folder_name, "BATCHSIZE", "BS_50",        $learning_rate, $momentum, $dropout_amt, 50          );
    GenerateConfigurationFile( $root_folder_name, "BATCHSIZE", "BS_100",       $learning_rate, $momentum, $dropout_amt, 100         );
    GenerateConfigurationFile( $root_folder_name, "BATCHSIZE", "BS_200",       $learning_rate, $momentum, $dropout_amt, 200         );
    GenerateConfigurationFile( $root_folder_name, "BATCHSIZE", "BS_500",       $learning_rate, $momentum, $dropout_amt, 500         );
    GenerateConfigurationFile( $root_folder_name, "BATCHSIZE", "BS_999999999", $learning_rate, $momentum, $dropout_amt, 999999999   );
    GenerateConfigurationFile( $root_folder_name, "DROPOUT",   "DAMT_0.0",     $learning_rate, $momentum, 0.0,          $batch_size );
    GenerateConfigurationFile( $root_folder_name, "DROPOUT",   "DAMT_0.1",     $learning_rate, $momentum, 0.1,          $batch_size );
    GenerateConfigurationFile( $root_folder_name, "DROPOUT",   "DAMT_0.3",     $learning_rate, $momentum, 0.3,          $batch_size );
    GenerateConfigurationFile( $root_folder_name, "DROPOUT",   "DAMT_0.5",     $learning_rate, $momentum, 0.5,          $batch_size );
    GenerateConfigurationFile( $root_folder_name, "DROPOUT",   "DAMT_0.7",     $learning_rate, $momentum, 0.7,          $batch_size );
    GenerateConfigurationFile( $root_folder_name, "DROPOUT",   "DAMT_0.9",     $learning_rate, $momentum, 0.9,          $batch_size );
    GenerateConfigurationFile( $root_folder_name, "MOMENTUM",  "M_0.1",        $learning_rate, 0.1,       $dropout_amt, $batch_size );
    GenerateConfigurationFile( $root_folder_name, "MOMENTUM",  "M_0.3",        $learning_rate, 0.3,       $dropout_amt, $batch_size );
    GenerateConfigurationFile( $root_folder_name, "MOMENTUM",  "M_0.5",        $learning_rate, 0.5,       $dropout_amt, $batch_size );
    GenerateConfigurationFile( $root_folder_name, "MOMENTUM",  "M_0.7",        $learning_rate, 0.7,       $dropout_amt, $batch_size );
    GenerateConfigurationFile( $root_folder_name, "MOMENTUM",  "M_0.9",        $learning_rate, 0.9,       $dropout_amt, $batch_size );
}

PrintLog( "~Fin", 1 );


##########################################################################
#                                                                        #
#    Support Functions                                                   #
#                                                                        #
##########################################################################

sub PrintLog
{
    my $str         = shift;
    my $force_print = shift;
    print( "$str\n" ) if( $debug_log == 1 || ( defined( $force_print ) && $force_print == 1 ) );
}

sub OpenFile
{
    my $file_path = shift;
    
    # Check(s)
    PrintLog( "OpenFile() - Error: File Path Is Empty String" ) if( $file_path eq "" );
    
    my $file_handle;
    open $file_handle, ">:", "$file_path" or die ( "ERROR: unable to open outFile: $file_path\n" );
    
    # Check(s)
    PrintLog( "OpenFile() - Error: File Handle Not Defined / Failed Creating File Handle" ) if !defined( $file_handle );
    PrintLog( "OpenFile() - Creating File: $file_path" );
    
    return $file_handle;
}

sub CloseFile
{
    my $file_handle = shift;
    
    # Check(s)
    if( !defined( $file_handle ) )
    {
        PrintLog( "CloseFile() - Error: File Handle Is Not Defined" );
    }
    else
    {
        PrintLog( "CloseFile() - Closing File Handle" );
        close( $file_handle );
    }
}

sub GenerateConfigurationFile
{
    # Configuration Variables
    my ( $root_folder_name, $super_folder_name, $label, $learning_rate, $momentum, $dropout_amt, $batch_size ) = @_;
    
    my $folder_name = "$root_folder_name/$super_folder_name/$label";
    my $file_name   = "$label" . ".cfg";
    
    mkdir( "$root_folder_name"  );
    mkdir( "$root_folder_name/$super_folder_name" );
    mkdir( "$folder_name"       );
    
    my $result      = 0;
    my $file_handle = OpenFile( "$folder_name/$file_name" );
    
    # Remove ".cfg" From File Name Variable
    $file_name =~ s/\.cfg//g;
    
    # Check(s)
    if( !defined( $file_handle ) || $result == -1 )
    {
        PrintLog( "GenerateConfigurationFile() - Error: Configuration File Could Not Be Created" );
        PrintLog( "GenerateConfigurationFile() -        BS: " . $batch_size . " - LR: " . $learning_rate . " - M: " . $momentum . " - D: " . $dropout_amt );
        rmdir( "$folder_name" );
        return -1;
    }
    
    # Include Configuration Variables
    print $file_handle ( "<Layer1Size>:"   . $layer_1_size  . "\n" );
    print $file_handle ( "<Layer2Size>:"   . $layer_2_size  . "\n" );
    print $file_handle ( "<DebugLog>:"     . $debug_log     . "\n" );
    print $file_handle ( "<WriteLog>:"     . $write_log     . "\n" );
    print $file_handle ( "<BatchSize>:"    . $batch_size    . "\n" );
    print $file_handle ( "<LearningRate>:" . $learning_rate . "\n" );
    print $file_handle ( "<Momentum>:"     . $momentum      . "\n" );
    print $file_handle ( "<DropoutAMT>:"   . $dropout_amt   . "\n" );
    
    # Include Configuration Constants
    print $file_handle ( "<NumberOfEpochs>:"      . $number_of_epochs         . "\n"                    );
    print $file_handle ( "<TrainFile>:"           . $training_file            . "\n"                    );
    print $file_handle ( "<EvaluateFile>:"        . $evaluation_file          . "\n"                    );
    print $file_handle ( "<PrintKeyFiles>:"       . $print_key_files          . "\n"                    );
    print $file_handle ( "<ConceptVectorFile>:"   . $concept_vector_file      . "\n"                    );
    print $file_handle ( "<PredicateVectorFile>:" . $predicate_vector_file    . "\n"                    );
    print $file_handle ( "<TrainingStatsFile>:"   . "$folder_name/$file_name" . "_training_stats.txt\n" );
    print $file_handle ( "<TestingStatsFile>:"    . "$folder_name/$file_name" . "_testing_stats.txt\n"  );
    print $file_handle ( "<OutputFileName>:"      . "$folder_name/$file_name" . "\n"                    );
    
    CloseFile( $file_handle );
    
    ExecuteTraining( "$folder_name/$file_name.cfg" );
    
    return 0;
}

sub ExecuteTraining
{
    my $file_name = shift;
    
    # Check(s)
    if( !defined( $file_name ) )
    {
        PrintLog( "ExecuteTraining() - Error: File Name Is Not Defined" );
    }
    elsif( $file_name eq "" )
    {
        PrintLog( "ExecuteTraining() - Error: File Name Is Empty String / DNE" )
    }
    elsif( !( -e $file_name ) )
    {
        PrintLog( "ExecuteTraining() - Error: File Name - " . $file_name . " - Does Not Exist" )
    }
    elsif( -z $file_name )
    {
        PrintLog( "ExecuteTraining() - Error: File Name - " . $file_name . " - Contains No Data" )
    }
    else
    {
        PrintLog( "ExecutingTraining() - Executing NNLBD using Config File: $file_name" );
        return system( "$python_exec_path \"$trainNN_path\" \"$file_name\"" );
    }
    
    PrintLog( "ExecuteTraining() - Error Detected: See Above For Further Information" );
    return -1;
}
