#!usr/bin/perl
##########################################################################
#                                                                        #
#    NNLBD Test Config Generator And Executor                            #
#                                                                        #
#     Usage: Set Vector File Paths, then "perl testgenerator.pl"         #
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
my $trainNN_path          = "/home/charityml/git_NN_LBD/NN_LBD/NN/trainNN/trainNN.py";       # If TrainNN.py is in a different path than working directory
my $training_file         = "/home/share/NN_LBD/data/testTrain/known_1000";
my $evaluation_file       = "/home/share/NN_LBD/data/testTrain/true_1000";
my $concept_vector_file   = "/home/share/NN_LBD/data/vectors/vectors_random_cuis";
my $predicate_vector_file = "/home/share/NN_LBD/data/vectors/vectors_onehot_predicates";


$SIG{'INT'} = sub { PrintLog( "EXITING PROGRAM" ); exit; }; # catch kill to terminate for good (not just python)

PrintLog( "Generating Configuration File(s)" );

# One-Hot Tests
GenerateConfigurationFile( "LEARNRATE", "LR_0.1",       0.1,            $momentum, $dropout_amt, $batch_size );
GenerateConfigurationFile( "LEARNRATE", "LR_0.3",       0.3,            $momentum, $dropout_amt, $batch_size );
GenerateConfigurationFile( "LEARNRATE", "LR_0.5",       0.5,            $momentum, $dropout_amt, $batch_size );
GenerateConfigurationFile( "LEARNRATE", "LR_0.7",       0.7,            $momentum, $dropout_amt, $batch_size );
GenerateConfigurationFile( "LEARNRATE", "LR_0.9",       0.9,            $momentum, $dropout_amt, $batch_size );
GenerateConfigurationFile( "BATCHSIZE", "BS_1",         $learning_rate, $momentum, $dropout_amt, 1           );
GenerateConfigurationFile( "BATCHSIZE", "BS_50",        $learning_rate, $momentum, $dropout_amt, 50          );
GenerateConfigurationFile( "BATCHSIZE", "BS_100",       $learning_rate, $momentum, $dropout_amt, 100         );
GenerateConfigurationFile( "BATCHSIZE", "BS_200",       $learning_rate, $momentum, $dropout_amt, 200         );
GenerateConfigurationFile( "BATCHSIZE", "BS_500",       $learning_rate, $momentum, $dropout_amt, 500         );
GenerateConfigurationFile( "BATCHSIZE", "BS_999999999", $learning_rate, $momentum, $dropout_amt, 999999999   );
GenerateConfigurationFile( "DROPOUT",   "DAMT_0.0",     $learning_rate, $momentum, 0.0,          $batch_size );
GenerateConfigurationFile( "DROPOUT",   "DAMT_0.1",     $learning_rate, $momentum, 0.1,          $batch_size );
GenerateConfigurationFile( "DROPOUT",   "DAMT_0.3",     $learning_rate, $momentum, 0.3,          $batch_size );
GenerateConfigurationFile( "DROPOUT",   "DAMT_0.5",     $learning_rate, $momentum, 0.5,          $batch_size );
GenerateConfigurationFile( "DROPOUT",   "DAMT_0.7",     $learning_rate, $momentum, 0.7,          $batch_size );
GenerateConfigurationFile( "DROPOUT",   "DAMT_0.9",     $learning_rate, $momentum, 0.9,          $batch_size );
GenerateConfigurationFile( "MOMENTUM",  "M_0.1",        $learning_rate, 0.1,       $dropout_amt, $batch_size );
GenerateConfigurationFile( "MOMENTUM",  "M_0.3",        $learning_rate, 0.3,       $dropout_amt, $batch_size );
GenerateConfigurationFile( "MOMENTUM",  "M_0.5",        $learning_rate, 0.5,       $dropout_amt, $batch_size );
GenerateConfigurationFile( "MOMENTUM",  "M_0.7",        $learning_rate, 0.7,       $dropout_amt, $batch_size );
GenerateConfigurationFile( "MOMENTUM",  "M_0.9",        $learning_rate, 0.9,       $dropout_amt, $batch_size );

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
    my ( $super_folder_name, $label, $learning_rate, $momentum, $dropout_amt, $batch_size ) = @_;
    
    my $folder_name = "$super_folder_name/$label";
    my $file_name   = "$label" . ".cfg";
    
    mkdir( "$super_folder_name" );
    mkdir( "$folder_name" );
    
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