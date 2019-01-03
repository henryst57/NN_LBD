# Filters the dataset by removing unwanted CUIs and Predicate types
# The user may specify one or both of two options:
#   $genericCuisFile - contains a list of CUIs to remove
#   $predicatesToKeepFile - contains a list of predicates to keep
# See the README for more details
use strict;
use warnings;

#user input, training file
my $genericCuisFile = '../../data/semmedDB/genericCuis';
my $predicatesToKeepFile = '../../data/semmedDB/predicatesToKeep';
my $dataIn = '../../data/semmedDB/allPredicates_1975_2009';
my $dataOut = '../../data/semmedDB/predicates_1975_2009_uniqueCuis_nonNegativePred';
&_filterDataSet($genericCuisFile, $predicatesToKeepFile, $dataIn, $dataOut);

#user input, test file
$dataIn = '../../data/semmedDB/allPredicates_2010_2018';
$dataOut = '../../data/semmedDB/predicates_2010_2018_uniqueCuis_nonNegativePred';
&_filterDataSet($genericCuisFile, $predicatesToKeepFile, $dataIn, $dataOut);


##########################################
#  Begin Code
##########################################
sub _filterDataSet {
    my $genericCuisFile = shift;
    my $predicatesToKeepFile = shift;
    my $dataIn = shift;
    my $dataOut = shift;

    #read in the cuis to discard
    open IN, $genericCuisFile 
	or die ("ERROR: cannot open genericCuisFile: $genericCuisFile\n");
    my %cuisToRemove = ();
    while (my $line = <IN>) {
	#each line is a single CUI that we add to the cuisToRemove hash
	chomp $line;
	$cuisToRemove{$line}=1;
    }
    close IN;

    #read in the predicates to keep
    open IN, $predicatesToKeepFile 
	or die ("ERROR: cannot open predicatesToKeepFile: $predicatesToKeepFile\n");
    my %predicatesToKeep = ();
    while (my $line = <IN>) {
	#each line is a single predicate that we add to the predicatesToKeepFile
	chomp $line;
	$predicatesToKeep{$line}=1;
    }
    close IN;

    #TODO - filter based on subject/object?  or just remove all?
    #read each line of the file and either copy over to the output file
    # or discard the line
    open OUT, ">$dataOut" or die ("ERROR: cannot open dataOut: $dataOut\n");
    open IN, $dataIn or die ("ERROR: cannot open dataIn: $dataIn\n");
    while (my $line = <IN>) {
	#each line contains a CUI\tPREDICATE\tCUI\tPMID
	my @vals = split(/\t/,$line);
	
	#see if you should keep it, only keep lines that contain
	# cuis that we want to keep and a predicate type we want to keep
	if (!exists $cuisToRemove{$vals[0]} 
	    && !exists $cuisToRemove{$vals[2]}
	    && exists $predicatesToKeep{$vals[1]}) {
	    print OUT "$line"
	}
    }
    close OUT;
    close IN;

    #Done, the input file has been filtered and the filtered dataset has been
    # output to the output file
    print "Done!\n";
}

