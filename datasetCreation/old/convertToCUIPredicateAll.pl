# converts a file of tab seperated lines containing:
#    CUI PREDICATE CUI anything else
# to a file containing CUI PREDICATE LIST of CUIS
# the output file is a line seperated list of CUI PREDICATE pairs
# followed by a tab seperated list of all CUIs for which that pair holds true
# e.g. C000 ISA C001 C002 C003
#      C005 AUGMENTS C006
use strict;
use warnings;

#User Input, training file
my $inFile = '../../data/semmedDB/predicates_1975_2009_uniqueCuis_nonNegativePred';
my $outFile = '../../data/semmedDB/data_1975_2009_uniqueCuis_nonNegativePred';
&convertToCUIPredicate($inFile, $outFile);

#User Input, test file
$inFile = '../../data/semmedDB/predicates_2010_2018_uniqueCuis_nonNegativePred';
$outFile = '../../data/semmedDB/data_2010_2018_uniqueCuis_nonNegativePred';
&convertToCUIPredicate($inFile, $outFile);



##########################################
#    BEGIN CODE
##########################################
# converts a predicate file to a format for training into a NN
sub convertToCUIPredicate {
    my $inFile = shift;
    my $outFile = shift;

    #open input and output files
    open IN, $inFile or die("Error: cannot open inFile: $inFile\n");
    open OUT, ">$outFile" or die("Error: cannot open outFile: $outFile\n");

    #build a hash of CUI PREDICATES, that has a value of a list of
    # CUIs for that key = CUI\tPREDICATE.
    my %cuiPredPairs = ();
    my @vals = ();
    while (my $line = <IN>) {
	chomp $line;
	@vals = split(/\t/, $line);
	# vals[0] = cui1
	# vals[1] = predicate
	# vals[2] = cui2
	$cuiPredPairs{"$vals[0]\t$vals[1]"} .= "$vals[2]\t";
    }
    close IN;

    #output the hash
    foreach my $key(keys %cuiPredPairs) {

	#get the cui list from the hash
	my $cuis .= $cuiPredPairs{$key};
	#chop the last tab of each value
	chop $cuis;
	#output the line
	print OUT "$key\t$cuis\n";
    }
    close OUT;

    print "done!\n";
}
