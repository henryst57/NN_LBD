#convertAssociationMatrixToVectors.pl
#  converts an association matrix file of the format:
#     rowCui<>cui,score<>cui,score<>...<>cui,score
#  to a sparse vector file of the format:
#     rowCui<>index,score<>...<>index,score
use strict;
use warnings;

#input association matrix file
my $inFile = '../../data/vectors/assocMatrix_x2_1975_2015_window8_ordered_threshold1_clean';

#output vector file
my $outFile = '../../data/vectors/assocVectors_x2_1975_2015_window8_ordered_threshold1_clean';

&makeVectorFile($inFile, $outFile);



#converts an association matrix file of the format:
#   rowCui<>cui,score<>cui,score<>...<>cui,score
#to a sparse vector file of the format:
#   rowCui<>index,score<>...<>index,score
#It does this by assigning an index to each CUI and 
# outputting 
sub makeVectorFile {
    #grab input
    my $inFile = shift;
    my $outFile = shift;

    #open the input file and output file
    open IN, "$inFile" or die ("Error: unable to open inFile: $inFile\n");
    open OUT, ">$outFile" or die ("Error: unable to open outFile: $outFile\n");
    
    #read the vocabulary of the matrix and assign indeces
    # to each term
    my %vocab = ();
    while (my $line = <IN>) {
	my @vals = split(/<>/,$line);
	#line is rowCui<>cui,score<>...<>cui,score
	
	#grab each CUI and add to the vocab
	my $rowCui = shift @vals;
	if (!defined $vocab{$rowCui}) {
	    $vocab{$rowCui} = scalar keys %vocab;
	}
	foreach my $pair(@vals) {
	    my @pairVals = split(/,/,$pair); 
	    #pair is cui,score
	    if (!defined $vocab{$pairVals[0]}) {
		$vocab{$pairVals[0]} = scalar keys %vocab;
	    } 
	}
    }
    close IN;
    #close in, then re-open it, cause it makes it start reading 
    # from the top again...there is a seek operation I think 
    # too, but this is just easier

    #output the matrix as a sparse vector file with indeces
    open IN, "$inFile" or die ("Error: unable to open inFile: $inFile\n");
    while (my $line = <IN>) {
	chomp $line;
	my @vals = split(/<>/,$line);
	#line is rowCui<>cui,score<>...<>cui,score
	
	#grab each CUI and add to the vocab
	my $rowCui = shift @vals;
	print OUT "$rowCui";
	foreach my $pair(@vals) {
	    my @pairVals = split(/,/,$pair); 
	    #pair is cui,score
	    print OUT "<>$vocab{$pairVals[0]},$pairVals[1]";
	}
	print OUT "\n";
    }
    close IN;
    close OUT;

    print "Done!\n";
}
