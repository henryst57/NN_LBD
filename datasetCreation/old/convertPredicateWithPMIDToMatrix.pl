#Removes the predication type from the input file and outputs to the output
# file. The input file is formatted as:
#    cui1\tpredType\tcui2\tpmid
# The output file is formatted as:
#    cui1\tcui2\tpredCount
# NOTE: outputs the count of unique articles the pair occurs (if it occurs multiple times in the same PMID that will not increment the count)

use strict;
use warnings;


#user input
my $inputPredicates = 'allPredicates_0_1999';
my $outputPredicates = 'allPredicates_0_1999_matrix';

######################################
#         Begin Code
######################################

#read in as a matrix
open IN, $inputPredicates or die ("ERROR: unable to open inputPredicatesL $inputPredicates\n");
my %matrix = ();
while (my $line = <IN>) {

    #line is cui1\tpredType\tcui2\tpmid
    my @vals = split(/\t/, $line);
    my $cui1 = $vals[0];
    my $cui2 = $vals[2];

    if (!defined $cui1 || !defined $cui2) {
	die ("ERROR: not all values defined: $cui1, $cui2\n");
    }
    if (!defined $matrix{$cui1}) {
	my %newHash = ();
	$matrix{$cui1} = \%newHash;
    }
    
    ${$matrix{$cui1}}{$cui2}++
}
close IN;


#output the matrix
open OUT, ">$outputPredicates" or die ("ERROR: unable to open outputPredicates: $outputPredicates\n");
foreach my $cui1 (keys %matrix) {
    foreach my $cui2 (keys %{$matrix{$cui1}}) {
	print OUT "$cui1\t$cui2\t${$matrix{$cui1}}{$cui2}\n"
    }
}
close OUT;

print "Done!\n";

