#!/bin/perl
use strict;
use warnings;
use lib '/home/share/NN_LBD_NEW/code';
require utilities;	#import the utilities.pm module

#TODO - read in semmedDB and create one hot vectors for each relation type

#Input: semmedDB matrix of known relations (known file)
#  format is:
#
#Output: dense (or sparse) vector file of format
#  denseFormat:
#
# sparseFormat:
#
#

sub semmedOut{
	my $inputFile = $ARGV[0];
	my $outputFile = "vectors_onehot_predicates";

	print "Taking INPUT:\n";
	print "\t$inputFile\n";
	print "Into one hot OUTPUT:\n";
	print "\t$outputFile\n\n";

	makeSomeSemMedHotties($inputFile, $outputFile);
	print "DONE!";
}


#Megan, please complete vector I/O in the utilities.pm
# taken care of -M
sub testUtil{

	print "SPARSE [in/out]:\n";
	chomp(my $inputFile2 = <STDIN>);
	chomp(my $outputFile2 = <STDIN>);
	my $v2 = utilities::readSparseVectorFile($inputFile2);
	utilities::outputSparseVectors($outputFile2, $v2);

}


#reads in the semmed file and makes it into one hot term vectors
sub makeSomeSemMedHotties{
	#parameters
	my $inputSemMedFile = shift;
	my $outputSemMedFile = shift;

	#import it I guess
	open(my $FILE, "<", $inputSemMedFile) || die "Cannot open input semmed file $inputSemMedFile\n";
	my @lines = <$FILE>;
	chomp(@lines);
	close $FILE;

	#read in all the preds listed
	my %predHash = ();
	foreach my $line(@lines){
		my @parts = split(/\t/, $line);
		foreach my $part (@parts){
			#check if the part is a pred
			if($part =~/\b[a-zA-Z]+\b/){
				$predHash{$part}++;	#add it to the set
			}
		}
	}

	#open the output file
	open (my $FILE2, ">", $outputSemMedFile) || die "Cannot open output semmed file $outputSemMedFile\n";

	#assign all the preds to the index
	my @predList = (sort keys %predHash);
	for(my $i=0;$i<@predList;$i++){
		print $FILE2 ($predList[$i] . "<>$i,1\n");
	}
	close $FILE2;
}

#reads in the dense vector file and makes it into one hot term vectors
sub makeSomeDenseHotties{
	#parameters
	my $inputVectorFile = shift;
	my $outputVectorFile = shift;

	my $v = utilities::readDenseVectorFile($inputVectorFile);
	my %vectors = %$v;

	#open the output file
	open (my $FILE, ">", $outputVectorFile) || die "Cannot open output vector file $outputVectorFile\n";

	#assign all the preds to the index
	my @predList = (sort keys %vectors);
	for(my $i=0;$i<@predList;$i++){
		print $FILE ($predList[$i] . "<>$i,1");
	}
	close $FILE;
}

#reads in the sparse vector file and makes it into one hot term vectors
sub makeSomeSparseHotties{
	#parameters
	my $inputVectorFile = shift;
	my $outputVectorFile = shift;

	my $v = utilities::readSparseVectorFile($inputVectorFile);
	my %vectors = %$v;

	#open the output file
	open (my $FILE, ">", $outputVectorFile) || die "Cannot open output vector file $outputVectorFile\n";

	#assign all the preds to the index
	my @predList = (sort keys %vectors);
	for(my $i=0;$i<@predList;$i++){
		print $FILE ($predList[$i] . "<>$i,1");
	}
	close $FILE;
}

semmedOut();

1;