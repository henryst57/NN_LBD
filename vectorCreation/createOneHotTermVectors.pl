#!/bin/perl
use strict;
use warnings;
use lib '../';
require utilities;	#import the utilities.pm module


#TODO - read in SemmedDB and create and create one hot vectors for each term in the vocab

#Input: semmedDB matrix of known relations (known file)
#  format is:
#
#Output:
# denseFormat:
# 	cui value value value ...
# 
# sparseFormat:
# 	"cui<>index,value<>index,value<> ..."
#


sub semmedOut{
	my $inputFile = $ARGV[0];
	my $outputFile = "vectors_onehot_cuis";

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

	#import it
	open(my $FILE, "<", $inputSemMedFile) || die "Cannot open input semmed file $inputSemMedFile\n";
	my @lines = <$FILE>;
	chomp(@lines);
	close $FILE;

	#read in all the cuis listed
	my %cuiHash = ();
	foreach my $line(@lines){
		my @parts = split(/\t/, $line);
		foreach my $part (@parts){
			#check if the part is a cui
			if($part =~/\bC[0-9]+\b/){
				$cuiHash{$part}++;	#add it to the set
			}
		}
	}

	#open the output file
	open (my $FILE2, ">", $outputSemMedFile) || die "Cannot open output semmed file $outputSemMedFile\n";

	#assign all the cuis to the index
	my @cuiList = (sort keys %cuiHash);
	for(my $i=0;$i<@cuiList;$i++){
		print $FILE2 ($cuiList[$i] . "<>$i,1\n");
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

	#assign all the cuis to the index
	my @cuiList = (sort keys %vectors);
	for(my $i=0;$i<@cuiList;$i++){
		print $FILE ($cuiList[$i] . "<>$i,1\n");
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

	#assign all the cuis to the index
	my @cuiList = (sort keys %vectors);
	for(my $i=0;$i<@cuiList;$i++){
		print $FILE ($cuiList[$i] . "<>$i,1");
	}
	close $FILE;
}

semmedOut();


1;
