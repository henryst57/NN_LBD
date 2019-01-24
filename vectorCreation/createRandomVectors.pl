# Generates vectors of random values for each cui and predicate in the training file
# Vector values range from 0 to 1
# Output is in vector file format:
#   (same as word2vec format) e.g.: first line contains two space seperated 
#   values, the number of vectors and dimensionality vector dimensionality 
#   (e.g. 508914 200)
#   Each subsequent line contsins the term and a space seperated list of 
#   values (e.g. term1 val1 val2 val3 ... valn)
# Results are output to the $cuiOutFile and $predOutFile.
use strict;
use warnings;
use lib '../';
use utilities;


############################################################
#  User Input
############################################################
#matrix file containing all words in the current vocabulary
my $trainingFile = '../../data/testTrain/known_newSmall';

#file to output the random vectors to
my $cuiOutFile = 'vectors_random_cuis';
my $predOutFile = 'vectors_random_predicates';
    
#vector sizes
my $cuiVectorSize = 200;
my $predicateVectorSize = 200;

#create the random vector
&createRandomOut($trainingFile, $cuiOutFile, $predOutFile, $cuiVectorSize, $predicateVectorSize);



############################################################
#  Begin Code
############################################################

#Creates random CUI and Predicate Vectors for all Cuis and predicates in the file
sub createRandomOut {
    my $trainingFile = shift;
    my $cuiOutFile = shift;
    my $predOutFile = shift; 
    my $cuiVectorSize = shift; 
    my $predicateVectorSize = shift;
    
    #read in the vocabulary by reading in the matrix file
    my $semMedMatrix = &utilities::readSemMedFile($trainingFile);     
    my ($vocabularyRef, $relationTypesRef) = &utilities::getVocabularyAndPredicateTypes($semMedMatrix);  

    #output cui and predicate random vectors
    &outputFile($cuiOutFile, $cuiVectorSize, $vocabularyRef);
    &outputFile($predOutFile, $predicateVectorSize, $relationTypesRef);

    print "Done!\n";
}

#Outputs random vectors for each term
sub outputFile {
    my $outFile = shift;
    my $vectorSize = shift;
    my $termsHash = shift;

    #generate random score vectors for each term and outputs
    open OUT, ">$outFile" 
	or die ("ERROR: cannot open outFile: $outFile\n");
    #output the header
    print OUT (scalar keys %{$termsHash})." $vectorSize\n";

    #create and output each vector
    foreach my $val (keys %{$termsHash}) {
	print OUT "$val";
	for (my $i = 0; $i < $vectorSize; $i++) {
	    print OUT " ".rand();
	}
	print OUT "\n";
    }
    close OUT;
}
