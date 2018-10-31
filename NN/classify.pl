# classifies a test set using a trained NN and outputs results as a classification file

#!/bin/perl
use strict;
use warnings;
use lib '/home/share/NN_LBD_NEW/code';
require utilities;	#import the utilities.pm module


# train semmedDB fil
# this is found under data/testTrain/true
my $file = "../../data/testTrain/true"

#import the train data using utilities
sub getTrainData{
	return utilities::readSemMedFile($trainFile);
}


