#Program to reduce the known and true datasets based on randomly
# selecting n terms from the vocabulary
use strict;
use warnings;
use lib '../';
use utilities;

#User Input
my $knownFile = '../../data/testTrain/known_newSmall';
my $trueFile = '../../data/testTrain/true_newSmall';
my $reducedVocabSize = 5000;

#Auto-Generated Output FileNames
my $knownOut = "../../data/testTrain/known_newSmall_$reducedVocabSize";
my $trueOut = "../../data/testTrain/true_newSmall_$reducedVocabSize";



################################################
# Begin Code
################################################


#read the matrix and vocab
my $knownMatrix = &utilities::readSemMedFile($knownFile);
my ($vocabularyRef, $relationTypesRef) = &utilities::getVocabularyAndPredicateTypes($knownMatrix);


#randomly?? select $vocabSize terms
my %sampledVocab = ();
foreach my $cui (keys %{$vocabularyRef}) {
    $sampledVocab{$cui} = 1;
    if (scalar keys %sampledVocab >= $reducedVocabSize) {
	last;
    }
}

#generate the reduced known set
my %reducedKnownMatrix = ();
foreach my $triplet (keys %{$knownMatrix}) {
    my ($subject, $predicate, $object) = split(',',$triplet);
    if (defined $sampledVocab{$subject} && defined $sampledVocab{$object}) {
	$reducedKnownMatrix{$triplet} = 1;
    }
}

#generate the reduced true set
my $trueMatrix = &utilities::readSemMedFile($trueFile);
my %reducedTrueMatrix = ();
foreach my $triplet (keys %{$trueMatrix}) {
    my ($subject, $predicate, $object) = split(',',$triplet);
    if (defined $sampledVocab{$subject} && $sampledVocab{$object}) {
	$reducedTrueMatrix{$triplet} = 1;
    }
}

print "size of reducedKnown: ".(scalar keys %reducedKnownMatrix)."\n";
print "size of reducedTrue: ".(scalar keys %reducedTrueMatrix)."\n";
print "size of sampledVocab: ".(scalar keys %sampledVocab)."\n";

#output the reduced known and reduced true
&utilities::outputDataset(\%reducedKnownMatrix, $knownOut);
&utilities::outputDataset(\%reducedTrueMatrix, $trueOut);

#Done!
print "Done!\n";


