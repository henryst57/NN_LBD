#Program to reduce the size of the dataset based on selecting subject
# CUIs only, and applying thresholds per number of object CUIs
use strict;
use warnings;
use lib '../';
use utilities;

#User Input
my $semmedFile = '../../data/testTrain/true_newSmall';
my $minNumObjectsThreshold = 18;             #removes if <= threshold
my $minNumSubjectsThreshold = $minNumObjectsThreshold;

#Auto-Generated Output FileNames
my $outFile = "../../data/testTrain/true_newSmall_threshold_$minNumObjectsThreshold"."_$minNumSubjectsThreshold"; 

#process the dataset
&thresholdDataset($semmedFile, $minNumObjectsThreshold, $minNumSubjectsThreshold, $outFile);

################################################
# Begin Code
################################################

#Removes any object CUIs that are not also subject CUIs
# and applies a minimum number of unique objects threshold
sub thresholdDataset {
    my $semmedFile = shift;
    my $minNumObjectsThreshold = shift;
    my $minNumSubjectsThreshold = shift;
    my $outFile = shift;

    #read the matrix and vocab
    my $predicates = &utilities::readSemMedFile($semmedFile);


    #remove non-subjects and apply threshld
    # This may need to be done iteratively because applying the
    # threshold may necessitate removing objects, then re-applying
    # the threshold and so on
    my $done = 0;
    while (!$done) {
	$predicates = &applyThreshold($predicates, $minNumSubjectsThreshold, $minNumObjectsThreshold);
	my ($objectsPerSubject, $subjectsPerObject) = &getCuisPerCui($predicates);


        #Check if done (threshold does not need to be re-applied)
	$done = 1;
	foreach my $subject (keys %{$objectsPerSubject}) {
	    my $numObjects = scalar keys %{${$objectsPerSubject}{$subject}};
	    if ($numObjects <= $minNumObjectsThreshold) {
		$done = 0;
		last;
	    }
	}
	if ($done) {
	    foreach my $object (keys %{$subjectsPerObject}) {
		my $numSubjects = scalar keys %{${$subjectsPerObject}{$object}};
		if ($numSubjects <= $minNumSubjectsThreshold) {
		    $done = 0;
		    last;
		}
	    }
	}
    }


    #output the final counts of unique subjects and objects
    #find the number of subject and object cuis
    my %uniqueSubjects = ();
    my %uniqueObjects = ();
    foreach my $triplet (keys %{$predicates}) {
	#grab the values
	my @vals = split(/,/,$triplet);
	my $subject = $vals[0];
	my $object = $vals[2];

	$uniqueSubjects{$subject} = 1;
	$uniqueObjects{$object} = 1;
    }
    print "\n Final Counts:\n";
    print "numSubjects = ".(scalar keys %uniqueSubjects)."\n";
    print "numObjects = ".(scalar keys %uniqueObjects)."\n";


    #output the reduced dataset
    &utilities::outputDataset($predicates, $outFile)    
}



#Applies a minimum subjects per object and minimum objects
# per subject threshold to the predicates hash
sub applyThreshold {
    my $predicates = shift;
    my $minNumSubjectsThreshold = shift;
    my $minNumObjectsThreshold = shift;

    #get lists of cuis per subject/object
    my ($objectsPerSubject, $subjectsPerObject) = &getCuisPerCui($predicates);

    #threshold the matrix
    foreach my $triplet (keys %{$predicates}) {
	#grab the values
	my @vals = split(/,/,$triplet);
	my $subject = $vals[0];
	my $object = $vals[2];

	#record the subject and object
	if (scalar keys %{${$objectsPerSubject}{$subject}} <= $minNumObjectsThreshold
	    || scalar keys %{${$subjectsPerObject}{$object}} <= $minNumSubjectsThreshold) {
	    delete ${$predicates}{$triplet};
	}
    }
    
    return $predicates;
}

#gets two hashes, which list the cuis that are objects
# for each subject (objectsPerSubject), and another that
# lists the subjects per object. Both are of the form:
# ${$cuisPer{$cui}}{$cui} = 1
sub getCuisPerCui {
    my $predicates = shift;

    #find the object lists and subject lists for each 
    # subject and object in the matrix
    my %objectsPerSubject = ();
    my %subjectsPerObject = ();
    foreach my $triplet (keys %{$predicates}) {
	#grab the values
	my @vals = split(/,/,$triplet);
	my $subject = $vals[0];
	my $relationType = $vals[1];
	my $object = $vals[2];

	#record the subject and object
	if (!defined $objectsPerSubject{$subject}) {
	    my %newHash = ();
	    $objectsPerSubject{$subject} = \%newHash;
	}
	if (!defined $subjectsPerObject{$object}) {
	    my %newHash = ();
	    $subjectsPerObject{$object} = \%newHash;
	}
	
	${$objectsPerSubject{$subject}}{$object} = 1;
	${$subjectsPerObject{$object}}{$subject} = 1;
    }

    return (\%objectsPerSubject, \%subjectsPerObject);
}
