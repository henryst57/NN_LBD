#Program to reduce the size of the dataset based on selecting subject
# CUIs only, and applying thresholds per number of object CUIs
use strict;
use warnings;
use lib '../';
use utilities;

#User Input
my $semmedFile = '../../data/testTrain/true_newSmall';
my $minNumObjectsThreshold = 5; #remove if <= threshold

#Auto-Generated Output FileNames
my $outFile = "../../data/testTrain/true_newSmall_smartReduce_$minNumObjectsThreshold"; 

#process the dataset
&removeNonSubjectsAndThreshold($semmedFile, $minNumObjectsThreshold, $outFile);

################################################
# Begin Code
################################################

#Removes any object CUIs that are not also subject CUIs
# and applies a minimum number of unique objects threshold
sub removeNonSubjectsAndThreshold {
    my $semmedFile = shift;
    my $threshold = shift;
    my $outFile = shift;

    #read the matrix and vocab
    my $semmedMatrix = &utilities::readSemMedFile($semmedFile);

    #grab the subject CUIs and record the objects CUIs for which
    # they are true as a ${hash{subject}}{$object}
    my %subjectCuis = ();
    foreach my $triplet (keys %{$semmedMatrix}) {
	#grab the values
	my @vals = split(/,/,$triplet);
	my $subject = $vals[0];
	my $relationType = $vals[1];
	my $object = $vals[2];

	#record the subject and object
	if (!defined $subjectCuis{$subject}) {
	    my %newHash = ();
	    $subjectCuis{$subject} = \%newHash;
	}
	${$subjectCuis{$subject}}{$object} = 1;
    }

    #remove non-subjects and apply threshld
    # This may need to be done iteratively because applying the
    # threshold may necessitate removing objects, then re-applying
    # the threshold and so on
    my $done = 0;
    my $reducedSubjectCuis = \%subjectCuis;
    while (!$done) {
        print STDERR "num subject cuis pre = ".(scalar keys %{$reducedSubjectCuis})."\n";
        #apply reduction steps
	$reducedSubjectCuis = &removeNonSubjectObjects($reducedSubjectCuis);
	$reducedSubjectCuis = &applyThreshold($reducedSubjectCuis, $threshold);
	$reducedSubjectCuis = &removeNonSubjectObjects($reducedSubjectCuis);
	print STDERR "num subject cuis post = ".(scalar keys %{$reducedSubjectCuis})."\n";

	#Check if done
	$done = 1;
	foreach my $subject (keys %{$reducedSubjectCuis}) {
	    my $numObjects = scalar keys %{${$reducedSubjectCuis}{$subject}};
	    if ($numObjects <= $threshold) {
		$done = 0;
		last;
	    }
	}
    }

    #reduce the original dataset now that the subject cuis
    # have been selected, and output it
    foreach my $triplet (keys %{$semmedMatrix}) {
	#grab the values
	my @vals = split(/,/,$triplet);
	my $subject = $vals[0];
	my $object = $vals[2];

	#record the subject and object
	if (!defined ${$reducedSubjectCuis}{$subject} || !defined ${$reducedSubjectCuis}{$object}) {
	    delete ${$semmedMatrix}{$triplet};
	}
    }


=comment
    #find the number of subject and object cuis
    my %uniqueSubjects = ();
    my %uniqueObjects = ();
    foreach my $triplet (keys %{$semmedMatrix}) {
	#grab the values
	my @vals = split(/,/,$triplet);
	my $subject = $vals[0];
	my $object = $vals[2];

	$uniqueSubjects{$subject} = 1;
	$uniqueObjects{$object} = 1;
    }
    print "numSubjects = ".(scalar keys %uniqueSubjects)."\n";
    print "numObjects = ".(scalar keys %uniqueObjects)."\n";
=cut

    #output the reduced dataset
    &utilities::outputDataset($semmedMatrix, $outFile)    
}

=comment
#removes any object that are not also subject CUIs
sub removeNonSubjectObjects {
    my $subjectCuis = shift;

    #remove any object CUIs that are not subject CUIs
    foreach my $subject (keys %{$subjectCuis}) {
	foreach my $object (keys %{${$subjectCuis}{$subject}}) {
	    if (!defined ${$subjectCuis}{$object}) {
		delete ${${$subjectCuis}{$subject}}{$object};
	    }
	}
    }

    return $subjectCuis;
}
=cut


#removes any subject CUIs with <= threshold unique object CUIs
sub applyThreshold {
    my $subjectCuis = shift;
    my $threshold = shift;

    #apply a minimum objects threshold
    foreach my $subject (keys %{$subjectCuis}) {
	my $numObjects = scalar keys %{${$subjectCuis}{$subject}};
        if ($numObjects <= $threshold) {
	    delete ${$subjectCuis}{$subject};
	}
    }

    return $subjectCuis;
}


