#Script to divide a SemMedDB matrix into true, false, and known sets
#Input:
#  preCutoffFile - semmedFile of pre-cutoff data
#  postCutoffFile - semmedFile pf post-cutoff data
#
#  semmedDB files contain for each line a subject CUI
#  and a predicate type, and a tab seperated list of all
#  obects CUIs for which that subject predicate pair holds true
#   Specific Format:
#      SUBJECT_CUI\tPRED_TYPE\tOBJECT_CUI\tOBJECT_CUI\t...\n
#
#
#  and the folder to output the split dataset into:
#     outputFolder
#
#Output:
#  known - dataset file of known predication triplets
#  true - dataset file of true predication triplets
#  false - dataset file of all possible false predication triplets
#
#  these three files are output to the output folder and have the
#  the same format as SemmedDB files specified at input (e.g.
#     SUBJECT_CUI\tPRED_TYPE\tOBJECT_CUI\tOBJECT_CUI\t...\n)
#
use strict;
use warnings;
use lib '../';
use utilities;


#user input
my $preCutoffFile = '../../data/semmedDB/data_1975_2009_uniqueCuis_nonNegativePred';
my $postCutoffFile = '../../data/semmedDB/data_2010_2018_uniqueCuis_nonNegativePred';
my $outputFolder = '../../data/testTrain/';
&_createTestTrain($preCutoffFile, $postCutoffFile, $outputFolder);




######################################################
#                   Begin Code
#######################################################

#creates True, False, and Known Sets from the pre and post-cutoff 
# SemMedDB files
sub _createTestTrain {
    #grab input
    my $preCutoffFile = shift;
    my $postCutoffFile = shift;
    my $outputFolder = shift;

    ### read the SemMedDB data, hashes are of the form hash{triplet}=1
    #     where triplet is: "subject,relation,object"
    my $preCutoffDataRef = &utilities::readSemMedFile($preCutoffFile);
    my $postCutoffDataRef = &utilities::readSemMedFile($postCutoffFile);
    my ($vocabularyRef, $predicateTypesRef) = &utilities::getVocabularyAndPredicateTypes($preCutoffDataRef);


    ### generate datasets
    #Note: we could save memory with our hash usage for known and future
    # triplets, but kept this way for clarity

    #known triplets are all triplets that are known prior to the cutoff date
    my %knownTriplets = %{$preCutoffDataRef};
    
    #generate true triplets, these are triplets that are true
    # future predications, so post-cutoff triplets that contains
    # pre-cutoff vocabulary and are not already known
    my %trueTriplets = ();
    foreach my $triplet (keys %{$postCutoffDataRef}) {
	#grab values from the truplet
	my @vals = split (/,/,$triplet);
	#triplet = subject,relation,object
	my $subject = $vals[0];
	my $relType = $vals[1];
	my $object = $vals[2];
	
	#add to true only if subject, object, and relType are in the 
	# pre-cutoff matrix vocabulary
        if (defined ${$vocabularyRef}{$subject} 
	    && defined ${$vocabularyRef}{$object}
	    && defined ${$predicateTypesRef}{$relType}) {
	    $trueTriplets{$triplet}=1;
	}
    }

=comment
    #generate false triplets, these are all possible triplets
    # that are not known and not true
    my %falseTriplets = ();
    foreach my $subject (keys %{$vocabularyRef}) {
	foreach my $object (keys %{$vocabularyRef}) {
	    foreach my $rel (keys %{$predicateTypesRef}) {
		#generate the triplet
		my $triplet = "$subject,$rel,$object";
		
		#check if this triplet is either a true future triplet
		# or an already known triplet (occurs in pre-cutoff)
		if (!defined $knownTriplets{$triplet} 
		    && !defined $trueTriplets{$triplet}) {
		    $falseTriplets{$triplet}=1;
		}
	    }
	}
    }
=cut
    
    ### Output each dataset
    &utilities::outputDataset(\%knownTriplets, $outputFolder.'known');
    &utilities::outputDataset(\%trueTriplets, $outputFolder.'true');
    #&utilities::outputDataset(\%falseTriplets, $outputFolder.'false');

    print "Done!\n";
}


