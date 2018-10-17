# File containing utility methods 
use strict;
use warnings;
package utilities;


#gets the vocabulary and predicate types in the semmedMatrix
# matrix is of the format:
#hash{SUBJECT_CUI,PRED_TYPE,OBJECT_CUI}=1;
sub getVocabularyAndPredicateTypes {
    my $semmedMatrix = shift;
    
    #construct hashes of unique cuis and predication types
    my %vocabulary = ();
    my %relationTypes = ();
    foreach my $triplet (keys %{$semmedMatrix}) {
	#grab the values
	my @vals = split (/,/,$triplet);
	#$triplet = "subjectCUI,relationType,objectCUI"
	my $subject = $vals[0];
	my $relationType = $vals[1];
	my $object = $vals[2];

	#add values to the vocab/relation types
	$relationTypes{$relationType}=1;
	$vocabulary{$subject}=1;
	$vocabulary{$object}=1;
    }

    return \%vocabulary, \%relationTypes;
}

#########################################################
#  SemMedDB Dataset I/O
#########################################################

#outputs the dataset of format:
#   hash{SUBJECT_CUI,PRED_TYPE,OBJECT_CUI}=1;
# to the file with the format:
#   SUBJECT_CUI\tPRED_TYPE\tOBJECT_CUI\tOBJECT_CUI\t...\n
sub outputDataset {
    my $dataSetRef = shift;
    my $outFile = shift;
    
    #convert the hash for output
    my %outputHash = ();
    foreach my $triplet (keys %{$dataSetRef}) {
	#grab the values
	my @vals = split (/,/,$triplet);
	#$triplet = "subjectCUI,relationType,objectCUI"
	my $subjectCUI = $vals[0];
	my $relationType = $vals[1];
	my $objectCUI = $vals[2];

	#add to the hash
	if (!defined $outputHash{"$subjectCUI,$relationType"}) {
	    my @newArray = ();
	    $outputHash{"$subjectCUI,$relationType"} = \@newArray;
	}
	push @{$outputHash{"$subjectCUI,$relationType"}}, $objectCUI;

    }

    #output the hash
    open OUT, ">$outFile" or die ("ERROR: unable to open outFile: $outFile\n");
    foreach my $pair (keys %outputHash) {
    	#grab pair values
    	my @vals = split (/,/,$pair);
    	my $subject = $vals[0];
    	my $rel = $vals[1];
    	
    	#output each object for which the pair holds true
    	print OUT $pair;
    	foreach my $object (@{$outputHash{$pair}}) {
    	    print OUT "\t".$object;
    	}
    	print OUT "\n";
    }
    close OUT;
}


#reads a semmedDB file contains for each line a subject CUI
# and a predicate type, and a tab seperated list of all
# obects CUIs for which that subject predicate pair holds true
# Specific Format:
#    SUBJECT_CUI\tPRED_TYPE\tOBJECT_CUI\tOBJECT_CUI\t...\n
# and returns a hash of hashes of format:
#    hash{SUBJECT_CUI}{PRED_TYPE}{OBJECT_CUI}=1;
sub readSemMedFile {
    my $fileName = shift;

    #read the file into the dataset
    open IN, $fileName or die ("ERROR: unable to open file: $fileName\n");
    my %dataSet = ();
    while (my $line = <IN>) {
    	chomp $line;
        
        #grab values from the line
        my @vals = split ("\t",$line);
    	#$line=subject\trelType\tlistOfObjects
    	my $subject = shift @vals;
    	my $relType = shift @vals;
    	
    	#add each triplet to the dataset
    	foreach my $object (@vals) {
    	    $dataSet{"$subject,$relType,$object"} = 1;
    	}
    }
    close IN;

    return \%dataSet;
}


####################################################
#       Vector I/O
####################################################


#Reads vectors from a dense vector file.
# Dense vector file is of the format:
#   CUI VAL VAL VAL ... 
# and returns the vector as a hash, of the format:
#   {CUI}{INDEX} = VALUE
#
# INPUT:
#   $vectorFile - the name of the file containing the vectors
# OUTPUT:
#   %vectors - a hash of the cuis, indexes, and values
sub readDenseVectorFile {
    my $vectorFile = shift;

    #set up variables
    my %vectors = ();

    #open the file and read in the data
    open(my $FILE, "<", $vectorFile) || die "Vector file not found @ $vectorFile\n";
    my @lines = <$FILE>;
    chomp(@lines);

    #format the data for the hash function
    foreach my $line(@lines){
    	# split up the line by the cui and values
    	# should be in the format:
    	#   CUI VAL VAL VAL ... 
    	my ($cui, @vals) = split(/\s/, $line);
    	my $index = 0;

        #iterate through each value and assign the cui-index pair to the value
    	for(my $i=0;$i<$#vals;$i++){
    		$vectors{$cui}{$i} = $vals[$i];
    	}
    	
    }

    #close the file
    close $FILE;

    #return the hash
    return \%vectors
}


#Reads vectors from a sparse vector file.
# Sparse vector file is of the format:
#   CUI<>index,val<>index,val<> ...
# and returns the vector as a hash, of the format:
#   {CUI}{INDEX} = VALUE
#
# INPUT:
#   $vectorFile - the name of the file containing the vectors
# OUTPUT:
#   %vectors - a hash of the cuis, indexes, and values
sub readSparseVectorFile {
    my $vectorFile = shift;

    #set up variables
    my %vectors = ();

    #open the file and read in the data
    open(my $FILE, "<", $vectorFile) || die "Vector file not found @ $vectorFile\n";
    my @lines = <$FILE>;
    chomp(@lines);

    #format the data for the hash function
    foreach my $line(@lines){
    	# split up the line by the cui, indexes, and values
    	# should be in the format:
    	#   CUI<>index,val<>index,val<> ...
    	my ($cui, @ivs) = split(/<>/, $line, 0);

        #assign the cui and index pair to the value
        foreach my $iv (@ivs){
            my ($index, $value) = split(",", $iv);
            $vectors{$cui}{$index} = $value;
        }
    	
    }

    #close the file
    close $FILE;

    #return the hash
    return \%vectors
}


#Outputs a vectors hash to a dense vector file
# Sparse vector file is of the format:
#   CUI VAL VAL VAL ... 
# and returns the vector as a hash, of the format:
#   {CUI}{INDEX} = VALUE
#
# INPUT:
#   $outFile - the name of the file containing the vectors
#   %vectors - a hash of the cuis, indexes, and values
sub outputDenseVectors {
    my $outFile = shift;
    my $vectorsRef = shift;
    my %vectors = %$vectorsRef;

    #open the file for writing
    my $FILE;
    open($FILE, ">", $outFile) || die "Cannot open output file $outFile\n";

    #iterate through every cui
    foreach my $cui (sort keys %vectors){
        print $FILE $cui;

        #print the values 
        foreach my $index (sort { $a <=> $b } keys $vectors{$cui}){
            print $FILE (" " . $vectors{$cui}{$index});
        }

        #next cui pair
        print $FILE "\n";   
    }

    #close the file
    close $FILE;
}

#Outputs a vectors hash to a sparase vector file
# Sparse vector file is of the format:
#   CUI<>index,val<>index,val<> ...
#
# INPUT:
#   $outFile - the name of the file containing the vectors
#   %vectors - a hash of the cuis, indexes, and values
sub outputSparseVectors {
    my $outFile = shift;
    my $vectorsRef = shift;
    my %vectors = %$vectorsRef;

    #open the file for writing
    my $FILE;
    open($FILE, ">", $outFile) || die "Cannot open output file $outFile\n";

    #iterate through every cui
    foreach my $cui (sort keys %vectors){
        print $FILE $cui;

        #print the index-value pairs
        foreach my $index (sort { $a <=> $b } keys $vectors{$cui}){
            print $FILE ("<>$index," . $vectors{$cui}{$index});
        }

        #next cui pair
        print $FILE "\n";   
    }

    #close the file
    close $FILE;
}



1;


