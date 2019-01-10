use lib '..';
use utilities;

#removes lines with uncommon CUI vocab

#gets the vocabulary and predicate types in the semmedMatrix
# matrix is of the format:
#hash{SUBJECT_CUI,PRED_TYPE,OBJECT_CUI}=1;
sub getVocabulary{
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

    return %vocabulary;
}

#gets the vocabulary and predicate types in the semmedMatrix
# matrix is of the format:
#hash{SUBJECT_CUI,PRED_TYPE,OBJECT_CUI}=1;
sub getPredicates{
    my $semmedMatrix = shift;
    
    #construct hashes of unique cuis and predication types
    my %vocabulary = ();
    my %relationTypes = ();
    foreach my $triplet (keys %{$semmedMatrix}) {
	#grab the values
	my @vals = split (/,/,$triplet);
	#$triplet = "subjectCUI,relationType,objectCUI"
	my $relationType = $vals[1];

	#add values to the vocab/relation types
	$relationTypes{$relationType}=1;
    }

    return %relationTypes;
}

#reads a semmedDB file contains for each line a subject CUI
# and a predicate type, and a tab seperated list of all
# obects CUIs for which that subject predicate pair holds true
# Specific Format:
#    SUBJECT_CUI\tPRED_TYPE\tOBJECT_CUI\tOBJECT_CUI\t...\n
# and returns a hash of hashes of format:
#    hash{SUBJECT_CUI,PRED_TYPE,OBJECT_CUI}=1;
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

    return %dataSet;
}


print "Type [CUI | PREDICATE]: ";
chomp(my $typeParam = <STDIN>);
#get the inputs
print "Dictionary File: ";			#assuming the dictionary file is a semmed file
chomp(my $vocabFile = <STDIN>);
print "File to strip: ";			#assuming the strip file is a dense format
chomp(my $stripFile = <STDIN>);

#save to an output file
print "Output File: ";
chomp(my $outFile = <STDIN>);

my %semMed = readSemMedFile($vocabFile);


my %set = ();
#read in the vocabulary
if($typeParam eq "CUI"){
	%set = getVocabulary(\%semMed);
}elsif($typeParam eq "PREDICATE"){
	%set = getPredicates(\%semMed);
}else{
	print "CANNOT RESOLVE TYPE $typeParam\n";
	exit;
}

#create a new set
my @newSet = ();
open(my $STRIPFILE, "<", $stripFile) || die "Cannot open the file to strip @ $stripFile\n";
my @lines = <$STRIPFILE>;
close($STRIPFILE);

#add first line by default
#push(@newSet, $lines[0]);

#check if the line's CUI is there
for my $line (@lines){
	my @parts = split(/\s/, $line);
	my $cui = $parts[0];
   	# print "$cui  --> ";
	#print $cui . " - " . $vocabSet{$cui} . "\n";
	if(exists $set{$cui}){
        #print "YES\n";
		push(@newSet, $line);
	}else{
        #print "NO\n";
    	}	
}

print("PRINTING " . $#newSet . " lines....\n");

#add number of lines and density
unshift @newSet, ($#newSet-1 . " 200\n");

#output the final result
open(my $OUTPUTFILE, ">", $outFile) || die "Cannot create output file @ $outFile\n";
for my $line2(@newSet){
	print $OUTPUTFILE "$line2";
}

close $OUTPUTFILE;



1;
