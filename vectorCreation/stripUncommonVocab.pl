use utilities;


#removes lines with uncommon CUI vocab

#get the inputs
print "Dictionary File: ";			#assuming the dictionary file is a sparse format
chomp(my $vocabFile = <STDIN>);
print "File to strip: ";			#assuming the strip file is a dense format
chomp(my $stripFile = <STDIN>);

#save to an output file
print "Output File: ";
chomp(my $outFile = <STDIN>);

#read in the vocabulary
my %vocabSet = readSparseFile($vocabFile);

#create a new set
my @newSet = ();
open(my $STRIPFILE, "<", $stripFile) || die "Cannot open the file to strip @ $stripFile\n";
my @lines = <$STRIPFILE>;
close($STRIPFILE);
#check if the line's CUI is there
for my $line (@lines){
	my @parts = split(/\s/, $line);
	my $cui = $parts[0];
	if(exists $vocabSet{$cui}){
		push(@newSet, $line);
	}
}

#output the final result
open(my $OUTPUTFILE, ">", $outFile) || die "Cannot create output file @ $outFile\n";
for my $line2(@newSet){
	print $OUTPUTFILE $line2;
}
exit;



1;
