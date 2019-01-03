# collects predications from PMIDs that occur in the date ranges and outputs 
# them in a format that is read into NN based LBD
use strict;
use warnings;

#start and end years (inclusive start, inclusive end)
my $startYear = 1975;
my $endYear = 2009;

#The folder containing subfolders with data by year
# each subfolder should be labeled the year, and contain file that is a line 
# seperated list of PMIDs for that year. This file should be called pmidList.txt
# See TODO to create the folder indeces
my $indexedByYearFolder = '/home/henryst/home/sam/lbd/data/indexedAll/';

#flat file containing predication info from a database. The file
# contains line seperated predications of the format:
#  CUI/tpredicateType/tCUI/tPMID/Number
# See semmedDB2Mat_withPredicates.pl to create the flat file
my $predicationFlatFile = '../../data/semmedDB/SemMedDB_Matrix_allPredications_withPredicates';

#The file that will contain the program output
my $outputFile = '../../data/semmedDB/allPredicates_'.$startYear.'_'.$endYear;

#Apply the date range
&_applyDateRange($startYear, $endYear, $indexedByYearFolder, $predicationFlatFile, $outputFile);


#Construct for the test set
$startYear = 2010;
$endYear = 2018;
$outputFile = '../../data/semmedDB/allPredicates_'.$startYear.'_'.$endYear;
&_applyDateRange($startYear, $endYear, $indexedByYearFolder, $predicationFlatFile, $outputFile);



#################################################################
# Begin Code
#################################################################
# This code first builds a hash of all pmids in the date range.
# It then reads a file containing predications and keeps only
# the predications that come from pmids in the date range. The
# output is then a file of new line seperated predications that
# come from the date range.
sub _applyDateRange {
    my $startYear = shift;
    my $endYear = shift;
    my $indexedByYearFolder = shift;
    my $predicationFlatFile = shift;
    my $outputFile = shift;

    #the file that contains pmids for eachyear
    my $pmidFileName = 'pmidList.txt';

    #build a hash of all pmids in the date range
    my %pmids = ();
    for (my $year = $startYear; $year <= $endYear; $year++) {
	my $fileName = $indexedByYearFolder.$year.'/'.$pmidFileName;
	if(open IN, $fileName) {
	    while (my $line = <IN>) {
		#lines are new line seperated pmids, so read, chomp, and add to hash
		chomp $line;
		$pmids{$line} = $1;
	    }
	    close IN;
	} 
	else {
	    print "error opening pmid file, skipping year: $fileName";
	}
    }
    #done reading the file of pmids

    # read file of predications and select only the predications that came from
    # one of the pmids in the date range
    #open input and output files
    open IN, $predicationFlatFile or die("Error: Cannot open predicationFlatFile: $predicationFlatFile\n");
    open OUT, ">$outputFile" or die ("Error: Cannot open outputFile: $outputFile\n");

    #read each line of the predication file and output if pmids match
    my @vals = ();
    while (my $line = <IN>) {
	chomp $line;

	#grab the pmid (3rd element on the line)
	@vals = split(/\t/, $line);
	my $pmid = $vals[3];

	#check if the pmid is within date range
	if (exists $pmids{$pmid}) {
	    #it exists, so output it to the output file
	    print OUT "$line\n";
	}
    }
    close OUT;
    close IN;
}






