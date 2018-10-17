#Outputs a file containing: CUI PRED CUI PMID for each entry in the 
# mysql predicate database
use DBI;
use strict;
use warnings;

my $dbh = "";

#add the desired semantic types to the list of acceptable predicates
my $outputFileName = '../../data/semmedDB/SemMedDB_Matrix_allPredications_withPredicates';

#get the frequency of cui pairs
&convert($outputFileName) ;


#gets the frequency cui pairs co-occur in SemMedDB regardless of the predicate type
sub convert {
    #grab and check input
    my $outputFileName = shift;
    open OUT, ">$outputFileName" or die ("ERROR: unable to open outputFile: $outputFileName");
    
    #set up the database
    my $database = "semmedVER30";
    my $hostname = "192.168.24.89";
    my $userid = "henryst";
    my $password = "OhFaht3eique";
    my %thingy = ();
    my $dsn = "DBI:mysql:database=$database;host=$hostname;";
    $dbh = DBI->connect($dsn, $userid, $password, \%thingy) or die $DBI::errstr;
    $dbh->{InactiveDestroy} = 1; #allows forking of threads containing this DB connect

    #set the query to get all cui pairs of the specified predicate types
    my $query =  'SELECT SUBJECT_CUI, PREDICATE, OBJECT_CUI, PMID FROM semmedVER30.PREDICATION';

    #query the db
    print "   querying\n";
    my $sth = $dbh->prepare($query);
    $sth->execute() or die $DBI::errstr;

    #print out each CUI, PRED, CUI, PMID info to file
    print "   processing\n";
    my %matrix = ();
    my @row;
    my $vals;
    while(@row = $sth->fetchrow_array()){
	#$cui1 = $row[0];
	#$predicte = $row[2];
	#$cui2 = $row[1];
	#$pmid = $row[3];
	#create the key, a cui predicate, cui triple
	$vals = "$row[0]\t$row[1]\t$row[2]\t$row[3]";

	#Filter out malformed triplets extracted from the DB
	#   ...I don't know why some are malformed, so I just ignore
	if ($vals !~ /C\d{7}\t[^\t]+\tC\d{7}\t\d+\b/) {
	    next;
	}

	print OUT "$vals\n"
    }
    $dbh->disconnect();
    print "   done\n";
}
