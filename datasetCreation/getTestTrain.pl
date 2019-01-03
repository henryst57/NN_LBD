#Outputs a file containing: CUI PRED CUI PMID for each entry in the 
# mysql predicate database
use DBI;
use strict;
use warnings;


#add the desired semantic types to the list of acceptable predicates
my $preOutFile = 'known';
my $goldOutFile = 'true';

#CHEM
my @chemTypes = ('aapp', 'antb', 'bacs', 'bodm', 'chem', 'chvf', 'chvs', 'clnd', 'elii', 'enzy', 'hops', 'horm', 'imft', 'irda', 'inch', 'nnon', 'orch', 'phsu', 'rcpt', 'vita');

#DISO
my @disoTypes = ('acab', 'anab', 'comd', 'cgab', 'dsyn', 'emod', 'fndg', 'inpo', 'mobd', 'neop', 'patf', 'sosy');

#PREDICATE types
my @predTypes = ('ISA', 'LOCATION_OF','PART_OF','USES','CAUSES','PROCESS_OF','TREATS','DIAGNOSES','ASSOCIATED_WITH','COEXISTS_WITH','METHOD_OF','AFFECTS','INTERACTS_WITH','OCCURS_IN','PRECEDES','COMPLICATES','PREVENTS','ADMINISTERED_TO','DISRUPTS','MANIFESTATION_OF','compared_with','PREDISPOSES','AUGMENTS','higher_than','INHIBITS','lower_than','same_as','STIMULATES','CONVERTS_TO','than_as');

#Pre and Post cutoff years
my $startYear = 1975;
my $cutoffYear = 2010;

#get the frequency of cui pairs
&convert();


##########################################################
#              BEGIN CODE
##########################################################


#generates known and gold standard predicate object list datasets split by year
sub convert {
 
#####
    #set up the database
    my $database = "semmed";
=comment
    #my $database = "1809onward";
    #my $hostname = "192.168.24.89";
    my $userid = "root";
    my $password = "";
    my %thingy = ();
    #my $dsn = "DBI:mysql:database=$database;host=$hostname;";
    print STDERR "TEST 1\n";
    my $dsn = "DBI:mysql:$database";
    print STDERR "TEST 2\n";
    #$dbh = DBI->connect($dsn, $userid, $password, \%thingy) or die $DBI::errstr;
    #$dbh = DBI->connect($dsn, $userid, $password) or die $DBI::errstr;
    my $dbh = DBI->connect("DBI:mysql:1809onward", "root", '') or die $DBI::errstr;
    print STDERR "TEST 3\n";
    $dbh->{InactiveDestroy} = 1; #allows forking of threads containing this DB connect
=cut
    my $dsn = "DBI:mysql:$database;mysql_read_default_group=client;";
    my $dbh = DBI->connect($dsn);

#########################################################
    print "   Getting Predicates\n";
    #construct the chem types string
    my $chemTypesString = '(';
    foreach my $type (@chemTypes) {
	$chemTypesString .= " \'$type\',";
    } 
    chop $chemTypesString;
    $chemTypesString .= ")";

    #construct diso types string
    my $disoTypesString = '(';
    foreach my $type (@disoTypes) {
	$disoTypesString .= " \'$type\',";
    }
    chop $disoTypesString;
    $disoTypesString .= ")";

    #construct the predicate types string
    my $predTypesString = '(';
    foreach my $type (@predTypes) {
	$predTypesString .= " \'$type\',";
    }
    chop $predTypesString;
    $predTypesString .= ')';

#####
    #construct the query
    my $query = "SELECT SUBJECT_CUI, PREDICATE, OBJECT_CUI, PMID FROM $database.PREDICATION WHERE ((SUBJECT_SEMTYPE IN $chemTypesString AND OBJECT_SEMTYPE IN $disoTypesString) OR (SUBJECT_SEMTYPE IN $disoTypesString AND OBJECT_SEMTYPE IN $chemTypesString)) AND PREDICATE IN $predTypesString AND SUBJECT_NOVELTY = \'1\' AND OBJECT_NOVELTY = \'1\'";
#    print "$query\n";

#####
    #query the db
    print "      Querying\n";
    my $sth = $dbh->prepare($query);
    $sth->execute() or die $DBI::errstr;
  
    #Get each CUI\tPRED\tCUI and store in a hash by PMID
    print "      Processing\n";
    my %predicates = ();
    while(my @row = $sth->fetchrow_array()){
	#$cui1 = $row[0];
	#$predicte = $row[1];
	#$cui2 = $row[2];
	#$pmid = $row[3];
	my $pmid = $row[3];
        my $predicate = "$row[0]\t$row[1]\t$row[2]";


	#Filter out malformed triplets extracted from the DB
	#   ...I don't know why some are malformed, so I just ignore
	if ($predicate !~ /C\d{7}\t[^\t]+\tC\d{7}/) {
	    next;
	}
	
	$predicates{$pmid} = $predicate;
    }
    print "      Got Predicates\n";


####################################################
    
    #query the db
    print "   Getting Years\n";
    print "      Querying\n";
    $query = "SELECT PMID, PYEAR FROM $database.CITATIONS";
    $sth = $dbh->prepare($query);
    $sth->execute() or die $DBI::errstr;

    #Get each published year (pyear) and store in a hash by pmid
    print "      Processing\n";
    my %pYear = ();
    while(my @row = $sth->fetchrow_array()){
	#$pmid = $row[0];
	#$pyear = $row[1];
	my $pmid = $row[0];
        my $year = $row[1];

	#print "pmid, year = $pmid, $year";
	
	$pYear{$pmid} = $year;
    }
    $dbh->disconnect(); #close DB, your done with it
    print "      Got Years\n";

###
    #Divide dataset into pre and post cutoff sets
    print "   Formatting the Dataset\n";
    print "      Dividing\n";
    my %preCutoffPreds = ();
    my %postCutoffPreds = ();
    foreach my $pmid (keys %predicates) {
	my $pYear = $pYear{$pmid};
	if (defined $pYear) {
	    if ($pYear >= $startYear && $pYear < $cutoffYear) {
		$preCutoffPreds{$pmid} = $predicates{$pmid};
	    }
	    elsif ($pYear > $cutoffYear) {
		$postCutoffPreds{$pmid} = $predicates{$pmid}
	    };
	}
    }


####
    print "      Compressing\n";
    #Combine all subject predicates to a single value
    my %preCutoffList = ();
    foreach my $pmid (keys %preCutoffPreds) {
	my ($subject, $predicate, $object) = split( /\t/, $preCutoffPreds{$pmid});
	if(!defined $preCutoffList{"$subject\t$predicate"}) {
	    $preCutoffList{"$subject\t$predicate"} = $object;
	}
	else {
	    $preCutoffList{"$subject\t$predicate"} .= "\t$object";
	}
    }
    my %postCutoffList = ();
    foreach my $pmid (keys %postCutoffPreds) {
	my ($subject, $predicate, $object) = split( /\t/, $postCutoffPreds{$pmid});
	if(!defined $postCutoffList{"$subject\t$predicate"}) {
	    $postCutoffList{"$subject\t$predicate"} = $object;
	}
	else {
	    $postCutoffList{"$subject\t$predicate"} .= "\t$object";
	}
    }


####
    print "      Filtering\n";
    #remove all pre-cutoff predicates from the post cutoff dataset
    my %goldPredicates = ();
    foreach my $pair (keys %preCutoffList) {
	if (defined $postCutoffList{$pair}) {
	    #grab the objects list from pre and post cutoff
	    my @preObjects = split (/\t/, $preCutoffList{$pair});
	    my @postObjects = split (/\t/, $postCutoffList{$pair});
	    
	    #convert post cutoff objects to a hash
	    my %postObjectsHash = ();
	    foreach my $object (@postObjects) {
		$postObjectsHash{$object} = 1;
	    }

	    #remove any precutoff objects from the post cutoff hash
	    foreach my $object (@preObjects) {
		if (!defined $postObjectsHash{$object}) {
		    if (!defined $goldPredicates{$pair}) {
			$goldPredicates{$pair} = $object;
		    }
		    else {
			$goldPredicates{$pair} .= "\t$object";
		    }
		}
	    }
	}
    }
    
####
    print "   Outputting Results\n";
    open OUT, ">$preOutFile" or die ("ERROR: unable to open preOutFile: $preOutFile\n");
    foreach my $pair (keys %preCutoffList) {
	print OUT $pair;
	my @objects = split (/\t/, $preCutoffList{$pair});
	foreach my $object (@objects) {
	    print OUT "\t$object";
	}
	print OUT "\n";
    }
    close OUT;

    #output the gold standard file
    open OUT, ">$goldOutFile" or die ("ERROR: unable to open goldOutFile: $goldOutFile\n");
    foreach my $pair (keys %goldPredicates) {
	print OUT $pair;
	my @objects = split (/\t/, $goldPredicates{$pair});
	foreach my $object (@objects) {
	    print OUT "\t$object";
	}
	print OUT "\n";
    }
    close OUT;

    print "   Done!\n";
}
