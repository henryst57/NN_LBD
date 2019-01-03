#Outputs a file containing: CUI PRED CUI PMID for each entry in the 
# mysql predicate database
use DBI;
use strict;
use warnings;


#add the desired semantic types to the list of acceptable predicates
my $preOutFile = 'known_newSmall';
my $goldOutFile = 'true_newSmall';

#CHEM
#my @chemTypes = ('aapp', 'antb', 'bacs', 'bodm', 'chem', 'chvf', 'chvs', 'clnd', 'elii', 'enzy', 'hops', 'horm', 'imft', 'irda', 'inch', 'nnon', 'orch', 'phsu', 'rcpt', 'vita');
my @chemTypes = ('clnd', 'phsu');

#DISO
#my @disoTypes = ('acab', 'anab', 'comd', 'cgab', 'dsyn', 'emod', 'fndg', 'inpo', 'mobd', 'neop', 'patf', 'sosy');
my @disoTypes = ('dsyn', 'sosy');

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
	my $pmid = $row[0];
        my $year = $row[1];

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
	my $year = $pYear{$pmid};
	if (defined $year) {
	    if ($year >= $startYear && $year < $cutoffYear) {
		$preCutoffPreds{$pmid} = $predicates{$pmid};
	    }
	    elsif ($year > $cutoffYear) {
		$postCutoffPreds{$pmid} = $predicates{$pmid};
	    }
	}
    }


####
    print "      Compressing\n";
    #Combine all subject predicates pairs to a key, where the value
    #  is a hash of objects for which it holds true
    # also collect pre-cutoff vocab
    # then remove out of vocab subjects and obects from post cutoff list
    my %preCutoffList = ();
    my %preCutoffVocab = ();
    foreach my $pmid (keys %preCutoffPreds) {
	my ($subject, $predicate, $object) = split( /\t/, $preCutoffPreds{$pmid});
	#add the object and create a new hash if needed
	if(!defined $preCutoffList{"$subject\t$predicate"}) {
	    my %newHash = ();
	    $preCutoffList{"$subject\t$predicate"} = \%newHash;
	}
	${$preCutoffList{"$subject\t$predicate"}}{$object} = 1;
	
	#update the vocab
	$preCutoffVocab{$subject} = 1;
	$preCutoffVocab{$object} = 1;
    }
    my %postCutoffList = ();
    foreach my $pmid (keys %postCutoffPreds) {
	my ($subject, $predicate, $object) = split( /\t/, $postCutoffPreds{$pmid});
	#add the predicate if both subject and object are in vocabulary
	if (defined $preCutoffVocab{$subject} && defined $preCutoffVocab{$object}) {
	    #add the object and create a new hash if needed
	    if(!defined $postCutoffList{"$subject\t$predicate"}) {
		my %newHash = ();
		$postCutoffList{"$subject\t$predicate"} = \%newHash;
	    }
	    ${$postCutoffList{"$subject\t$predicate"}}{$object} = 1;
	}
    }


####
    print "      Filtering\n";
    #remove all pre-cutoff predicates from the post-cutoff dataset
    my %goldPredicates = ();
    foreach my $pair (keys %postCutoffList) {
	#iterate over all objects of this subject-object pair
	#add post cutoff objects that are not in the pre-cutoff hash
	# to the list of gold standard objects
	foreach my $object (keys %{$postCutoffList{$pair}}) {
	    my $add = 0;
	    #check if the subject-predicate pair exists
	    if (!defined $preCutoffList{$pair}) {
		$add = 1;
	    }
	    #check if the subject-predicate-object triplet exists
	    elsif(!defined ${$preCutoffList{$pair}}{$object}) {
		$add = 1;
	    }

	    #add the pair if needed
	    if ($add > 0) {
		#initialize the list if needed
		if (!defined $goldPredicates{$pair}) {
		    my %newHash = ();
		    $goldPredicates{$pair} = \%newHash;
		}
		${$goldPredicates{$pair}}{$object} = 1;
	    }
	}
    }
    
####
    print "   Outputting Results\n";
    open OUT, ">$preOutFile" or die ("ERROR: unable to open preOutFile: $preOutFile\n");
    foreach my $pair (keys %preCutoffList) {
	print OUT $pair;
	foreach my $object (keys %{$preCutoffList{$pair}}) {
	    print OUT "\t$object";
	}
	print OUT "\n";
    }
    close OUT;

    #output the gold standard file
    open OUT, ">$goldOutFile" or die ("ERROR: unable to open goldOutFile: $goldOutFile\n");
    foreach my $pair (keys %goldPredicates) {
	print OUT $pair;
	foreach my $object (keys %{$goldPredicates{$pair}}) {
	    print OUT "\t$object";
	}
	print OUT "\n";
    }
    close OUT;

    print "   Done!\n";
}
