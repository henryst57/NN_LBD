#Queries SemMedDB to return and output a list of generic CUIs
# these are the 'non-novel' cuis that are too general to 
# provide useful information
# output format is a line seperated file of CUIs (C0000000)
use DBI;
use strict;
use warnings;


my $outFile = '../../data/semmedDB/genericCuis';


#######################################
# Begin Code
########################################

#set up the database
my $database = "semmedVER30";
my $hostname = "192.168.24.89";
my $userid = "henryst";
my $password = "OhFaht3eique";
my %thingy = ();
my $dsn = "DBI:mysql:database=$database;host=$hostname;";
my $dbh = DBI->connect($dsn, $userid, $password, \%thingy) or die $DBI::errstr;
$dbh->{InactiveDestroy} = 1; #allows forking of threads containing this DB connect

#set the query to get all generic concept cuis
my $query =  'SELECT CUI FROM semmedVER30.GENERIC_CONCEPT';
print "query = $query\n";

#query the db
print "   querying\n";
my $sth = $dbh->prepare($query);
$sth->execute() or die $DBI::errstr;


#output the results
print "   outputting results\n";
open OUT, ">$outFile" or die ("Error: unable to open $outFile\n");
while(my @row = $sth->fetchrow_array()){
    print OUT "@row\n";
}
print "Done!\n";
