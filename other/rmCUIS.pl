#!/bin/perl
use strict;
use warnings;

#print "What file do you want to remove cuis from?: ";
#chomp(my $filename = <STDIN>);
my $filename = "smdb.allpredicate.s1.mc0.bin";
my $FILE;
open($FILE, "<", $filename) || die "Cannot open file $1";
my @lines = <$FILE>;
foreach my $line (@lines){
	chomp($line);
	if($line =~ m/^(C)[0-9]+/g){
		next;
	}
	print "$line\n";
}
