#script to clean up the output data, making it easier to plot in excel
use strict;
use warnings;

#User Input
my $inFile = 'trainingOut_2000BatchSize_withWeights';
my $outFile = "cleaned_$inFile";
#&_cleanOutput($inFile, $outFile);

$inFile = 'output_replicateOld';
$outFile = $inFile;
#$outFile =~ s/nohup/trainingOut_batch/;
$outFile .= '_cleaned';
&_cleanOutput($inFile, $outFile);





###################################
#    Begin Code
##################################
sub _cleanOutput {
 
    #process the file, and output the results as its being processed
    #open each file
    open IN, $inFile or die ("ERROR: unable to open inFile: $inFile\n");
    open OUT, ">$outFile" or die ("ERROR: unable to open outFile: $outFile\n");

    #print the header info to the output file:
    print OUT "Batch Number\tETA\tLoss\tAccuracy\tPrecision\tRecall\tMatthews Correlation\n";

    #process each line of the file
    my $batchNumber = 1;
    while (my $line = <IN>) {

	#only read in lines reporting the progres
	if ($line =~ /(\d+)\/\d+ .+ - ETA: (\d+:\d{2}) - loss: (\d+\.\d+e?[+-]?\d*) - acc: (\d+\.\d+e?[+-]?\d*) - Precision: (\d+\.\d+e?[+-]?\d*) - Recall: (\d+\.\d+e?[+-]?\d*) - Matthews_Correlation: (-?\d+\.\d+e?[+-]?\d*)/) {
	    
	    #grab the values
	    #my $batchNumber = $1;
	    my $eta = $2;
	    my $loss = $3;
	    my $acc = $4;
	    my $precision = $5;
	    my $recall = $6;
	    my $mcc = $7;

	    #keep your own count of batch number
	    # because it resets at the end of each epoch
	    $batchNumber++;

	    #output the values in a tab seperated format
	    print OUT "$batchNumber\t$eta\t$loss\t$acc\t$precision\t$recall\t$mcc\n";
	}   
    }
    close IN;

    print "Done!\n";
}
