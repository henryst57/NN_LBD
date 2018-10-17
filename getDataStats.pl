#!usr/bin/perl

# Gets stats about the datasets (true, false, and known)
# These stats include:
#    count true, count false, count known
#    counts true, false, known for each relation type
#    average and standard deviation predications for each CUI
#    average and standard deviation predications for each predication type
#
# Input: 
#    true, false, and known datasets of the format:
#
# Output:
#    prints statistics to screen (or file?)
#


#TODO - implement me, see utlities.pm for loading data

#######################################################################################################
#                                                                                                     #
#    Main                                                                                             #
#                                                                                                     #
#######################################################################################################

for my $arg ( @ARGV )
{
    my $data_hash_ref = Load_Data( $arg );
    my $result_hash   = Generate_Results( $data_hash_ref );
    my $result        = PrintData( $arg, $result_hash );
}

print "~Fin\n";


#######################################################################################################
#                                                                                                     #
#    Support Functions                                                                                #
#                                                                                                     #
#######################################################################################################

sub Load_Data
{
    my $file_path = shift;
    
    my %data_hash = ();
    
    # Check(s)
    print "Load_Data() - Error: File Path Not Defined\n" if !defined( $file_path );
    print "Load_Data() - Error: File \"$file_path\" Does Not Exist\n" if defined( $file_path ) && !( -e $file_path );
    print "Load_Data() - Error: File \"$file_path\" Contains No Data\n" if defined( $file_path ) && ( -z $file_path );
    return undef if !defined( $file_path ) || defined( $file_path ) && ( !( -e $file_path ) || ( -z $file_path ) );
    
    my $result = 0;
    
    print "Loading File: \"$file_path\"\n";
    
    open( IN, "$file_path" ) or $result = -1;
    
    # Check(s)
    if( $result == -1 )
    {
        print "Load_Data() - Error Opening File: \"$file_path\"\n";
        return undef;
    }
    
    my $count = 0;
    
    # Read The Data Into A Hash
    while( my $line = <IN> )
    {
        $line = uc( $line );
        my @data = split( ',', $line );
        
        my $subject_CUI  = shift @data;
        my $predicate    = "";
        my @subject_data = split( ' ', $data[-1] );
        
        $predicate = shift( @subject_data );
        
        my $predicate_hash_ref = undef;
        my $object_hash_ref    = undef;
        
        # Check If Predicate Hash Already Exists
        $predicate_hash_ref = $data_hash{ "$subject_CUI" };
        
        for my $object_CUI ( @subject_data )
        {
            if( keys %{ $predicate_hash_ref } == 0 )
            {
                my %predicate_hash = ();
                my %object_hash    = ();
                $object_hash{ "$object_CUI" }   = 1;
                $predicate_hash{ "$predicate" } = \%object_hash;
                $data_hash{ "$subject_CUI" }    = \%predicate_hash;
                $predicate_hash_ref             = \%predicate_hash;
                $count++;
            }
            else
            {
                # Predicate Already Exists In Predicate Hash
                if( exists( $predicate_hash_ref->{ "$predicate" } ) )
                {
                    $object_hash_ref = $predicate_hash_ref->{ "$predicate" };
                    $object_hash_ref->{ $object_CUI } += 1;
                }
                elsif( !exists( $predicate_hash_ref->{ "$predicate" } ) )
                {
                    my %object_hash    = ();
                    $object_hash{ $object_CUI } = 1;
                    $predicate_hash_ref->{ "$predicate" } = \%object_hash;
                }
                
                $count++;
            }
        }
    }
    print "File: \"$file_path\" Loaded - $count Elements\n";
    
    close( IN );
    return \%data_hash;
}

sub Generate_Results
{
    my $data_hash_ref = shift;
    
    # Check(s)
    print "Error: Data Hash Length Equals Zero\n" if ( scalar keys %{ $data_hash_ref } == 0 );
    return undef                                  if ( scalar keys %{ $data_hash_ref } == 0 );
    
    my %stats_hash                             = ();
    my %predicates_per_subject_cui_hash        = ();
    my %object_cuis_per_subject_cui_hash       = ();
    my %object_cuis_per_predicate_hash         = ();
    my %total_predicates_per_subject_cui_hash  = ();
    my %mean_object_cuis_per_subject_cui_hash  = ();    # Mean Object CUI Values Per Subject CUI
    my %var_object_cuis_per_subject_cui_hash   = ();    # Variance Object CUI Values Per Subject CUI
    my %stdev_object_cuis_per_subject_cui_hash = ();    # Standard Deviation Object CUI Values Per Subject CUI
    my %predicate_object_cui_hash              = ();
    my %mean_object_cuis_per_predicate_hash    = ();
    my %var_object_cuis_per_predicate_hash     = ();
    my %stdev_object_cuis_per_predicate_hash   = ();
    my %subject_unique_cuis                    = ();
    my %unique_predicates                      = ();
    my %object_unique_cuis                     = ();
    my $total_number_of_elements               = 0;
    my $total_number_of_triplets               = 0;
    my $total_subject_cuis                     = 0;
    my $total_predicates                       = 0;
    my $total_object_cui_count                 = 0;
    my $total_unique_subject_cuis              = 0;
    my $total_unique_predicates                = 0;
    my $total_unique_object_cuis               = 0;
    my $average_predicates_per_cui             = 0;
    my $average_objects_per_cui                = 0;
    my $average_objects_per_predicate          = 0;
    
    # Parse Hash Data
    my @subject_cui_keys = sort keys %{ $data_hash_ref };
    
    for my $subject_cui ( @subject_cui_keys )
    {
        my $predicate_hash = $data_hash_ref->{ $subject_cui };
        my @predicate_keys = sort keys %{ $predicate_hash };
        
        $total_predicates_per_subject_cui_hash{ $subject_cui } = ( scalar @predicate_keys );   # Total Number of Predicates Per Subject CUI
        
        for my $predicate ( @predicate_keys )
        {
            my $object_hash     = $predicate_hash->{ $predicate };
            my @object_cui_keys = sort keys %{ $object_hash };
            
            for my $object_cui ( @object_cui_keys )
            {
                $total_object_cui_count += 1;                                                                        # Add One For Total Object CUIs
                $object_unique_cuis{ $object_cui } += 1;                                                        # Add Object CUI Only Once (Unique)
                $total_number_of_elements += $object_hash->{ $object_cui };                                     # Add Value For Object CUIs
                $object_cuis_per_predicate_hash{ $predicate } += ( $object_hash->{ $object_cui } );             # Add Value For Object CUIs Per Predicate
                $object_cuis_per_subject_cui_hash{ $subject_cui } += ( $object_hash->{ $object_cui } );         # Add Value For Object CUIs Per Subject CUI
                
                # Add Object CUIs For Each Predicate To Predicate Hash
                if( !exists( $predicate_object_cui_hash{ $predicate } ) )
                {
                    my %object_cui_hash = ();
                    $object_cui_hash{ $object_cui } = 1;
                    $predicate_object_cui_hash{ $predicate } = \%object_cui_hash;
                }
                else
                {
                    $predicate_object_cui_hash{ $predicate }->{ $object_cui } += 1;
                }
            }
            
            $unique_predicates{ $predicate } += 1;                                              # Add Predicate Only Once (Unique)
        }
        
        $subject_unique_cuis{ $subject_cui } += 1;                                              # Add Subject CUI Only Once (Unique)
        $total_predicates += scalar @predicate_keys;                                            # Add Value For Total Predicates
        $total_subject_cuis += scalar @predicate_keys;                                          # Add Value For Total Subject CUI
        $total_number_of_elements += scalar @predicate_keys;                                    # Add Value For Total Predicates
        $predicates_per_subject_cui_hash{ $subject_cui } += ( scalar @predicate_keys );         # Add Number Of Predicates Per Subject
        $total_number_of_elements++;                                                            # Add One For Subject CUI
    }
    
    for my $object_cui ( keys %predicates_per_subject_cui_hash )
    {
        $average_predicates_per_cui += $predicates_per_subject_cui_hash{ $object_cui };
    }
    
    for my $object_cui ( keys %object_cuis_per_subject_cui_hash )
    {
        $average_objects_per_cui += $object_cuis_per_subject_cui_hash{ $object_cui };
    }
    
    for my $object_cui ( keys %object_cuis_per_predicate_hash )
    {
        $average_objects_per_predicate += $object_cuis_per_predicate_hash{ $object_cui };
    }
    
    $average_predicates_per_cui    /= scalar keys %predicates_per_subject_cui_hash;
    $average_objects_per_cui       /= scalar keys %object_cuis_per_subject_cui_hash;
    $average_objects_per_predicate /= scalar keys %object_cuis_per_predicate_hash;
    
    $average_predicates_per_cui    = sprintf( "%.6f", $average_predicates_per_cui );           # Round To Nearest Sixth Decimal Place
    $average_objects_per_cui       = sprintf( "%.6f", $average_objects_per_cui );              # Round To Nearest Sixth Decimal Place
    $average_objects_per_predicate = sprintf( "%.6f", $average_objects_per_predicate );        # Round To Nearest Sixth Decimal Place
    
    $total_number_of_triplets  = $total_object_cui_count;                                      # Total # Of Triplets = Total # Of Subject CUIs
    $total_unique_subject_cuis = scalar keys %subject_unique_cuis;
    $total_unique_predicates   = scalar keys %unique_predicates;
    $total_unique_object_cuis  = scalar keys %object_unique_cuis;
    
    # Calculate Standard Deviation Per Subject CUI
    for my $subject_cui ( sort keys %total_predicates_per_subject_cui_hash )
    {
        # Calculate Subject Mean From Object CUIs Per Subject CUI And Square The Result
        $mean_object_cuis_per_subject_cui_hash{ $subject_cui } = sprintf( "%.6f", $object_cuis_per_subject_cui_hash{ $subject_cui } / $total_predicates_per_subject_cui_hash{ $subject_cui } );
        
        my $predicate_hash_ref = $data_hash_ref->{ $subject_cui };
        
        for my $predicate ( sort keys %{ $predicate_hash_ref } )
        {
            my $object_hash_ref = $predicate_hash_ref->{ $predicate };
            my $object_cui_total = 0;
            
            for my $object_cui ( sort keys %{ $object_hash_ref } )
            {
                $object_cui_total += $object_hash_ref->{ $object_cui };
            }
            
            $var_object_cuis_per_subject_cui_hash{ $subject_cui } += ( $object_cui_total - $mean_object_cuis_per_subject_cui_hash{ $subject_cui } ) ** 2;
        }
        
        # Calculate Mean Of Squared Objects CUI Results (Variance)
        $var_object_cuis_per_subject_cui_hash{ $subject_cui } /= ( $total_predicates_per_subject_cui_hash{ $subject_cui } );
        $var_object_cuis_per_subject_cui_hash{ $subject_cui } = sprintf( "%.6f", $var_object_cuis_per_subject_cui_hash{ $subject_cui } );
        
        # Calculate Standard Deviation Of Squared Objects CUI Results
        $stdev_object_cuis_per_subject_cui_hash{ $subject_cui } = sprintf( "%.6f", sqrt( $var_object_cuis_per_subject_cui_hash{ $subject_cui } ) );
    }
    
    # Calculate Standard Deviation Per Predicate
    for my $predicate ( sort keys %predicate_object_cui_hash )
    {
        my $object_cui_hash = $predicate_object_cui_hash{ $predicate };
        my $object_cui_count = 0;
        
        for my $object_cui ( sort keys %{ $object_cui_hash } )
        {
            $object_cui_count += $object_cui_hash->{ $object_cui };
        }
        
        # Calculate Subject Mean From Object CUIs Per Subject CUI And Square The Result
        $mean_object_cuis_per_predicate_hash{ $predicate } = sprintf( "%.6f", $object_cui_count / ( scalar %{ $object_cui_hash } ) );
        
        # Calculate Mean Of Squared Objects CUI Results (Variance)
        for my $object_cui ( sort keys %{ $object_cui_hash } )
        {
            $var_object_cuis_per_predicate_hash{ $predicate } += ( $object_cui_hash->{ $object_cui } - $mean_object_cuis_per_predicate_hash{ $predicate } ) ** 2;
        }
        
        $var_object_cuis_per_predicate_hash{ $predicate } /= ( scalar %{ $object_cui_hash } );
        $var_object_cuis_per_predicate_hash{ $predicate } = sprintf( "%.6f", $var_object_cuis_per_predicate_hash{ $predicate } );
        
        # Calculate Standard Deviation Of Squared Objects CUI Results
        $stdev_object_cuis_per_predicate_hash{ $predicate } = sprintf( "%.6f", sqrt( $var_object_cuis_per_predicate_hash{ $predicate } ) );
    }
    
    # Add Data To Statistics Hash
    $stats_hash{ "Total Number Of Elements"             } = $total_number_of_elements;
    $stats_hash{ "Total Number Of Triplets"             } = $total_number_of_triplets;
    $stats_hash{ "Total Subject CUIs"                   } = $total_subject_cuis;
    $stats_hash{ "Total Predicates"                     } = $total_predicates;
    $stats_hash{ "Total Object CUIs"                    } = $total_object_cui_count;
    $stats_hash{ "Total Unique Subject CUIs"            } = $total_unique_subject_cuis;
    $stats_hash{ "Total Unique Predicates"              } = $total_unique_predicates;
    $stats_hash{ "Total Unique Object CUIs"             } = $total_unique_object_cuis;
    $stats_hash{ "Average Predicates Per CUI"           } = $average_predicates_per_cui;
    $stats_hash{ "Average Objects Per CUI"              } = $average_objects_per_cui;
    $stats_hash{ "Average Objects Per Predicate"        } = $average_objects_per_predicate;
    $stats_hash{ "Total Predicates Per Subject CUI"     } = \%total_predicates_per_subject_cui_hash;
    $stats_hash{ "Total Object CUIs Per Subject CUI"    } = \%object_cuis_per_subject_cui_hash;
    $stats_hash{ "Mean Object CUIs Per Subject CUI"     } = \%mean_object_cuis_per_subject_cui_hash;
    $stats_hash{ "Variance Object CUIs Per Subject CUI" } = \%var_object_cuis_per_subject_cui_hash;
    $stats_hash{ "St. Dev Object CUIs Per Subject CUI"  } = \%stdev_object_cuis_per_subject_cui_hash;
    $stats_hash{ "Mean Object CUIs Per Predicate"       } = \%mean_object_cuis_per_predicate_hash;
    $stats_hash{ "Variance Object CUIs Per Predicate"   } = \%var_object_cuis_per_predicate_hash;
    $stats_hash{ "St. Dev Object CUIs Per Predicate"    } = \%stdev_object_cuis_per_predicate_hash;
    
    return \%stats_hash;
}

sub PrintData
{
    my $file_name       = shift;
    my $result_hash_ref = shift;
    
    # Check(s)
    print "PrintData() - Error: No Specified File Name\n"               if !defined( $file_name ) || $file_name eq "";
    print "PrintData() - Error: Result Hash Reference Is Not Defined\n" if !defined( $result_hash_ref );
    print "PrintData() - Error: Result Hash Contains No Data\n"         if ( scalar %{ $result_hash_ref } ) == 0;
    return -1 if !defined( $file_name ) || $file_name eq "" || !defined( $result_hash_ref ) || ( scalar %{ $result_hash_ref } ) == 0;
    
    my $result = 0;
    
    open( OUT, ">$file_name.results" ) or $result = -1;
    
    # Check(s)
    print "PrintData() - Error: Failed To Create Write File\n" if ( $result == -1 );
    return -1                                                  if ( $result == -1 );
    
    print OUT "File Name: \"$file_name\" - Results\n\n";
    
    # Print Variable Data
    for my $result ( sort keys %{ $result_hash_ref } )
    {
        if( ref $result_hash_ref->{ $result } ne ref {} )
        {
            print OUT $result . ": " . $result_hash_ref->{ $result } . "\n";
        }
    }
    
    # Print Hash Data
    for my $result ( sort keys %{ $result_hash_ref } )
    {
        if( ref $result_hash_ref->{ $result } eq ref {} )
        {
            my $hash_ref = $result_hash_ref->{ $result };
            
            if( $result eq "Total Predicates Per Subject CUI" )
            {
                print OUT "\n# Of Predicates Per Subject CUI\n";
                print OUT "-------------------------------\n";
            }
            elsif( $result eq "Total Object CUIs Per Subject CUI" )
            {
                print OUT "\n# Of Object CUIs Per Subject CUI\n";
                print OUT "--------------------------------\n";
            }
            elsif( $result eq "Mean Object CUIs Per Subject CUI" )
            {
                print OUT "\nMean Object CUIs Per Subject CUI\n";
                print OUT "--------------------------------\n";
            }
            elsif( $result eq "Variance Object CUIs Per Subject CUI" )
            {
                print OUT "\nVariance Object CUIs Per Subject CUI\n";
                print OUT "------------------------------------\n";
            }
            elsif( $result eq "St. Dev Object CUIs Per Subject CUI" )
            {
                print OUT "\nSt. Dev Object CUIs Per Subject CUI\n";
                print OUT "-----------------------------------\n";
            }
            elsif( $result eq "Mean Object CUIs Per Predicate" )
            {
                print OUT "\nMean Object CUIs Per Predicate CUI\n";
                print OUT "----------------------------------\n";
            }
            elsif( $result eq "Variance Object CUIs Per Predicate" )
            {
                print OUT "\nVariance Object CUIs Per Predicate\n";
                print OUT "----------------------------------\n";
            }
            elsif( $result eq "St. Dev Object CUIs Per Predicate" )
            {
                print OUT "\nSt. Dev Object CUIs Per Predicate\n";
                print OUT "---------------------------------\n";
            }
            
            for my $key ( sort keys %{ $hash_ref } )
            {
                print OUT "$key: " . $hash_ref->{ $key } . "\n";
            }
            
            print OUT "\n\n";
        }
    }
    
    return $result;
}