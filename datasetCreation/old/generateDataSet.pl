#Script to generate the datasets used in NN_LBD
`perl semmedDB2Mat_withPMIDs.pl`;
`perl applyDateRange.pl`;
`perl getGenericCuis.pl`;
`perl filterDataSet.pl`;
`perl convertToCUIPredicateAll.pl`;
#`perl createTestAndTrain.pl`;
