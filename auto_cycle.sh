#!/bin/sh

for num in {0..9} 
do
	python process_0_make_variable.py -n $num
	python process_1_prepare_dataset.py --tsv variable_1_prepare_dataset.tsv
	python process_2_CNN_training.py --tsv variable_2_CNN_training.tsv
	python process_3_MIL_training.py --tsv variable_3_MIL_training.tsv
	python process_4_analysis_results.py --tsv variable_4_analysis_results.tsv
done
