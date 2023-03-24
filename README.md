test upload

![image](./analysis/statistical_graph.jpg)   
![image](./analysis/gif/compare/TCGA-49-6744-01Z-00-DX4.gif)   
![image](./analysis/gif/overlap/TCGA-49-6744-01Z-00-DX4.gif)   
   
```bash
python process_0_make_variable.py -n 0
```
   
```bash
python process_1_prepare_dataset.py --tsv variable_1_prepare_dataset.tsv
```
   
```bash
python process_2_CNN_training.py --tsv variable_2_CNN_training.tsv
```
   
```bash
python process_3_MIL_training.py --tsv variable_3_MIL_training.tsv
```
   
```bash
python process_4_analysis_results.py --tsv variable_4_analysis_results.tsv
```
   
```bash
python organization_analysis_results.py
```
   