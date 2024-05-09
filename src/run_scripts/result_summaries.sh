#!/bin/bash
# For collecting results from a cluster. This packages them into a set of tar files.
cat demo_batch/*/results.csv > results_sweep.csv
tar -cf sweep_end_vtp.tar demo_batch/*/*4099.vtp 
tar -cf sweep_end_csv.tar demo_batch/*/*4099.csv 
tar -cf sweep_genome.tar demo_batch/*/genome.csv
tar -cf sweep_Specfile.tar demo_batch/*/SpecificationFile.txt
