#!/bin/bash
# Change folder selection string as needed.
for file in $HOME/Programming/Cuda/Morphogenesis/data/demo_batch/demo_intstiff*.*00/
do
    echo $file
    make_demo2 $file $file
done
