#!/bin/bash

cd $ROCM_PATH/.hip
./remove_origin.sh
cd -
hipcc -o test-hytsf test-hytsf.cpp -lhytsf -lsqueue -L.
