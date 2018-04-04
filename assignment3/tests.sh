#!/bin/bash
        for i in `seq 1 4`;
        do
                printf "\nRunning test $i: " >> results.txt
                python assignment3.py < tests/$i.in >> results.txt
        done
