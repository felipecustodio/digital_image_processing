#!/bin/bash
        for i in `seq 1 10`;
        do
                printf "\nRunning test $i: " >> results.txt
                python assignment2.py < tests/$i.in >> results.txt
        done
