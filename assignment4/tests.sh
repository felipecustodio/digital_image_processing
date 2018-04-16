#!/bin/bash
        rm results.txt
        for i in `seq 1 9`;
        do
                printf "\nRunning test $i\n"
                printf "\nRunning test $i: " >> results.txt
                python assignment4.py < tests/$i.in >> results.txt
        done
