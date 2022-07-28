#!/bin/bash
for split in 0 1 2 3 4
do
   /nfs/stak/users/noelt/Documents/Project/noelt_masters_project/envs/project/bin/python3 /nfs/stak/users/noelt/Documents/Project/noelt_masters_project/experiment.py -b --split=$split --seed=$split

    for loss in "CE" "margin"
    do
        for detection_type in "KS" "LS"
        do
            /nfs/stak/users/noelt/Documents/Project/noelt_masters_project/envs/project/bin/python3 /nfs/stak/users/noelt/Documents/Project/noelt_masters_project/experiment.py --loss=$loss --detection_type=$detection_type --split=$split --seed=$split 
        done
    done
done

