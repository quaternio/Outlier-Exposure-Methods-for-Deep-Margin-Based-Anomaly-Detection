#!/bin/bash
for split in 0 1 2 3 4
do
    for loss in "margin"
    do
            /nfs/stak/users/noelt/Documents/Project/noelt_masters_project/envs/project/bin/python3 /nfs/stak/users/noelt/Documents/Project/noelt_masters_project/experiment.py --loss=$loss --detection_type="KS" --split=$split --seed=$split 
    done
done

