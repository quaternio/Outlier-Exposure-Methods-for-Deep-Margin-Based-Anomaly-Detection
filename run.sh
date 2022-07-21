#!/bin/bash
for split in 0 1 2 3 4
do
    # Baseline first with and without oe_test
    /nfs/stak/users/noelt/Documents/Project/noelt_masters_project/envs/project/bin/python3 /nfs/stak/users/noelt/Documents/Project/noelt_masters_project/experiment.py -b -t --split=$split --seed=$split --oe_test 

   /nfs/stak/users/noelt/Documents/Project/noelt_masters_project/envs/project/bin/python3 /nfs/stak/users/noelt/Documents/Project/noelt_masters_project/experiment.py -b -t --split=$split --seed=$split

    for loss in "CE" "margin"
    do
        for detection_type in "KS" "LS"
        do
            /nfs/stak/users/noelt/Documents/Project/noelt_masters_project/envs/project/bin/python3 /nfs/stak/users/noelt/Documents/Project/noelt_masters_project/experiment.py --loss=$loss --detection_type=$detection_type -t --split=$split --seed=$split --oe_test 
            /nfs/stak/users/noelt/Documents/Project/noelt_masters_project/envs/project/bin/python3 /nfs/stak/users/noelt/Documents/Project/noelt_masters_project/experiment.py --loss=$loss --detection_type=$detection_type -t --split=$split --seed=$split 
        done
    done
done

