#!/bin/bash
for split in 0 1 2 3 4
do
    /nfs/stak/users/noelt/Documents/Project/noelt_masters_project/envs/project/bin/python3 /nfs/stak/users/noelt/Documents/Project/noelt_masters_project/experiment.py --loss="margin" --detection_type="LS" --split=$split --top_k=1 --seed=$split 
done

