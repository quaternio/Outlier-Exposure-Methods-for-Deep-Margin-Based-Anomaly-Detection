#!/bin/bash
for split in 1 2 3 4
do
   /nfs/stak/users/noelt/Documents/Project/noelt_masters_project/envs/project/bin/python3 /nfs/stak/users/noelt/Documents/Project/noelt_masters_project/experiment_single.py -b --margin_baseline --split=$split --seed=$split

done

