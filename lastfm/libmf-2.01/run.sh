#!/bin/bash
./mf-train  -l2 0.1 -k 10 -t 100 -r 0.01 -s 40 /home/lrh/graduation_project/data/lastmf/finMFtr.txt ../MFModel
./mf-predict /home/lrh/graduation_project/data/lastmf/finMFte.txt ../MFModel /home/lrh/graduation_project/data/lastmf/rating_res
