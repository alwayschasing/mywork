#!/bin/bash
./mf-train -l2 0.2 -k 10 -t 500 -r 0.05 -s 40 /home/lrh/graduation_project/data/CiaoDVD/finMFtr.txt ../MFModel
./mf-predict /home/lrh/graduation_project/data/CiaoDVD/finMFte.txt ../MFModel /home/lrh/graduation_project/data/CiaoDVD/rating_res
