#!/bin/bash
./mf-train -l2 0.1 -k 10 -t 1000 -r 0.01 -s 36 /home/lrh/graduation_project/data/ml-1m/mf/trainForMFlib ../MFModel
#./mf-predict /home/lrh/graduation_project/data/ml-1m/rnn_rec_res.csv ../MFModel /home/lrh/graduation_project/data/ml-1m/rating_res
