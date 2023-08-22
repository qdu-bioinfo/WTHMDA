#!/bin/bash

python ../bin/get_microbe_feature.py --mis example/microbe_I_S.csv

python ../bin/get_disease_feature.py --dis example/disease_I_S.csv

python ../bin/get_edges.py --mlist example/microbe_list.xlsx

python ../bin/train_test.py --train example/train_edge.csv --test example/test_edge.csv --o ../result/ 