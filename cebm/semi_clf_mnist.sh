#!/bin/sh

for i in 1 2 3 4 5
    do
python3 semi_clf.py --num_shots=1 --seed=$i --device=1 --dataset=mnist --data_dir=../../sebm_data/ --batch_size=100 --optimizer=Adam --lr=1e-4  --num_epochs=100 --activation=ReLU

python3 semi_clf.py --num_shots=10 --seed=$i --device=1 --dataset=mnist --data_dir=../../sebm_data/ --batch_size=100 --optimizer=Adam --lr=1e-4  --num_epochs=100 --activation=ReLU

python3 semi_clf.py --num_shots=100 --seed=$i --device=1 --dataset=mnist --data_dir=../../sebm_data/ --batch_size=100 --optimizer=Adam --lr=1e-4  --num_epochs=100 --activation=ReLU

python3 semi_clf.py --seed=$i --device=1 --dataset=mnist --data_dir=../../sebm_data/ --batch_size=100 --optimizer=Adam --lr=1e-4  --num_epochs=100 --activation=ReLU
    done