#! /bin/bash

for i in {1..8}
do
    experiment_name="TCN_experiment_day_$i"
    python meteo_model/training/perform_TCN_experiments.py --n_days $i --experiment_name $experiment_name
done