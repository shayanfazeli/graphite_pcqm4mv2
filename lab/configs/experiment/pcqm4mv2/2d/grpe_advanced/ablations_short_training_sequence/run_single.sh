#!/bin/bash

export CUBLAS_WORKSPACE_CONFIG=:16:8;
export PYTHONPATH="$PYTHONPATH:/home/shayan/phoenix/graphite";
python3 ./graphite/bin/graphite_train.py ./lab/configs/experiment/pcqm4mv2/2d/grpe_advanced/ablations_short_training_sequence/${1}.py \
--seed=1994 \
--name="${1}" \
--project="grpe_ablations" \
--gpu=${2} \
--logdir="/data/pcqm4mv2_datahub/warehouse/graphite/pcqm4mv2/grpe_ablations_short_training_sequence/2d/" \
--clean