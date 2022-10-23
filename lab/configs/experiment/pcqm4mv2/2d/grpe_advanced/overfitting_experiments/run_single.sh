#!/bin/bash

export CUBLAS_WORKSPACE_CONFIG=:16:8;
export PYTHONPATH="$PYTHONPATH:/home/shayan/phoenix/graphite";
python3 ./graphite/bin/graphite_train.py ./lab/configs/experiment/pcqm4mv2/2d/grpe_advanced/overfitting_experiments/${1}.py \
--seed=2023 \
--name="${1}" \
--project="graphite_grpe_advanced" \
--gpu=${2} \
--logdir="/data/pcqm4mv2_datahub/warehouse/graphite/pcqm4mv2/overfitting_experiments/2d/graphite_grpe_advanced" \
--clean