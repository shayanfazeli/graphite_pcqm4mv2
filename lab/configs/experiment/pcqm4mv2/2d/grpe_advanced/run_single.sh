#!/bin/bash

export CUBLAS_WORKSPACE_CONFIG=:16:8;
export PYTHONPATH="$PYTHONPATH:/home/shayan/phoenix/graphite";
python3 ./graphite/bin/graphite_train.py ./lab/configs/experiment/pcqm4mv2/2d/grpe_advanced/${1}.py \
--seed=2023 \
--name="${1}" \
--project="graphite_grpe_advanced" \
--gpu=${2} \
--logdir="/data/pcqm4mv2_datahub/warehouse/graphite/pcqm4mv2/2d/graphite_grpe_advanced" \
--clean \
--wandb_offline \
--wandb_apikey="382ea80a0befa8bf3f3616bc9d9b99fc46ee43bf"