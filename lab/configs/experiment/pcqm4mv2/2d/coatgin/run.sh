#!/bin/bash

export CUBLAS_WORKSPACE_CONFIG=:16:8;
export PYTHONPATH="$PYTHONPATH:/home/shayan/phoenix/graphite";
torchrun \
--standalone \
--nnodes=1 \
--nproc_per_node=4 \
./graphite/bin/graphite_train.py ./lab/configs/experiment/pcqm4mv2/2d/coatgin/${1}.py \
--seed=2022 \
--name="${1}" \
--project="graphite_coatgin" \
--gpu=0 \
--logdir="/data/warehouse/graphite/pcqm4mv2/2d/graphite_coatgin" \
--clean