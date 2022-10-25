#!/bin/bash

export CUBLAS_WORKSPACE_CONFIG=:16:8;
export PYTHONPATH="$PYTHONPATH:/workspace";
torchrun \
--standalone \
--nnodes=1 \
--nproc_per_node=4 \
./graphite/bin/graphite_train.py ./lab/configs/experiment/pcqm4mv2/2d/grpe_advanced/jp_splits/grpe_large+ndce/${1}.py \
--seed=4091 \
--name="${1}" \
--project="graphite_grpe_docker_folds" \
--gpu=0 \
--logdir="/data/pcqm4mv2_datahub/warehouse/graphite/pcqm4mv2/2d/graphite_grpe_docker_folds" \
--clean