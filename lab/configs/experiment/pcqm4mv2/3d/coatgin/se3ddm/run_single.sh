#!/bin/bash

export CUBLAS_WORKSPACE_CONFIG=:16:8;
export PYTHONPATH="$PYTHONPATH:/workspace";
python3 ./graphite/bin/graphite_train.py ./lab/configs/experiment/pcqm4mv2/3d/coatgin/se3ddm/${1}.py \
--seed=1999 \
--name="${1}" \
--project="coatgin_ablations" \
--gpu=${2} \
--logdir="/data/pcqm4mv2_datahub/warehouse/graphite/pcqm4mv2/3d/coatgin/se3ddm" \
--clean \
--wandb_offline \
--wandb_apikey="debug"