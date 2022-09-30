# Exp1

## Single process

```bash
export PYTHONPATH="$PYTHONPATH:/home/shayan/phoenix/graphite";
python3 ./graphite/bin/graphite_train.py ./lab/configs/experiment/pcqm4mv2/2d/grpe/exp1.py \
--seed=1819 \
--name="exp1" \
--project="graphite_1" \
--gpu=0 \
--logdir="/home/shayan/warehouse/graphite/pcqm4mv2/2d/grpe/exp1"
```

## Distributed

```bash
#!/bin/bash
export CUBLAS_WORKSPACE_CONFIG=:16:8;
export PYTHONPATH="$PYTHONPATH:/home/shayan/phoenix/graphite";
torchrun \
--standalone \
--nnodes=1 \
--nproc_per_node=4 \
./graphite/bin/graphite_train.py ./lab/configs/experiment/pcqm4mv2/2d/grpe/exp1.py \
--seed=1819 \
--name="exp5" \
--project="graphite_2" \
--gpu=0 \
--logdir="/home/shayan/warehouse/graphite/pcqm4mv2/2d/grpe/exp1" \
--clean
```