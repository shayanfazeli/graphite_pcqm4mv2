# GRPE Advanced

The customized version of GRPE with additional options

## Single process
```bash
export PYTHONPATH="$PYTHONPATH:/home/shayan/phoenix/graphite";
python3 ./graphite/bin/graphite_train.py ./lab/configs/experiment/pcqm4mv2/2d/grpe_advanced/base.py \
--seed=1819 \
--name="exp1" \
--project="customized_grpe" \
--gpu=0 \
--logdir="/home/shayan/warehouse/graphite/pcqm4mv2/2d/grpe_advanced" \
#--wandb_offline \
#--wandb_apikey='YOURKEY' \
--clean
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
./graphite/bin/graphite_train.py ./lab/configs/experiment/pcqm4mv2/2d/grpe_advanced/base.py \
--seed=1819 \
--name="exp1" \
--project="customized_grpe" \
--gpu=0 \
--logdir="/home/shayan/warehouse/graphite/pcqm4mv2/2d/grpe_advanced" \
--clean
```


## Exp: twice path length

```bash
#!/bin/bash
export CUBLAS_WORKSPACE_CONFIG=:16:8;
export PYTHONPATH="$PYTHONPATH:/home/shayan/phoenix/graphite";
torchrun \
--standalone \
--nnodes=1 \
--nproc_per_node=4 \
./graphite/bin/graphite_train.py ./lab/configs/experiment/pcqm4mv2/2d/grpe_advanced/longer_path_2.py \
--seed=1819 \
--name="exp3" \
--project="customized_grpe" \
--gpu=0 \
--logdir="/home/shayan/warehouse/graphite/pcqm4mv2/2d/grpe_advanced" \
--clean
```


```bash
#!/bin/bash
export CUBLAS_WORKSPACE_CONFIG=:16:8;
export PYTHONPATH="$PYTHONPATH:/home/shayan/phoenix/graphite";
torchrun \
--standalone \
--nnodes=1 \
--nproc_per_node=4 \
./graphite/bin/graphite_train.py ./lab/configs/experiment/pcqm4mv2/2d/grpe_advanced/exp5.py \
--seed=1819 \
--name="exp5" \
--project="customized_grpe" \
--gpu=0 \
--logdir="/home/shayan/warehouse/graphite/pcqm4mv2/2d/grpe_advanced" \
--clean
```