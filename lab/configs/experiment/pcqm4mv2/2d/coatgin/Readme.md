# Line Graphs and CoAtGIN

Sample run command

```bash
export PYTHONPATH="$PYTHONPATH:/home/shayan/phoenix/graphite"; # code repo
python3 ./graphite/bin/graphite_train.py ./lab/configs/experiment/pcqm4mv2/2d/coatgin/2d_linegraph.py \
--seed=1686 \
--name="2d_linegraph_kpgt" \
--project="graphite_coatgin" \
--gpu=0 \
--logdir="/home/shayan/warehouse/graphite/pcqm4mv2/2d/coatgin/exps" \
#--wandb_offline \
#--wandb_apikey="yourkey"
```


```bash
torchrun \
--standalone \
--nnodes=1 \
--nproc_per_node=4 \
./graphite/bin/graphite_train.py ./lab/configs/experiment/pcqm4mv2/2d/coatgin/2d_linegraph_kpgt.py \
--seed=1819 \
--name="exp1" \
--project="graphite_coatgin" \
--gpu=0 \
--logdir="/home/shayan/warehouse/graphite/pcqm4mv2/2d/coatgin/exps"
```