# Line Graphs and CoAtGIN

Sample run command

```bash
export PYTHONPATH="$PYTHONPATH:/home/shayan/phoenix/graphite"; # code repo
python3 ./graphite/bin/graphite_train.py ./lab/configs/experiment/pcqm4mv2/2d/coatgin/2d_linegraph_kpgt.py \
--seed=1819 \
--name="exp2" \
--project="graphite_coatgin" \
--gpu=0 \
--logdir="/home/shayan/warehouse/graphite/pcqm4mv2/2d/coatgin/exp2" \
--wandb_offline \
--wandb_apikey="382ea80a0befa8bf3f3616bc9d9b99fc46ee43bf"
```
