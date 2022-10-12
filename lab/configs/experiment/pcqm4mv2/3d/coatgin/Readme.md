# Line Graphs and CoAtGIN

Sample run command

(currently there is the possibility of all-nan comenet features for molecules with no conformers: #todo: remove from training indices)

```bash
export PYTHONPATH="$PYTHONPATH:/home/shayan/phoenix/graphite"; # code repo
python3 ./graphite/bin/graphite_train.py ./lab/configs/experiment/pcqm4mv2/3d/coatgin/3d_bond_linegraph_kpgt.py \
--seed=1819 \
--name="exp2" \
--project="graphite_coatgin" \
--gpu=0 \
--logdir="/home/shayan/warehouse/graphite/pcqm4mv2/3d/coatgin/exps" \
--wandb_offline \
--wandb_apikey="382ea80a0befa8bf3f3616bc9d9b99fc46ee43bf"
```
