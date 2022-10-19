# Graph-modeling with Transformers

## Introduction
* [Quickstart](https://github.com/shayanfazeli/graphite_pcqm4mv2/tree/master/getting_started.md)
* [Sample Experiment](https://github.com/shayanfazeli/graphite_pcqm4mv2/tree/master/lab/configs/experiment/pcqm4mv2/2d/grpe)
* [Datasets and Transforms](https://github.com/shayanfazeli/graphite_pcqm4mv2/tree/master/graphite/data/pcqm4mv2/pyg)

## In Progress Experiments
* GRPE Large with modifications (cont'd)
```bash
./lab/configs/experiment/pcqm4mv2/2d/grpe_advanced/run.sh exp9  # lambda4
```

* CoATGIN_base (10m) on 3dbond
```bash
./lab/configs/experiment/pcqm4mv2/3d/coatgin/run.sh 3d_bond_exp1 # lambda3
```

* CoATGIN_base (10m) on 3dbond + KPGT Losses
```bash
./lab/configs/experiment/pcqm4mv2/3d/coatgin/run.sh 3d_bond_kpgt_exp1 # lambda5
```


## Additional Resources
### Docker
Please find a step-by-step guide on how to set up a proper docker container to run `graphite` in at the following
link: [view](https://github.com/shayanfazeli/graphite_pcqm4mv2/blob/master/docker_info.md)

### Dataset stats
Please find more information in [this file](https://github.com/shayanfazeli/graphite_pcqm4mv2/blob/master/dataset_stats.md).

### Conformers
* 10 conformers per molecule with energy minimization: [[Download](https://drive.google.com/file/d/1xSNWO5sjGH5ZLHbeR8h9MIR7qXVmhrjd/view?usp=sharing)]
* generation command:
```python3
import datamol as dm
new_mol = dm.conformers.generate(
    mol, 
    align_conformers=True,
    n_confs=10,
    num_threads=16,
    minimize_energy=True,
    ignore_failure=True,
    energy_iterations=100,
)
```
