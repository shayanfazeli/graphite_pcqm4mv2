# Graph-modeling with Transformers

* [Sample Experiment](https://github.com/shayanfazeli/graphite_pcqm4mv2/tree/master/lab/configs/experiment/pcqm4mv2/2d/grpe)
* [Datasets and Transforms](https://github.com/shayanfazeli/graphite_pcqm4mv2/tree/master/graphite/data/pcqm4mv2/pyg)


## Additional Resources

## Conformers
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


