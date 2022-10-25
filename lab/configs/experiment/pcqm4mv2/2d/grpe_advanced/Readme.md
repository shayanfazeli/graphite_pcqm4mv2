# Experiments
__Projects__: `graphite_grpe_advanced` and `graphite_grpe_docker`


## In progress...
### `exp11`



## To run:


## Previously...
### `exp10`
* same as `exp9` except with KPGT projected (3 layer mlp) on fingerprint and descriptor + attention pooling
* WandB Report: [[Link](https://wandb.ai/shayanfazeli/graphite_grpe_docker/reports/Untitled-Report--VmlldzoyODQ2NDY2?accessToken=e7rvk943n8tw7uaws89vokt5c60et386himsthizkv40n5a7idjxutalf1728aho)]
* Observations
  * Compared to exp9 which was with KPGT regularization, it does not seem to incur an improvement in the early training. In fact, it
  seems that the KPGT as regularization is consistently better than it.

### `exp9`
* Number of parameters: 117,110,017
* base: GRPE Large
* Upperbounding shortest path length type to `5`.
* Upperbounding node-degree in degree centrality encoding to `5`.
* Using ogb methods like `ogb.utils.features.get_atom_feature_dims()` to set vocabulary size
  * __Remark__: atom-feature vocabulary lengths were set based on dataset stats which was: `36, 3, 7, 7, 5, 4, 6, 2, 2`
    * In OGB it is `[119, 4, 12, 12, 10, 6, 6, 2, 2] `
    * Edge vocab is also `4, 3, 2`, and ogb is `5, 6, 2`
    * The values we use are based on dataset stats and the values that are actually observed in this dataset. This
    does NOT include the SDF file, and refers only to smiles-based mol objects.
* Longer (usual) training sequence of 400 epochs with effective batchsize of `4*300` instead of original `8*64`, and twice the learning rate.
* Observations:
  * Training is slow, clip gradient does not make much of a difference.
  * Had to early terminate as the full training takes too long.
* WandB Report: [[Link](https://wandb.ai/shayanfazeli/graphite_grpe_docker/reports/Exp9-with-and-without-clip-gradient--VmlldzoyODQ2NDU2?accessToken=s5qi6fqqhhkek18x14naioyn0ju7if8kxbhli0lqe7h5cjai19dj4sp6v1kayrkp)]



### `exp8`
* Number of parameters: 116,753,929
* base: GRPE Large
* Added toeplitz correction
* Upperbounding shortest path length type to `10` instead of `5`.
* No node-degree centrality encoding / no path encoding
* Smaller weight to KPGT loss (`1e-2`)
* larger weight decay (`5e-3`)
* Modified learning schedule: larger learning rate and shorter sequence
* Cosine annealing instead of polynomialLR
* Observations:
  * The shorter-training sequence still failed, and it appeared that removing the extra components that were present
  in exp7 led to slower fitting validation as well.
* WandB Report [[Link](https://wandb.ai/shayanfazeli/graphite_grpe_docker/reports/GRPE-Large-exp7-and-exp8--VmlldzoyODE5NDYw?accessToken=8c8t10ggo6dt0qz4daa510p068qrjs6az76bfa1ipssjd86sg91ytbtmvbbn6wjr)]



### `exp7`
* base: GRPE Large
* Added Graphormer path encoding, node encoding, and toeplitz corrections
* Modified learning schedule: larger learning rate and shorter sequence
* Observations:
  * So far on 60 epochs, validation mae is 0.090 and train on 0.054, so indication of an overfit.
  * Letting it finish another 40 epochs to see what happens.
* WandB Report: [[Link](https://wandb.ai/shayanfazeli/graphite_grpe_docker/reports/GRPE-Large-Graphormer-KPGT-Loss-for-Regularization--VmlldzoyODA3MTc1?accessToken=2vpp89vx63wevtily23b9xcmu5bcqw29rlx5n946nuc0vzbpepevbi36hvy7rca0)]


### `exp5`
* base: `exp1`
* Main Hyerparameters:
  * longer  training sequence  (400 epochs)
  * Optimization: identical to the original GRPE
  * The main goal is to run the original GRPE with higher effective batch-size and with the graphormer capabilities
```python
optimizer = dict(
    type='AdamW',
    args=dict(
        lr=2e-4,
        weight_decay=0
    ),
)

scheduler = dict(
    type='PolynomialDecayLR',
    args=dict(
        warmup_updates=__warmup_epochs * ((__number_of_training_items // __batch_size) // __number_of_processes),
        tot_updates=__max_epochs * ((__number_of_training_items // __batch_size) // __number_of_processes),
        lr=2e-4,
        end_lr=1e-9,
        power=1.0
    ),
    interval='step'
)
```
* Observations:
  * Overfit ~> validation came down to ~0.890 and then went up.
  * However, we did not let the training complete (it was scheduled for 400 epochs with decreasing LR, killed at 100)
* WandB Report: [Link](https://wandb.ai/shayanfazeli/graphite_grpe_advanced/reports/GRPE-Large-Graphormer--VmlldzoyODA3MTQw?accessToken=phumvh8o8n8qlqbyjnsb83ifoi7h707a3flys8osrk2u2jy50bo10zogcrauzlyz)

### `exp6`
At the core, the same main set of hyperparameters are used again:
```python
__batch_size = 256
__warmup_epochs = 3
__max_epochs = 400
__shortest_path_length_type_upperbound = 5  # for the shortest-path-type (discrete) to be embedded
__shortest_path_length_upperbound = 5  # for graphormer-like path embedding

__attention_biases = [
                    'edge',
                    'shortest_path_length',
                    'shortest_path'
                ]
__path_encoding_code_dim = 4
__encode_node_degree_centrality = True
```
* Observations:
  * Overfit ~> validation came down to ~0.890 and then went up.
  * However, we did not let the training complete (it was scheduled for 400 epochs with decreasing LR, killed at 100)
* WandB Report: [Link](https://wandb.ai/shayanfazeli/graphite_grpe_advanced/reports/GRPE-Large-Graphormer--VmlldzoyODA3MTQw?accessToken=phumvh8o8n8qlqbyjnsb83ifoi7h707a3flys8osrk2u2jy50bo10zogcrauzlyz)

### `exp1`
* Model: *GRPE Large + Node degree centrality + Path encoding*
* Main Hyperparameters:
```python
__batch_size = 128
__warmup_epochs = 1
__max_epochs = 20
__shortest_path_length_type_upperbound = 20  # for the shortest-path-type (discrete) to be embedded
__shortest_path_length_upperbound = 20  # for graphormer-like path embedding

__attention_biases = [
                    'edge',
                    'shortest_path_length',
                    'shortest_path'
                ]
__path_encoding_code_dim = 16
__encode_node_degree_centrality = True
```
* Result
  * `NaN` loss encountered, rerunning with different seed



### `exp2`

* Main Hyerparameters:
  * base: `exp1`
  * different optimization and scheduling

```python
optimizer = dict(
    type='AdamW',
    args=dict(
        lr=1e-2,
        weight_decay=0
    ),
)

scheduler = dict(
    type='CosineAnnealingLR',
    args=dict(
        T_max=__max_epochs * ((__number_of_training_items // __batch_size) // __number_of_processes)
    ),
    interval='step'
)
```

* Results
  * Loss quickly became `NaN`

### `exp3`
* base: `exp1`
* Main Hyerparameters:
  * shorter paths covered and lower path encoding dim
```python
__batch_size = 128
__warmup_epochs = 1
__max_epochs = 20
__shortest_path_length_type_upperbound = 5  # for the shortest-path-type (discrete) to be embedded
__shortest_path_length_upperbound = 5  # for graphormer-like path embedding

__attention_biases = [
                    'edge',
                    'shortest_path_length',
                    'shortest_path'
                ]
__path_encoding_code_dim = 4
__encode_node_degree_centrality = True
```


* Observations
  * The training MAE seems to be increasing
  * NaN loss

### `exp4`
* base: `exp1`
* Main Hyerparameters:
  * no path encoding
```python
__attention_biases = [
                    'edge',
                    'shortest_path_length',
                    # 'shortest_path'
                ]
```
* Result
  * `NaN` loss encountered (the 1e-3 learning rate seems to break every experiment and not a good choice)





