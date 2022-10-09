


__Project__: `graphite_grpe_advanced`

## `exp5`
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
  * 

## `exp6`
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
  * 





# Previously...

## `exp1`
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



## `exp2`

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

## `exp3`
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

## `exp4`
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





