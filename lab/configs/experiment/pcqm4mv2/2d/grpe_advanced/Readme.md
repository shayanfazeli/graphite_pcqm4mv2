


__Project__: `graphite_grpe_advanced`

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