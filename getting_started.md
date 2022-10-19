
# Graphite: Quickstart
This document is a quick intro to the different parts of `graphite` and serves as a quickstart tutorial.

## Configurations
We have followed [MMCV](https://github.com/open-mmlab/mmcv) protocol of nested-py configurations,
allowing a flexible environment for defining base structures and simplifying 
experiment design with inheritence and modifications.


## Preparing a dataset
To create a dataset module, please refer to the examples in the [`graphite.data`](https://github.com/shayanfazeli/graphite_pcqm4mv2/tree/master/graphite/data).
An example class which is a wrapper around `InMemoryDataset` of `torch_geometric`, is the following
dataset class:
* [`PCQM4Mv2DatasetFull`](https://github.com/shayanfazeli/graphite_pcqm4mv2/blob/master/graphite/data/pcqm4mv2/pyg/dataset.py)
    * This dataset allows using 2d version (smiles-based) of [PCQM4Mv2](https://ogb.stanford.edu/docs/lsc/pcqm4mv2/) dataset.
    * In this class, the 3d information can be imported (such as custom generated conformers) by providing the path to the `conformer_memmap`. 
    * The corresponding file must be a `numpy.memmap` of dim `(3746620, 10, 60, 3)` providing up to 10 conformers for all items in the dataset.
    * The `pre_transform` can be leveraged to preprocess and store a version of the dataset with a customized transformation pipeline being applied on it already.
      * In the PCQM4Mv2 dataset, often times this leads to a huge data file and therefore in most cases it is not recommended to perform
      the transformations in the `pre_transform`.
    * Just like most of the computer vision datasets, you can use the `torchvision.transforms.Compose` along with
    the predefined transforms to get a custom view of your dataset. Please find a complete list of transforms below.
    * The [`MultiviewPCQM4Mv2Dataset`](https://github.com/shayanfazeli/graphite_pcqm4mv2/blob/master/graphite/data/pcqm4mv2/pyg/dataset_multiview.py) is another important
  dataset class, which allows you to create a multi-view wrapper on your datasets. This is useful for various pretraining approaches
  such as an algorithm like SimCLR, SwAv, or 3d-specific denoising pretraining objectives such as SE(3)-DDM.

Our __data_hub__ containing the preprocessed and stored
version of many of the critical data is available online in this link: [[Link]](https://drive.google.com/drive/folders/1pWE3alcpk7MsMklsJss6nv7slIvANMFC?usp=sharing)

Please find the supported dataset instances and their corresponding
configurations in the list below (to access the cached version for each, please refer to our __data_hub__):
* For GNNs
  * 2D-graph Generic: [[config]](https://github.com/shayanfazeli/graphite_pcqm4mv2/blob/master/lab/configs/dataset/pcqm4mv2/dataset_2d.py)
  * 2D-graph with [KPGT](https://arxiv.org/pdf/2206.03364.pdf) Molecular Information: [[config]](https://github.com/shayanfazeli/graphite_pcqm4mv2/blob/master/lab/configs/dataset/pcqm4mv2/dataset_2d.py)
  * 2D-graph - Represented as Line Graph (edge-to-vertex dual): [[config]](https://github.com/shayanfazeli/graphite_pcqm4mv2/blob/master/lab/configs/dataset/pcqm4mv2/dataset_2d_linegraph.py)
  * 2D-graph with [KPGT](https://arxiv.org/pdf/2206.03364.pdf) Molecular Information- Represented as Line Graph (edge-to-vertex dual) : [[config]](https://github.com/shayanfazeli/graphite_pcqm4mv2/blob/master/lab/configs/dataset/pcqm4mv2/dataset_2d_kpgt_linegraph.py)
  * 3D-graph with Bonds as edges: [[config]](https://github.com/shayanfazeli/graphite_pcqm4mv2/blob/master/lab/configs/dataset/pcqm4mv2/dataset_3d_bond.py)
  * 3D-graph with Bonds as edges with [KPGT](https://arxiv.org/pdf/2206.03364.pdf) Molecular Information: [[config]](https://github.com/shayanfazeli/graphite_pcqm4mv2/blob/master/lab/configs/dataset/pcqm4mv2/dataset_3d_bond_kpgt.py)
  * 3D-graph with Bonds as edges - Represented as Line Graph (edge-to-vertex dual): [[config]](https://github.com/shayanfazeli/graphite_pcqm4mv2/blob/master/lab/configs/dataset/pcqm4mv2/dataset_3d_bond_linegraph.py)
  * 3D-graph with Bonds as edges with [KPGT](https://arxiv.org/pdf/2206.03364.pdf) Molecular Information - Represented as Line Graph (edge-to-vertex dual): [[config]](https://github.com/shayanfazeli/graphite_pcqm4mv2/blob/master/lab/configs/dataset/pcqm4mv2/dataset_3d_bond_kpgt_linegraph.py)
* For Transformers
  * 2D-graph Generic - For Transformers: [[config]](https://github.com/shayanfazeli/graphite_pcqm4mv2/blob/master/lab/configs/dataset/pcqm4mv2/dataset_2d_for_transformers_base.py)
  * 2D-graph with [KPGT](https://arxiv.org/pdf/2206.03364.pdf) Molecular Information - For Transformers: [[config]](https://github.com/shayanfazeli/graphite_pcqm4mv2/blob/master/lab/configs/dataset/pcqm4mv2/dataset_2d_for_transformers_kpgt.py)
* Multi-view
  * 3D-graph with Bonds as edges - two view by perturbing conformers: [[config]](https://github.com/shayanfazeli/graphite_pcqm4mv2/blob/master/lab/configs/dataset/pcqm4mv2/dataset_3d_bond_doubleview.py)
    * This dataset is particularly useful for single-model multi-view pretraining with SE(3)-DDM objective.

__Remark__: The representation as *Line Graph* is being actively worked on. Currently, the strategy
is to leverage [`LineGraphTransform`](https://github.com/shayanfazeli/graphite_pcqm4mv2/blob/master/graphite/data/pcqm4mv2/pyg/transforms/graph/line_graph.py) transform and set `self_loop_strategy='only_isolated_nodes'`. This is to ensure
that information for isolated atoms don't get lost while also not resulting in an unnecessarily large graph.


### Transforms

#### Node-level

* transform: [`AddTaskNode`](https://github.com/shayanfazeli/graphite_pcqm4mv2/blob/master/graphite/data/pcqm4mv2/pyg/transforms/node/add_task_node.py)
  * Adding a task node at the end of the graph. Adding a virtual node with a pre-defined node-attribute is
  required for Transformer-based approaches such as GRPE and Graphormer.
* transform: [`EncodeNode2NodeConnectionType`](https://github.com/shayanfazeli/graphite_pcqm4mv2/blob/master/graphite/data/pcqm4mv2/pyg/transforms/node/node2node_connection_type_encoding.py)
  * The edges in PCQM4Mv2 are characterized by a 3-dimensional set of discrete attributes. This transform helps with embedding such connection as well
  handling the special edges such as task edge. The main use of this transform is in training with Transformers.
* transform: [`EncodeNode2NodeShortestPathFeatureTrajectory`](https://github.com/shayanfazeli/graphite_pcqm4mv2/blob/master/graphite/data/pcqm4mv2/pyg/transforms/node/node2node_shortest_path_encoding.py)
  * This transform provides the path encoding between two nodes which would allow a Graphormer-like encoding of edges comprising the shortest path between two nodes.
* transform: [`EncodeNode2NodeShortestPathLengthType`](https://github.com/shayanfazeli/graphite_pcqm4mv2/blob/master/graphite/data/pcqm4mv2/pyg/transforms/node/node2node_shortest_path_length_type_encoding.py)
  * The shortest path length (in terms of number of edges) will be encoded using this transform. Note that the longest path are usually upperbounded to a low value such as 5.
* transform: [`EncodeNodeDegreeCentrality`](https://github.com/shayanfazeli/graphite_pcqm4mv2/blob/master/graphite/data/pcqm4mv2/pyg/transforms/node/node_degree_centrality_encoding.py)
  * Degree centrality encoding of the nodes can be done using this transform.
* transform: [`EncodeNodeType`](https://github.com/shayanfazeli/graphite_pcqm4mv2/blob/master/graphite/data/pcqm4mv2/pyg/transforms/node/node_type_encoding.py)
  * Encoding the node type, which in this case is adding the corresponding `offset` to the feature positions.
* transform: [`PairwiseDistances`](https://github.com/shayanfazeli/graphite_pcqm4mv2/blob/master/graphite/data/pcqm4mv2/pyg/transforms/node/pairwise_distances.py)
  * 
* transform: [`Position3DGaussianNoise`](https://github.com/shayanfazeli/graphite_pcqm4mv2/blob/master/graphite/data/pcqm4mv2/pyg/transforms/node/position_3d_noise.py)

#### Edge-level
* transform: [`ComenetEdgeFeatures`](https://github.com/shayanfazeli/graphite_pcqm4mv2/blob/master/graphite/data/pcqm4mv2/pyg/transforms/edge/comenet_features_3d.py)
* transform: [`EncodeEdgeType`](https://github.com/shayanfazeli/graphite_pcqm4mv2/blob/master/graphite/data/pcqm4mv2/pyg/transforms/edge/edge_encoding.py)
* transform: [`RadiusGraphEdges`](https://github.com/shayanfazeli/graphite_pcqm4mv2/blob/master/graphite/data/pcqm4mv2/pyg/transforms/edge/radius_graph.py)
* 
#### Graph-level
* transform: [`LineGraphTransform`](https://github.com/shayanfazeli/graphite_pcqm4mv2/blob/master/graphite/data/pcqm4mv2/pyg/transforms/graph/line_graph.py)
  * The Line-graph representation transform which internally uses PyG's `LineGraph` class.


## Methods and Models
A `model` in graphite refer to all subclasses of `torch.nn.Module` that will be used in an active training pipeline. The main
difference is that `method`s are models that would work using an internal `model` for representing graph nodes, and will
output a `loss` and an output bundle.
There are also `pretext` modules that can be defined and used, making the third superclass of differentiable modules in graphite.

As an example, GRPE is a model but not a method as it is focused on converting a graph to latent representations.
a Regressor, however, is a method as its ultimate output is focused on predicting the HUMO-LUMO Gap.

## Method

### Regression
* [`Regressor`](https://github.com/shayanfazeli/graphite_pcqm4mv2/blob/master/graphite/cortex/model/method/supervised/regression/regressor.py)
  * Generic regressor model that can be a wrapper over complicated modules such as GRPE. The internal module is configured
  usig the `model_config`, as well as the `loss_config` which defines the base loss.
* [`RegressorWithKPGTRegularization`](https://github.com/shayanfazeli/graphite_pcqm4mv2/blob/master/graphite/cortex/model/method/supervised/regression/regressor_kpgt.py)
  * The option for KPGT regularization losses is provided to this method, which is an extension of [`Regressor`](https://github.com/shayanfazeli/graphite_pcqm4mv2/blob/master/graphite/cortex/model/method/supervised/regression/regressor.py).
  * The 512 binary vector corresponding to molecular fingerprint, as well as a 200 dimensional molecular descriptor are there as 
  prediction objectives.
  * The `pos_weight` for the `BCEWithLogitsLoss` type loss which is presumed to be selected for fingerprint is precomputed
  and present in the module, and will be automatically applied.
  * This would be deprecated in favor of `Pretext` type modules.

### Pre-training
* [`SingleModelSingleViewPretrainingWithPretexts`](https://github.com/shayanfazeli/graphite_pcqm4mv2/blob/master/graphite/cortex/model/method/self_supervised/single_model/singleview_pretraining.py)
* [`SingleModelMultiViewPretrainingWithPretexts`](https://github.com/shayanfazeli/graphite_pcqm4mv2/blob/master/graphite/cortex/model/method/self_supervised/single_model/multiview_pretraining.py)
  * The single-model multi-view target regression + pretext loss augmentation, which would allow defining and using pretexts that work on a single view and
  leveraging the latent representations generated by a single model.

Here is an example configuration for such pre-training model designs:
```python

model = dict(
    type='SingleModelMultiViewPretrainingWithPretexts',
    args=dict(
        num_layers=0,
        input_dim=256,
        output_dim=1,
        model_config=dict(
            type='CoAtGINGeneralPipeline',
            args=dict(
                node_encoder_config=dict(
                    type='CoAtGIN',
                    args=dict(
                        num_layers=6,
                        num_heads=16,
                        model_dim=256,
                        conv_hop=3,
                        conv_kernel=2,
                        use_virt=True,
                        use_att=True,
                        line_graph=False,
                        pos_features=None
                    )
                ),
                graph_pooling="sum"
            )
        ),
        pretext_configs=dict(se3ddm=dict(
            type='SE3DDMPretext',
            args=dict(
                weight=1.,
                mlp_distances_args=dict(
                    input_dim=1,
                    output_dim=256,
                    hidden_dim=32,
                    num_hidden_layers=1,
                    norm='LayerNorm',
                    activation='ReLU',
                    input_norm='BatchNorm1d',
                    input_activation='none'
                ),
                mlp_merge_args=dict(
                    input_dim=256,  # graph rep dim
                    output_dim=1,
                    hidden_dim=32,
                    num_hidden_layers=1,
                    norm='LayerNorm',
                    activation='ReLU',
                    input_norm='BatchNorm1d',
                    input_activation='none'),
                scale=0.1,
                mode='with_distances'
            )
        ))
    )
)
```

## Model

### Graph Relative Positional Encoding [GRPE]
* Re-implementation of the [GRPE paper](https://openreview.net/pdf?id=GNfAFN_p1d): [[`GraphRelativePositionalEncodingNetwork`]](https://github.com/shayanfazeli/graphite_pcqm4mv2/blob/master/graphite/cortex/model/model/grpe/grpe.py)
* Customizable Extended GRPE: [[`GraphRelativePositionalEncodingNetworkAdvanced`]](https://github.com/shayanfazeli/graphite_pcqm4mv2/blob/master/graphite/cortex/model/model/grpe_advanced/grpe.py)
  * This allows leveraging degree centrality encoding and path encoding as well (from [Graphormer](https://proceedings.neurips.cc/paper/2021/file/f1c1592588411002af340cbaedd6fc33-Paper.pdf) paper) and toeplitz attention-bias focus retainment (from [this paper](https://arxiv.org/pdf/2205.13401.pdf)).

### GNN / GCN / Attention / Virtual Node /CoAtGIN
* Modules for general GNN/GCN and [CoAtGIN](https://www.biorxiv.org/content/10.1101/2022.08.26.505499v1) paper
* Please see [this notebook](https://github.com/shayanfazeli/graphite_pcqm4mv2/blob/master/lab/notebooks/models/coatgin/adapted_coatgin.ipynb) for the demo usage and possible options.

### MLP pipeline
* [`CustomMLPHead`](https://github.com/shayanfazeli/graphite_pcqm4mv2/blob/master/graphite/cortex/model/model/mlp/mlp.py): A customizeable quick to instantiate module for creating MLP heads (also please check out the Gated Linear Unit for such pipelines).

### Pre-text tasks
* [`SE3DDMPretext`](https://github.com/shayanfazeli/graphite_pcqm4mv2/blob/master/graphite/cortex/model/pretext/se3ddm.py): A single model double view pretext task, with the idea of
perturbing 3d positions of a conformer and its perturbed version, and do a SwAV-like loss. This pretext task is from [this article](https://arxiv.org/pdf/2206.13602.pdf).
  * By setting `mode` to different values you can select where to perform pair-wise distance computations.


## Conformer generation
[`graphite_confgen`](https://github.com/shayanfazeli/graphite_pcqm4mv2/blob/master/graphite/bin/graphite_confgen.py): Our multi-process multi-threaded conformer generator script.