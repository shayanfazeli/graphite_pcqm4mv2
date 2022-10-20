# 3D CoATGIN

## Experiment Group: `3d_bond`
### Exp 1
* number of parameters: 9,770,769
* The CoAtGIN with the following 3d representation:
```python
self.pos_encoder.append(
    CustomMLPHead(
        input_dim=pos_features,
        output_dim=width,
        input_norm='BatchNorm1d',
        num_hidden_layers=2,
        hidden_dim=width//width_head,
        activation='ReLU',
        norm='LayerNorm',
        output_norm='LayerNorm',
        dropout=0.2
    )
)
```
and masking all zero comenet features, and formulating the impact in an additive fashion (like Transformer-M)
```python
pos_mask = torch.any(edge_attr[:, 3:], dim=1, keepdim=True)
ea = self.bond_encoder[layer](edge_attr[:, :3].long()) + pos_mask * self.pos_encoder[layer](edge_attr[:, 3:])

```
* WandB Report: [[Link](https://wandb.ai/shayanfazeli/graphite_coatgin_docker/reports/CoAtGIN-base-Generated-conformer-3d-KPGT-Losses--VmlldzoyODI2MTQ2?accessToken=ia83542v1bkk6h3l8bac0jt5hzand8i4mi3fqprluoqtsqxt3zu1sutcp0u751ss)]

### Exp 2
* number of parameters:
* This experiment is leveraging the contrastive denoising objective of [this article](https://arxiv.org/pdf/2206.13602.pdf).

### Exp 3
* The same as Exp1 except:
  * Larger version of CoAtGIN (deeper with 12 layers and 19,496,481 parameters)
  * The 3d bond dataset __from SDF__ for train and using generated conformers for val


## Experiment Group: `3d_bond_kpgt`
### Exp 1
Same setup as the exp1 for `3d_bond`, with additional KPGT loss.
* params: 10,086,361
* WandB Report: [[Link](https://wandb.ai/shayanfazeli/graphite_coatgin_docker/reports/CoAtGIN-base-Generated-conformer-3d-KPGT-Losses--VmlldzoyODI2MTQ2?accessToken=ia83542v1bkk6h3l8bac0jt5hzand8i4mi3fqprluoqtsqxt3zu1sutcp0u751ss)]


## Base experiments: Deprecated
* non-masked (regarding masking all 0 (nan) edges) comenet features were being utilized
* Gated linear unit was used instead of the above MLP for representing positional features
* WandB Report: [[Link](https://wandb.ai/shayanfazeli/graphite_coatgin_docker/reports/Basic-3D-CoATGIN--VmlldzoyODA3MzIw?accessToken=4a25cfsr1hetl2f9z1d23c2ceohokylf1eo60qxq1x5qkcxp5beudfdw5qopydji)]
