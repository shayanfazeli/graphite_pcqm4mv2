# 3D CoATGIN

## Experiment Group: `3d_bond`
### Exp 1
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

## Experiment Group: `3d_bond_kpgt`
### Exp 1
Same setup as the exp1 for `3d_bond`, with additional KPGT loss.
* params: 10,086,361


## Base experiments: Deprecated
* non-masked comenet features
* Gated linear unit was used instead of the above MLP for representing positional features
* WandB Report: [[Link](https://wandb.ai/shayanfazeli/graphite_coatgin_docker/reports/Basic-3D-CoATGIN--VmlldzoyODA3MzIw?accessToken=4a25cfsr1hetl2f9z1d23c2ceohokylf1eo60qxq1x5qkcxp5beudfdw5qopydji)]
