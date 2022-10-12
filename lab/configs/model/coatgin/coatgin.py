
model = dict(
    type='Regressor',
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
                        num_layers=4,
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
        )
    )
)
