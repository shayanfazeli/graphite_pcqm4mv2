import argparse
from graphite.utilities.config.manager.utilities import read_config
from graphite.utilities.device import get_device
from graphite.utilities.randomization.seed import fix_random_seeds
from graphite.utilities.argument_parsing.train import base_args, distributed_args
from graphite.utilities.distributed.utilities import setup_distributed_training_if_requested
import graphite.data.handler as data_handler_lib
import graphite.cortex.optimization.optimizer as optimizer_lib
import graphite.cortex.optimization.scheduler as scheduler_lib
import graphite.cortex.optimization.loss as loss_lib
import graphite.cortex.model as model_lib
import graphite.cortex.trainer as trainer_lib
from graphite.utilities.wandb.utilities import initialize_wandb


def main(args: argparse.Namespace) -> None:
    # - initializing wandb
    initialize_wandb(args=args)

    # - fixing random seed
    fix_random_seeds(seed=args.seed)

    # - preparing the distributed learning
    setup_distributed_training_if_requested(args=args)

    # - reading the configuration
    config = read_config(args)

    # - preparing the data-handler
    data_handler = getattr(data_handler_lib, config['data']['type'])(
        distributed=args.distributed,
        distributed_sampling='all',
        **config['data']['args']
    )

    # - preparing the model
    model = getattr(model_lib, config['model']['type'])(**config['model']['args'])

    # - preparing the optimizer and scheduler and the criterion
    if 'scheduler' not in config:
        config['scheduler'] = dict(
            type='NoScheduler',
            args=dict()
        )

    criterion = getattr(loss_lib, config['loss']['type'])(**config['loss']['args'])
    optimizer = getattr(optimizer_lib, config['optimizer']['type'])(model.parameters(), **config['optimizer']['args'])
    scheduler = getattr(scheduler_lib, config['scheduler']['type'])(optimizer, **config['scheduler']['args'])

    trainer = getattr(trainer_lib, config['trainer']['type'])(
        device=get_device(args.gpu),
        args=args,
        config=config,
        model=model,
        data_handler=data_handler,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduling_interval=config['scheduler'].get('interval', 'step'),
        **config['trainer']['args']
    )

    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = base_args(parser)
    parser = distributed_args(parser)
    args = parser.parse_args()
    main(args=args)