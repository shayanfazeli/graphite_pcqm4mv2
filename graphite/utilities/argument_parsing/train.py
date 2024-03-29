import argparse


def distributed_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='local rank for distributed training')
    return parser


def base_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("config", help="path to the config file")
    parser.add_argument("--seed", default=42, help="Random seed.", type=int)
    parser.add_argument("--clean", action="store_true", help="Restart training and erase previous checkpoint contents.")
    parser.add_argument("--name", type=str, required=True, help="experiment title")
    parser.add_argument("--project", type=str, required=True, help="project title")
    parser.add_argument("--id", type=str, required=False, default=None, help="the project id, only needed for resuming")
    parser.add_argument("--wandb_offline", action="store_true", help="""
    If this parameter is set to true, the wandb logging will take place in an offline manner.
    If provided as true, the `wandb_apikey` has to be set as well.""")
    parser.add_argument("--evaluate", action="store_true", help="""
        If `True` the script will go into evaluation mode and generates test results.""")
    parser.add_argument("--wandb_apikey", type=str, required=False, default=None, help="""
        wandb api key""")
    parser.add_argument("--gpu", required=False, default=None, help="The gpu device to use. this will be overriden in case of distributed `torchrun` script.", type=int)

    parser.add_argument("--config_overrides", required=False, default=None, type=str, help="""
        For a variety of purposes, one might one to override the default values of a config file.
        An example would be to cover a range for a variable, etc.

        The syntax is:
        --config_overrides='{"trainer": {"args": {"lr": 0.1}}}'

        What happens is that the `json.loads` function will be called on the contents of this parameter, and the
        config will be updated with these values accordingly.
        """)

    parser.add_argument("--resume", action="store_true", help="""
    When resuming an experiment, please provide the `--resume` and also make sure to provide the wandb id in `--id`.""")

    parser.add_argument("--logdir", required=True, type=str, help="""
    The information regarding this experiment will be stored in the `os.path.join(args.logdir, args.project, args.name, args.id)`.
    Please note that for an experiment running for the first time (when `--resume` is not provided),
    the id must be left alone and it will be generated by the script. For resuming, however,
    the id has to be provided.""")
    return parser


def training_args(parser) -> argparse.ArgumentParser:
    """
    Returns the argument parser for the train command.
    """

    return parser
