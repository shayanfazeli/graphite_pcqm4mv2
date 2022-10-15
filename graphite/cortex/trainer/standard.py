import sys
from typing import Dict, List, Any
import apex
import copy
import wandb
from torch.cuda.amp import GradScaler
from tqdm import tqdm
import os
from torch_geometric.data import Data, Batch
import numpy
import torch
import torch.distributed
import torch.nn
import torchmetrics
from graphite.cortex.trainer.base import TrainerBase
from graphite.utilities.distributed.utilities import prepare_model_for_ddp_if_requested, sync_batchnorms
from graphite.utilities.logging import get_logger, log_message
from graphite.utilities.logging.logger import grad_stats
from graphite.utilities.mixed_precision import apex_initialize_optimizer

logger = get_logger(__name__)


class Trainer(TrainerBase):
    def __init__(
            self,
            args,
            config,
            modes: List[str],
            max_epochs: int,
            model,
            metric_monitor: Dict[str, Any],
            device: torch.device,
            data_handler,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler._LRScheduler,
            scheduling_interval: str,
            mixed_precision: bool,
            mixed_precision_backend: str,
            limit_batch_count: int = None,
            validation_interval: int = 1,
            wandb_watch_model: bool = False,
            clip_gradient: float = None
    ):
        super(Trainer, self).__init__()

        # - storing the variables
        self.args = args
        self.config = config
        self.start_epoch = 0
        self.max_epochs = max_epochs
        self.device = device
        self.modes = modes
        self.limit_batch_count = limit_batch_count
        self.validation_interval = validation_interval
        self.clip_gradient = clip_gradient

        # - modules
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.its_wandb_process = (not self.args.distributed) or (self.args.local_rank == 0)

        # - distributed learning preparations if needed
        if self.args.distributed:
            self.model = prepare_model_for_ddp_if_requested(model=self.model, args=self.args)
            self.model = sync_batchnorms(model=self.model, args=args, strategy=None)
            log_message(logger, '~> model is modified for distributed training', self.args)
        else:
            self.model = self.model.to(self.device)

        # - mixed precision
        self.mixed_precision = mixed_precision
        self.mixed_precision_backend = mixed_precision_backend
        self.mixed_precision_preparations()

        self.wandb_watch_model = wandb_watch_model
        if self.its_wandb_process and self.wandb_watch_model:
            wandb.watch(self.model, criterion=self.model.criterion, log='all', log_freq=1)
            log_message(logger, "~> wandb watch is activated.", self.args)

        # - building the metrics
        self.reset_metrics()
        self.metric_monitor = metric_monitor
        self.initialize_metric_monitor()
        log_message(logger, '~> metrics are prepared.', self.args)

        # - owning the data handler
        self.data_handler = data_handler

        # - preparing the dataloaders
        self.dataloaders = data_handler.get_dataloaders()
        log_message(logger, '~> dataloaders are ready.', self.args)

        # - the scheduling interval
        assert scheduling_interval in ['step', 'epoch'], f'unknown scheduling interval: {scheduling_interval}'
        self.scheduling_interval = scheduling_interval  # step or epoch

    def initialize_metric_monitor(self):
        if self.metric_monitor['direction'] == 'min':
            self.best_monitored_metric = numpy.inf
        elif self.metric_monitor['direction'] == 'max':
            self.best_monitored_metric = - numpy.inf
        else:
            raise ValueError

    def mixed_precision_preparations(self):
        """
        mixed precision setup
        """
        if not self.mixed_precision:
            return

        log_message(logger, f"~> mixed precision training (backend: {self.mixed_precision_backend})", self.args)

        if self.mixed_precision_backend == 'amp':
            self.grad_scaler = GradScaler()
        elif self.mixed_precision_backend == 'apex':
            self.model, self.optimizer = apex_initialize_optimizer(model=self.model, optimizer=self.optimizer)
        else:
            raise ValueError()

    def reset_metrics(self) -> None:
        """
        This method can be used to build the metrics required.

        __Remark__: metric mode is not necessarily the same as training mode or dataset modes.
        Please note that the :cls:`Trainer` instance will have a `metrics_config` object, which will be used to create these.
        """
        # for each metric, we need to have the following:
        # 1- a torchmetrics class, 2- the arguments mapping
        self.metrics = dict()
        for mode in self.config['metrics']:
            self.metrics[mode] = dict()
            for cfg in self.config['metrics'][mode]:
                metric = getattr(torchmetrics, cfg['type'])(**cfg['init_args']).to(self.device)
                arg_mapping = cfg['arg_mapping']
                self.metrics[mode][cfg['name']] = dict(
                    metric=metric,
                    arg_mapping=arg_mapping
                )

    def train_epoch(
            self,
            mode: str,
            epoch_index: int,
            dataloader
    ):
        log_message(logger, f"""
        ~> training (split: {mode}) - [epoch: {epoch_index}]
        """, self.args)

        self.model.train()
        dataloader_tqdm = tqdm(enumerate(dataloader), total=len(dataloader))
        for batch_index, batch_data in dataloader_tqdm:
            if self.limit_batch_count is not None:
                if batch_index > self.limit_batch_count:
                    break
            batch_loss_value = self.train_step_and_handle_mixed_precision(mode=mode, batch_index=batch_index, batch_data=self.move_batch_to_device(batch_data))
            dataloader_tqdm.set_description(f'batch_loss: {batch_loss_value:.2f}')
            dataloader_tqdm.refresh()
        return self.monitor_metrics(epoch_index=epoch_index, mode=mode)

    def monitor_metrics(self, epoch_index, mode: str):
        # - computing metrics
        metrics = self.metrics_compute(mode=mode)
        self.monitor_computed_metrics(
            metrics=metrics,
            epoch_index=epoch_index,
            mode=mode
        )
        metrics.update(
            dict(epoch_index=epoch_index)
        )

        if self.its_wandb_process:
            wandb.log({mode: metrics})

        return metrics

    def zero_grad(self):
        for p in self.model.parameters():
            if p.requires_grad:
                p.grad = None

    def check_loss(self, loss: torch.Tensor):
        if loss > 100000:
            logger.error("loss is diverging: terminating the experiment, please check the logs.")
            wandb.finish()
            sys.exit(0)

        if torch.isnan(loss).item():
            logger.error("NaN loss encountered: terminating the experiment, please check the logs.")
            wandb.finish()
            sys.exit(0)

    def clip_model_gradients(self):
        if self.clip_gradient is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_gradient)

    def train_step_and_handle_mixed_precision(self, mode: str, batch_index: int, batch_data):
        self.zero_grad()
        if self.mixed_precision:
            if self.mixed_precision_backend == 'amp':
                with torch.cuda.amp.autocast():
                    loss = self.train_step(mode=mode, batch_index=batch_index, batch_data=batch_data)
                    self.check_loss(loss)
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            elif self.mixed_precision_backend == 'apex':
                loss = self.train_step(mode=mode, batch_index=batch_index, batch_data=batch_data)
                self.check_loss(loss)
                with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                self.optimizer.step()
            else:
                raise ValueError()
        else:
            loss = self.train_step(mode=mode, batch_index=batch_index, batch_data=batch_data)
            self.check_loss(loss)
            loss.backward()
            self.optimizer.step()

        if self.scheduling_interval == 'step':
            self.scheduler.step()

        return loss.item()

    def move_batch_to_device(self, batch):
        if isinstance(batch, Batch):
            batch = batch.to(self.device)
        elif isinstance(batch, Dict):
            for k in batch:
                if isinstance(batch[k], torch.Tensor) or isinstance(batch[k], Batch) or isinstance(batch[k], Data):
                    batch[k] = batch[k].to(self.device)
                elif isinstance(batch[k], List):
                    batch[k] = [self.move_batch_to_device(e) for e in batch[k]]
                elif isinstance(batch[k], Dict):
                    batch[k] = {k2: self.move_batch_to_device(v) for k2, v in batch[k].items()}
                else:
                    raise Exception(f"unsupported batch element")
        else:
            raise Exception(f"unrecognized batch type: {type(batch)}")

        return batch

    def train_step(self, mode: str, batch_index: int, batch_data):
        # - latent representations
        self.model.train()
        loss, outputs = self.model(batch_data)
        self.metrics_forward(mode=mode, outputs=outputs)
        if self.its_wandb_process:
            wandb.log({mode: dict(loss=loss, step_index=batch_index)})
        return loss

    @torch.no_grad()
    def validate_epoch(
            self,
            mode: str,
            epoch_index: int,
            dataloader
    ):
        log_message(logger, f"""
        ~> validation (split: {mode}) - [epoch: {epoch_index}]
        """, self.args)
        self.model.eval()
        dataloader_tqdm = tqdm(enumerate(dataloader))
        for batch_index, batch_data in dataloader_tqdm:
            if self.limit_batch_count is not None:
                if batch_index > self.limit_batch_count:
                    break
            self.validate_step(mode=mode, batch_index=batch_index, batch_data=self.move_batch_to_device(batch_data))
        return self.monitor_metrics(epoch_index=epoch_index, mode=mode)

    @torch.no_grad()
    def validate_step(self, mode: str, batch_index: int, batch_data):
        self.model.eval()
        loss, outputs = self.model(batch_data)
        self.metrics_forward(mode=mode, outputs=outputs)
        return loss

    def resume(self):
        filepath = os.path.join(self.args.logdir, 'ckpts', f"epoch=latest.pth")
        data = torch.load(filepath, map_location='cpu')

        self.model.load_state_dict(data['model'])
        self.optimizer.load_state_dict(data['optimizer'])
        self.scheduler.load_state_dict(data['scheduler'])

        if self.mixed_precision:
            if self.mixed_precision_backend == 'amp':
                self.grad_scaler.load_state_dict(data['grad_scaler'])
            elif self.mixed_precision_backend == 'apex':
                apex.amp.load_state_dict(data['amp'])
            else:
                raise ValueError()

        self.start_epoch = data['epoch_index'] + 1

        if self.args.distributed:
            torch.distributed.barrier()

    def save(self, epoch_index: int, prefix: str = None):
        if (not self.args.distributed) or (self.args.distributed and self.args.rank == 0):
            data_dump = dict(
                epoch_index=epoch_index,
                model=self.model.state_dict(),
                optimizer=self.optimizer.state_dict(),
                scheduler=self.scheduler.state_dict()
            )

            if prefix is None:
                filepath = os.path.join(self.args.logdir, 'ckpts', f"epoch=latest.pth")
            else:
                filepath = os.path.join(self.args.logdir, 'ckpts', f"epoch={epoch_index}{prefix}.pth")

            if self.mixed_precision:
                if self.mixed_precision_backend == 'amp':
                    data_dump['grad_scaler'] = self.grad_scaler.state_dict()
                elif self.mixed_precision_backend == 'apex':
                    data_dump['amp'] = apex.amp.state_dict()
                else:
                    raise ValueError()

            torch.save(data_dump, filepath)
            log_message(logger, f"~> checkpoint stored (filepath={filepath}).", self.args)

        if self.args.distributed:
            torch.distributed.barrier()

    def run(self):
        log_message(logger, '~> running...\n', self.args)
        # - trying to resume
        if self.args.resume:
            log_message(logger, "~> attempting to resume from existing checkpoint", self.args)
            self.resume()

        for epoch_index in range(self.start_epoch, self.max_epochs, 1):
            for mode in [e for e in self.modes if 'train' in e]:
                self.train_epoch(
                    epoch_index=epoch_index,
                    dataloader=self.dataloaders[mode],
                    mode=mode,
                )
            if self.scheduling_interval == 'epoch':
                self.scheduler.step()

            if epoch_index % self.validation_interval == self.validation_interval - 1:
                for mode in [e for e in self.modes if 'train' not in e]:
                    self.validate_epoch(
                        epoch_index=epoch_index,
                        dataloader=self.dataloaders[mode],
                        mode=mode
                    )

            self.save(epoch_index=epoch_index)
            self.reset_metrics()

    def metric_forward(self, metric_name: str, mode: str, outputs: Dict[str, Any]) -> torch.Tensor:
        """
        Parameters
        ----------
        metric_name: `str`, required
            The name of the metric

        mode: `str`, required
            The mode referring to the buffer for the metric computation

        outputs: `Dict[str, Any]`, required
            This information bundle will be used in "progressing" the metric. The corresponding
            `arg_mapping` is provided in the configuration, allowing the agent to know what
            data needs to be fed where.

        Returns
        ----------
        `torch.Tensor`: The outputs of the `torchmetrics` object when we give htem the information
        """
        args_map = self.metrics[mode][metric_name]['arg_mapping']
        input_bundle = {k: outputs[args_map[k]] for k in args_map.keys()}
        return self.metrics[mode][metric_name]['metric'](**input_bundle)

    def metrics_forward(self, mode: str, outputs: Dict[str, Any]) -> None:
        """
        This method, using the data in the outputs, runs the :meth:`metric_forward` on all
        of the metric names corresponding to the *buffer* specified by `mode`.

        Parameters
        ----------
        mode: `str`, required
            The mode referring to the buffer for the metric computation

        outputs: `Dict[str, Any]`, required
            This information bundle will be used in "progressing" the metric. The corresponding
            `arg_mapping` is provided in the configuration, allowing the agent to know what
            data needs to be fed where.
        """
        for metric_name in self.metrics[mode].keys():
            self.metric_forward(metric_name, mode, outputs)

    def metric_compute(self, metric_name, mode) -> torch.Tensor:
        """
        Parameters
        ----------
        metric_name: `str`, required
            The name of the metric

        mode: `str`, required
            The mode referring to the buffer for the metric computation

        Returns
        ----------
        `torch.Tensor`: The outputs of the `torchmetrics` object performing the final computation for the `metric_name`.
        """
        return self.metrics[mode][metric_name]['metric'].compute()

    def metrics_compute(self, mode) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        mode: `str`, required
            The mode referring to the buffer for the metric computation

        Returns
        ----------
        `Dict[str, torch.Tensor]`: For each metric buffer, it includes the outputs of the `torchmetrics` object
        performing the final computation for the `metric_name`.
        """
        output = dict()
        for metric_name in self.metrics[mode].keys():
            output[metric_name] = self.metric_compute(metric_name, mode)

        return output

    def monitor_computed_metrics(self, epoch_index, mode, metrics):
        if mode == self.metric_monitor.get('mode', ''):
            observed = metrics[self.metric_monitor['metric']]
            if (self.metric_monitor['direction'] == 'min' and observed < self.best_monitored_metric) or (self.metric_monitor['direction'] == 'max' and observed > self.best_monitored_metric):
                    self.best_monitored_metric = observed
                    self.save(
                        epoch_index=epoch_index,
                        prefix=f"best-{self.metric_monitor['metric']}")#={observed:.5f}")
