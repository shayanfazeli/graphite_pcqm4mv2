# 4-fold training

* Model: `grpe_large + ndce`
  * Corresponding to `exp10`

## Running
* First, please correct the paths in the base config
* Please correct `__number_of_processes = 4` in the base config to your gpu counts.
* Multi-gpu:
  * Try: `./lab/configs/experiment/pcqm4mv2/2d/grpe_advanced/jp_splits/grpe_large+ndce/run.sh fold_0`
* Single-gpu:
  * Try: `./lab/configs/experiment/pcqm4mv2/2d/grpe_advanced/jp_splits/grpe_large+ndce/run_single.sh fold_0 1`

### Resuming
To resume, use the corresponding `resume_run.sh [exp_name] [wandb_id]` or `resume_run_single [exp_name] [gpu_id] [wandb_id]`.