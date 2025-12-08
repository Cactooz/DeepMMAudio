from datetime import timedelta
import logging
import random

import hydra
import numpy as np
import torch
import torch.distributed as distributed
from tqdm import tqdm
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from mmaudio.data.data_setup import setup_val_datasets
from mmaudio.model.sequence_config import CONFIG_16K, CONFIG_44K
from mmaudio.runner import Runner
from mmaudio.utils.dist_utils import info_if_rank_zero, local_rank, world_size
from mmaudio.utils.logger import TensorboardLogger

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

log = logging.getLogger()


def distributed_setup():
    distributed.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    local_rank = 0
    world_size = 1
    log.info(f'Initialized: local_rank={local_rank}, world_size={world_size}')
    return local_rank, world_size


@hydra.main(version_base='1.3.2', config_path='config', config_name='eval_config.yaml')
def evaluate(cfg: DictConfig):
    # initial setup
    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    distributed_setup()
    num_gpus = world_size
    run_dir = HydraConfig.get().run.dir

    # patch data dim
    if cfg.model.endswith('16k'):
        seq_cfg = CONFIG_16K
    elif cfg.model.endswith('44k'):
        seq_cfg = CONFIG_44K
    else:
        raise ValueError(f'Unknown model: {cfg.model}')
    with open_dict(cfg):
        cfg.data_dim.latent_seq_len = seq_cfg.latent_seq_len
        cfg.data_dim.clip_seq_len = seq_cfg.clip_seq_len
        cfg.data_dim.sync_seq_len = seq_cfg.sync_seq_len

    log = TensorboardLogger(cfg.exp_id,
                            run_dir,
                            logging.getLogger(),
                            is_rank0=(local_rank == 0),
                            enable_email=False)

    info_if_rank_zero(log, f'All configuration: {cfg}')
    info_if_rank_zero(log, f'Number of GPUs detected: {num_gpus}')
    info_if_rank_zero(log, f'Number of dataloader workers (per GPU): {cfg.num_workers}')

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    info_if_rank_zero(log, f'Evaluation configuration: {cfg}')
    cfg.batch_size //= num_gpus
    info_if_rank_zero(log, f'Batch size (per GPU): {cfg.batch_size}')

    _, _, eval_loader = setup_val_datasets(cfg)
    val_cfg = cfg.data.ExtractedVGG_val

    evaluator = Runner(cfg,
                       log=log,
                       run_path=run_dir,
                       for_training=False).enter_val()

    if cfg['checkpoint'] is not None:
        evaluator.load_weights(cfg['checkpoint'])
        info_if_rank_zero(log, 'Model weights loaded!')
    else:
        raise ValueError("Checkpoint path must be provided for evaluation.")

    info_if_rank_zero(log, 'Starting evaluation...')
    audio_path = None
    
    for data in tqdm(eval_loader):
        audio_path = evaluator.inference_pass(data,
                                             it=0,
                                             data_cfg=val_cfg,
                                             save_eval=False)
    distributed.barrier()
    if audio_path is not None:
        evaluator.eval(audio_path, it=0, data_cfg=val_cfg)

    log.complete()
    distributed.barrier()
    distributed.destroy_process_group()


if __name__ == '__main__':
    evaluate()
