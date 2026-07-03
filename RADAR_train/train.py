import os
import argparse
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from lavis.common.registry import registry
from lavis.common.utils import now

from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners import *
from lavis.tasks import *

torch.autograd.set_detect_anomaly(True)


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=False, default='radar_config.yaml', help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls


def main():

    job_id = now()

    cfg = Config(parse_args())
    
    init_distributed_mode(cfg.run_cfg)

    setup_seeds(cfg)

    setup_logger()

    cfg.pretty_print()

    task = tasks.setup_task(cfg)

    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)

    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    runner.train()


if __name__ == "__main__":
    main()
