import os
import os.path as op
import random
import time

import numpy as np
import torch

from datasets import build_dataloader
from model.build_multi_vector import build_model_multi_vector
from processor.processor_multi_vector import do_train_multi_vector
from solver import build_lr_scheduler, build_optimizer
from utils.checkpoint import Checkpointer
from utils.comm import get_rank, synchronize
from utils.iotools import save_train_configs
from utils.logger import setup_logger
from utils.metrics_multi_vector import EvaluatorMultiVector
from utils.options_multi_vector import get_args_multi_vector
from utils.swanlab_logger import build_swanlab_logger


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    args = get_args_multi_vector()
    set_seed(1 + get_rank())
    name = args.name

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    device = "cuda"
    cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    args.output_dir = op.join(args.output_dir, args.dataset_name, f"{cur_time}_{name}")
    logger = setup_logger("IRRA", save_dir=args.output_dir, if_train=args.training, distributed_rank=get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(str(args).replace(",", "\n"))
    save_train_configs(args.output_dir, args)
    swanlab_logger = build_swanlab_logger(args)

    train_loader, val_img_loader, val_txt_loader, num_classes = build_dataloader(args)
    model = build_model_multi_vector(args, num_classes)
    logger.info("Total params: %2.fM" % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    optimizer = build_optimizer(args, model)
    scheduler = build_lr_scheduler(args, optimizer)

    is_master = get_rank() == 0
    checkpointer = Checkpointer(model, optimizer, scheduler, args.output_dir, is_master)
    evaluator = EvaluatorMultiVector(
        val_img_loader,
        val_txt_loader,
        global_score_weight=args.eval_global_score_weight,
        local_score_weight=args.eval_local_score_weight,
        chunk_size=args.eval_local_chunk_size,
    )

    start_epoch = 1
    if args.resume:
        checkpoint = checkpointer.resume(args.resume_ckpt_file)
        start_epoch = checkpoint["epoch"]

    try:
        do_train_multi_vector(
            start_epoch,
            args,
            model,
            train_loader,
            evaluator,
            optimizer,
            scheduler,
            checkpointer,
            swanlab_logger=swanlab_logger,
        )
    finally:
        swanlab_logger.finish()
