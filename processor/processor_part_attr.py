import logging
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.comm import get_rank, synchronize
from utils.meter import AverageMeter


def _to_scalar(value):
    if torch.is_tensor(value):
        return float(value.item())
    return float(value)


def do_train_part_attr(start_epoch, args, model, train_loader, evaluator, optimizer, scheduler, checkpointer, swanlab_logger=None):
    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {"num_epoch": num_epoch, "iteration": 0, "epoch": start_epoch - 1}

    logger = logging.getLogger("IRRA.train")
    logger.info("start training")

    meters = {
        "loss": AverageMeter(),
        "sdm_loss": AverageMeter(),
        "itc_loss": AverageMeter(),
        "id_loss": AverageMeter(),
        "mlm_loss": AverageMeter(),
        "part_attr_loss": AverageMeter(),
        "upper_part_sdm": AverageMeter(),
        "lower_part_sdm": AverageMeter(),
        "shoes_part_sdm": AverageMeter(),
        "img_acc": AverageMeter(),
        "txt_acc": AverageMeter(),
        "mlm_acc": AverageMeter(),
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)
    best_top1 = 0.0

    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        model.train()

        for n_iter, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            ret = model(batch)
            total_loss = sum([v for k, v in ret.items() if "loss" in k])

            batch_size = batch["images"].shape[0]
            meters["loss"].update(total_loss.item(), batch_size)
            for key in meters.keys():
                if key in ret and key != "loss":
                    value = ret[key]
                    meters[key].update(value.item() if torch.is_tensor(value) else value, batch_size)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                swanlab_data = {
                    "train/epoch": epoch,
                    "train/iter": n_iter + 1,
                    "train/step": (epoch - 1) * len(train_loader) + n_iter + 1,
                    "train/lr": scheduler.get_lr()[0],
                }
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                        swanlab_data[f"train/{k}"] = float(v.avg)
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)
                if swanlab_logger is not None:
                    swanlab_logger.log(swanlab_data, step=swanlab_data["train/step"])

        tb_writer.add_scalar("lr", scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar("temperature", _to_scalar(ret["temperature"]), epoch)
        for k, v in meters.items():
            if v.avg > 0:
                tb_writer.add_scalar(k, v.avg, epoch)
        if swanlab_logger is not None:
            epoch_log = {
                "epoch/epoch": epoch,
                "epoch/lr": scheduler.get_lr()[0],
                "epoch/temperature": _to_scalar(ret["temperature"]),
            }
            for k, v in meters.items():
                if v.avg > 0:
                    epoch_log[f"epoch/{k}"] = float(v.avg)
            swanlab_logger.log(epoch_log, step=epoch)

        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            speed = train_loader.batch_size / time_per_batch
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(
                    epoch, time_per_batch, speed
                )
            )
            if swanlab_logger is not None:
                swanlab_logger.log(
                    {
                        "epoch/time_per_batch": time_per_batch,
                        "epoch/samples_per_sec": speed,
                    },
                    step=epoch,
                )

        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                if args.distributed:
                    top1 = evaluator.eval(model.module.eval())
                else:
                    top1 = evaluator.eval(model.eval())

                torch.cuda.empty_cache()
                if best_top1 < top1:
                    best_top1 = top1
                    arguments["epoch"] = epoch
                    checkpointer.save("best", **arguments)
                if swanlab_logger is not None:
                    eval_log = {
                        "eval/epoch": epoch,
                        "eval/best_R1": float(best_top1),
                    }
                    for key, value in evaluator.latest_result.items():
                        eval_log[f"eval/{key}"] = float(value)
                    swanlab_logger.log(eval_log, step=epoch)

    if get_rank() == 0:
        logger.info(f"best R1: {best_top1} at epoch {arguments['epoch']}")
        if swanlab_logger is not None:
            swanlab_logger.log(
                {
                    "summary/best_R1": float(best_top1),
                    "summary/best_epoch": int(arguments["epoch"]),
                },
                step=num_epoch,
            )
    tb_writer.close()
