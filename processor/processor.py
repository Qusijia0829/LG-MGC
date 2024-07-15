import logging
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable


def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer):
    log_period = args.log_period
    eval_period = args.eval_period
    pool_weight = args.pool_weight
    top_weight = args.top_weight
    pred_weight = args.pred_weight
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("LG-MGC.train")
    logger.info('start training')

    meters = {
        "loss": AverageMeter(),
        "itc_loss": AverageMeter(),
        "itc_combined_loss": AverageMeter(),
        "itc_topsm_loss": AverageMeter(),
        "loss_pred_ip": AverageMeter(),
        "loss_pred_tp": AverageMeter(),
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_top1 = 0.0

    # train
    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        model.train()

        for n_iter, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            total_loss = 0
            ret = model(batch ,epoch)
            for k, v in ret.items():
                if "loss" in k:
                    if "pred" in k:
                        total_loss = total_loss + pred_weight * v
                    elif "topsm" in k:
                        total_loss = total_loss + pool_weight * v
                    elif "combined" in k:
                        total_loss = total_loss + top_weight * v
                    else:
                        total_loss = total_loss + v


            batch_size = batch['images'].shape[0]
            meters['loss'].update(total_loss.item(), batch_size)
            meters['itc_loss'].update(ret.get('itc_loss', 0), batch_size)
            meters['itc_combined_loss'].update(ret.get('itc_combined_loss', 0), batch_size)
            meters['itc_topsm_loss'].update(ret.get('itc_topsm_loss', 0), batch_size)
            meters['loss_pred_ip'].update(ret.get('loss_pred_ip', 0), batch_size)
            meters['loss_pred_tp'].update(ret.get('loss_pred_tp', 0), batch_size)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)

        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        for k, v in meters.items():
            if v.avg > 0:
                tb_writer.add_scalar(k, v.avg, epoch)

        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                if args.distributed:
                    top1 = evaluator.eval(model.module.eval(),epoch)
                else:
                    top1 = evaluator.eval(model.eval(),epoch)

                torch.cuda.empty_cache()
                #
                if best_top1 < top1  :
                    best_top1 = top1
                    arguments["epoch"] = epoch
                    checkpointer.save("best", **arguments)
    if get_rank() == 0:
        logger.info(f"best mean: {best_top1} at epoch {arguments['epoch']}")


def do_inference(model, test_img_loader, test_txt_loader):
    logger = logging.getLogger("LG-MGC.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader)
    mean = evaluator.eval(model.eval())
