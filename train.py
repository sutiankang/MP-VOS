import argparse
import datetime
import time
import os
import os.path as osp
import random
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, DistributedSampler, BatchSampler, SequentialSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from mpvos.utils import (init_distributed_mode, get_size, is_main_process, save_on_master, get_rank,
                         setup_logger, create_dataset, get_scheduler)
from mpvos.engine import create_iou_bce_loss, train_one_epoch, evaluate
from mpvos.models import MPVOS


def get_parser():
    parser = argparse.ArgumentParser("MP-VOS")

    parser.add_argument("--start-epoch", type=int, default=1, help="start training epoch")
    parser.add_argument("-e", "--epochs", type=int, default=30, help="training total epochs")
    parser.add_argument("-b", "--batch-size", type=int, default=2, help="batch size for DataLoader")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for single gpu")
    parser.add_argument("--img_size", type=int, nargs="+", default=512, help="training image size")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="choose using device")
    parser.add_argument("--summary", action="store_true", default=False, help="use summary writer to record")
    parser.add_argument("--save_best", action="store_true", default=False, help="only save best weights")
    parser.add_argument("--data-dir", type=str, default="/data/stk/UVOS_data/", help="dataset path")
    parser.add_argument("--eval_interval", type=int, default=1, help="evaluate interval")

    # ["YouTubeVOS-2018", "DAVIS-2016"]
    parser.add_argument("-t", "--train_datasets", type=str, nargs="+",
                        default=["YouTubeVOS-2018"])
    # ["DAVIS-2016", "FBMS", "ViSal"]
    parser.add_argument("-v", "--val_datasets", type=str, nargs="+",
                        default=["DAVIS-2016"])

    parser.add_argument("--experiment", type=str, default="", help="experiment name")
    parser.add_argument("--model", type=str, default="resnet101", help="model name",
                        choices=["resnet50", "resnet101"])

    parser.add_argument("--stride", type=int, default=7, help="sample youtube frames")

    # optimizer
    parser.add_argument("--optimizer", type=str, default="adamW", choices=["sgd", "adam", "adamW"])
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "lr_step"],
                        help="learning rate strategy")
    # use lr_step
    parser.add_argument("--lr-decay-epochs", default=[10, 20, 40], nargs="+", type=int,
                        help='decrease lr every step-size epochs')
    parser.add_argument('--lr_decay_steps', type=int, default=10, help='for step scheduler. step size to decay lr')
    parser.add_argument("--warmup_epochs", type=int, default=0, help="warmup epochs, 0 no use")
    parser.add_argument("--warmup_multiplier", type=int, default=100, help="warmup_multiplier")
    parser.add_argument('--lr-gamma', default=0.9, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="optimizer weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="sgd momentum parameters")

    parser.add_argument("--mean", type=list, default=[0.485, 0.456, 0.406], help="imagenet mean")
    parser.add_argument("--std", type=list, default=[0.229, 0.224, 0.225], help="imagenet std")
    parser.add_argument("--sync_bn", action="store_true", default=False,
                        help="distributed training merge batch_norm layer mean and std")
    parser.add_argument("--max_norm", type=float, default=0., help="gradient clipping max norm")

    parser.add_argument("--multi_scale", action="store_true", default=False, help="multi-scale training.")
    parser.add_argument("--scales", type=float, nargs="+", default=[0.5, 0.75, 1.0, 1.25, 1.5], help="multi scale ratio.")
    parser.add_argument("--pretrained", action="store_true", default=False, help="use pretrained weights")
    parser.add_argument("--finetune", type=str, help="finetune weight path",
                        default="")
    parser.add_argument('--resume', type=str, default='', help='resume from checkpoint')

    return parser.parse_args()


def main():

    args = get_parser()
    if not args.multi_scale:
        cudnn.benchmark = True
    init_distributed_mode(args)

    if not args.experiment:
        if not args.finetune:
            args.experiment = osp.join(osp.dirname(__file__), "runs/train")
        else:
            args.experiment = osp.join(osp.dirname(__file__), "runs/finetune")
    else:
        if not args.finetune:
            args.experiment = osp.join(args.experiment, "train")
        else:
            args.experiment = osp.join(args.experiment, "finetune")

    img_size = get_size(args.img_size)

    now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    args.experiment = osp.join(args.experiment, args.model + f"_epoch_{args.epochs}_batch_{args.batch_size}_lr_{args.lr}"
                               f"_warmup_{args.warmup_epochs}_seed_{args.seed}_size_{img_size[0]}x{img_size[1]}"
                               f"_optimizer_{args.optimizer}_scheduler_{args.scheduler}_{now}")

    # create weights dir
    os.makedirs(osp.join(args.experiment, "weights"), exist_ok=True)

    logger = setup_logger(output=args.experiment, name="train")
    for key, value in sorted(vars(args).items()):
        if is_main_process():
            logger.info(str(key) + ': ' + str(value))

    if args.summary and is_main_process():
        writer_dict = {
            "writer": SummaryWriter(log_dir=osp.join(args.experiment, "tensorboard")),
            "train_global_steps": 0,
            "val_global_steps": 0
        }
    else:
        writer_dict = None

    seed = args.seed + get_rank()
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    criterion = create_iou_bce_loss

    model = MPVOS(backbone=args.model, pretrained=args.pretrained)
    device = torch.device(args.device)
    model.to(device)

    # dataset
    train_dataset = create_dataset(args, is_train=True, datasets=args.train_datasets)
    val_dataset = create_dataset(args, is_train=False, datasets=args.val_datasets)
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = RandomSampler(train_dataset)
        val_sampler = SequentialSampler(val_dataset)

    train_batch_sampler = BatchSampler(train_sampler, args.batch_size, drop_last=True)
    train_data_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler,
                                   num_workers=8, pin_memory=True if args.device == "cuda" else False)
    val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, drop_last=False,
                                 num_workers=8, pin_memory=True if args.device == "cuda" else False)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    n_learn_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_parameters = sum(p.numel() for p in model.parameters())

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamW':
        optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        if args.distributed:
            args.lr = args.lr * args.batch_size * args.world_size / 256
        else:
            args.lr = args.lr * args.batch_size / 256
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise ValueError("Not support this optimizer: {}".format(args.optimizer))

    scheduler = get_scheduler(optimizer, len(train_data_loader), args)

    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint["model"], strict=True)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0 and is_main_process():
            logger.info('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0 and is_main_process():
            logger.info('Unexpected Keys: {}'.format(unexpected_keys))
        if is_main_process():
            logger.info("loading checkpoint from {} to finetune model.".format(args.finetune))

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0 and is_main_process():
            logger.info('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0 and is_main_process():
            logger.info('Unexpected Keys: {}'.format(unexpected_keys))
        if 'optimizer' in checkpoint and 'scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            if is_main_process():
                logger.info(optimizer.param_groups)
            scheduler.load_state_dict(checkpoint['scheduler'])
            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop and args.scheduler == "lr_step":
                logger.info('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                scheduler.step_size = args.lr_decay_epochs
                scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            scheduler.step(scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1
        if is_main_process():
            logger.info("loading checkpoint from {} to continue last training.".format(args.resume))

    if is_main_process():
        logger.info(model)
        logger.info("Random Seed: {}.".format(args.seed))
        logger.info("Epochs: {}.".format(args.epochs))
        logger.info("Image size: {} x {}.".format(img_size[0], img_size[1]))
        logger.info("Max norm: {}.".format(args.max_norm))
        logger.info("Number of trainable params (M): %.2f M." % (n_learn_parameters/1024/1024))
        logger.info('Number of total params (M): %.2f M.' % (n_parameters/1024/1024))
        logger.info("Learning rate: {}.".format(args.lr))
        logger.info("Optimizer: {}.".format(args.optimizer))
        logger.info("Warmup epochs: {}.".format(args.warmup_epochs))
        logger.info("Learning rate strategy: {}.".format(args.scheduler))
        logger.info("Train datasets: {}.".format(args.train_datasets))
        logger.info("Val datasets: {}.".format(args.val_datasets))
        logger.info("Train dataset sample number: {}.".format(len(train_dataset)))
        logger.info("Val dataset sample number: {}.".format(len(val_dataset)))
        logger.info('Start training.')

    tic = time.time()
    best_iou, best_epoch = 0., 0
    for epoch in range(args.start_epoch, args.epochs + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(args, logger, model, criterion, train_data_loader, optimizer, device, epoch,
                        writer_dict, scheduler)
        if epoch % args.eval_interval == 0:
            val_stats = evaluate(args, model, criterion, val_data_loader, device, epoch, writer_dict)

            save_files = {'model': model_without_ddp.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'scheduler': scheduler.state_dict(),
                          'args': args,
                          'epoch': epoch}

            if best_iou < val_stats["mIoU"]:
                best_iou = val_stats["mIoU"]
                best_epoch = epoch

            if args.save_best:
                if best_iou == val_stats["mIoU"]:
                    save_on_master(save_files, osp.join(args.experiment, "weights", f'{args.model}_best.pth'))
            else:
                save_on_master(save_files, osp.join(args.experiment, "weights", f'{args.model}_{epoch}.pth'))

            if epoch == args.epochs:
                save_on_master(save_files, osp.join(args.experiment, "weights", f'{args.model}_last.pth'))

            logger.info("best mIoU: {}, best epoch: {}".format(best_iou, best_epoch))

    if is_main_process():
        logger.info("finish training, cost training time in {} epochs: {:.2f}h".
                    format(args.epochs - args.start_epoch + 1, (time.time() - tic) / 3600))


if __name__ == '__main__':
    main()