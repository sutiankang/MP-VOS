import torch
import math
import sys
from tqdm import tqdm
import random
import torch.nn.functional as F

from mpvos.metrics import db_eval_iou_multi
from mpvos.utils import is_main_process, reduce_value, get_total_grad_norm, AverageMeter


def train_one_epoch(args, logger, model, criterion, train_loader, optimizer, device,
                    epoch, writer_dict, scheduler):

    model.train()

    loss_record = AverageMeter()
    lr_record = AverageMeter()
    iou_record = AverageMeter()
    grad_record = AverageMeter()

    if isinstance(args.scales, float):
        scales = [args.scales]
    else:
        scales = args.scales

    with tqdm(total=len(train_loader)) as tbar:
        for image1, image2, mask, _, _ in train_loader:

            tbar.set_description(f"Train: Epoch [{epoch}/{args.epochs}]")

            image1, image2, mask = image1.to(device), image2.to(device), mask.to(device)

            h, w = image1.shape[-2:]
            if args.multi_scale:
                scale = random.choice(scales)
                if scale != 1.0:
                    h = int(round(h * scale / 32) * 32)
                    w = int(round(w * scale / 32) * 32)
                    image1 = F.interpolate(image1, size=(h, w), mode="bilinear", align_corners=False)
                    image2 = F.interpolate(image2, size=(h, w), mode="bilinear", align_corners=False)
                    mask = F.interpolate(mask, size=(h, w), mode="nearest")

            output = model(image1, image2)
            # output no sigmoid
            losses = criterion(output, mask)
            loss_value = reduce_value(losses, average=True).item()

            if not math.isfinite(loss_value) and is_main_process():
                logger.info("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            torch.cuda.synchronize()

            if args.max_norm > 0:
                grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
            else:
                grad_total_norm = get_total_grad_norm(model.parameters(), args.max_norm)

            # mask 0-1, output need 0-1
            precise_output = torch.sigmoid(output).cpu().detach().numpy()
            precise_output[precise_output >= 0.5] = 1
            precise_output[precise_output < 0.5] = 0

            iou = db_eval_iou_multi(mask.cpu().detach().numpy(), precise_output)
            iou = reduce_value(torch.tensor(iou, device=device), average=True).item()
            # Each Batch Update
            scheduler.step()

            loss_record.update(loss_value)
            grad_record.update(grad_total_norm.detach().cpu().numpy())
            iou_record.update(iou)
            lr_record.update(optimizer.param_groups[0]["lr"])

            # update bar
            tbar.set_postfix(size=(h, w), loss=loss_record.average, lr=lr_record.value, mIoU=iou_record.average,
                             grad_norm=grad_record.average)
            tbar.update(1)

            if args.summary and is_main_process():
                writer_dict["writer"].add_scalar("train_loss", loss_record.average,
                                                 writer_dict["train_global_steps"])
                writer_dict["writer"].add_scalar("learning_rate", optimizer.param_groups[0]["lr"],
                                                 writer_dict["train_global_steps"])
                writer_dict["writer"].add_scalar("train_mean_iou", iou_record.average,
                                                 writer_dict["train_global_steps"])
                writer_dict['train_global_steps'] += 1


@torch.no_grad()
def evaluate(args, model, criterion, val_loader, device, epoch, writer_dict):
    torch.cuda.empty_cache()
    model.eval()

    loss_record = AverageMeter()
    iou_record = AverageMeter()

    with tqdm(total=len(val_loader)) as tbar:
        for image1, image2, mask, _, _ in val_loader:

            tbar.set_description(f"Test: Epoch [{epoch}/{args.epochs}]")

            image1, image2, mask = image1.to(device), image2.to(device), mask.to(device)
            output = model(image1, image2)
            losses = criterion(output, mask)
            loss_value = reduce_value(losses, average=True).item()

            # mask 0-1 output need 0-1
            precise_output = torch.sigmoid(output).cpu().detach().numpy()
            precise_output[precise_output >= 0.5] = 1
            precise_output[precise_output < 0.5] = 0

            iou = db_eval_iou_multi(mask.cpu().detach().numpy(), precise_output)
            iou = reduce_value(torch.tensor(iou, device=device), average=True).item()

            loss_record.update(loss_value)
            iou_record.update(iou)
            # update bar
            tbar.set_postfix(loss=loss_record.average, mIoU=iou_record.average)
            tbar.update(1)

            if args.summary and is_main_process():
                writer_dict["writer"].add_scalar("valid_loss", loss_record.average,
                                                 writer_dict["val_global_steps"])
                writer_dict["writer"].add_scalar("valid_mean_iou", iou_record.average,
                                                 writer_dict["val_global_steps"])
                writer_dict['val_global_steps'] += 1

        output_dicts = {"loss": loss_record.average, "mIoU": iou_record.average}

    return output_dicts