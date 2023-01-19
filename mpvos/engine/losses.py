import torch


def _iou(pred, target):
    b = pred.shape[0]
    IoU = 0.0
    for i in range(0, b):
        # compute the IoU of the foreground
        Iand1 = torch.sum(target[i, :, :, :] * pred[i, :, :, :])
        Ior1 = torch.sum(target[i, :, :, :]) + torch.sum(pred[i, :, :, :]) - Iand1
        IoU1 = Iand1 / Ior1

        # IoU loss is (1-IoU1)
        IoU = IoU + (1 - IoU1)

    return IoU / b


class IOU(torch.nn.Module):
    def __init__(self):
        super(IOU, self).__init__()

    def forward(self, pred, target):
        return _iou(pred, target)


def create_iou_bce_loss(predict, target):
    bce_loss = torch.nn.BCEWithLogitsLoss()
    iou_loss = IOU()
    # target 0 or 1, predict not 0-1
    criterion = bce_loss(predict, target) + iou_loss(torch.sigmoid(predict), target)
    return criterion