import os
import os.path as osp
import torch
import argparse
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F

from torch.utils.data import DataLoader
from mpvos.utils import create_dataset
from mpvos.models import MPVOS


def get_argparse():

    parser = argparse.ArgumentParser("Test Data.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="choose using device")
    parser.add_argument("--img_size", type=int, nargs="+", default=512, help="training image size")
    parser.add_argument("--mean", type=list, default=[0.485, 0.456, 0.406], help="imagenet mean")
    parser.add_argument("--std", type=list, default=[0.229, 0.224, 0.225], help="imagenet std")
    parser.add_argument("--stride", type=int, default=1, help="sample train frames")
    parser.add_argument("--data-dir", type=str, default="/data/stk/UVOS_data/", help="dataset path")
    parser.add_argument("--save_dir", type=str, default="runs", help="save test data dir")
    parser.add_argument("--pretrained", action="store_true", default=False, help="use pretrained weights")
    parser.add_argument("--use_flip", action="store_true", default=False, help="use flip test")
    parser.add_argument("--multi_scale", action="store_true", default=False, help="multi_scale testing.")
    parser.add_argument("--scales", type=float, nargs="+", default=[0.5, 0.75, 1.0, 1.25, 1.5], help="multi scale ratio.")
    parser.add_argument("-v", "--val_datasets", type=str, nargs="+",
                        default=["DAVIS-2016"])
    parser.add_argument("--weights", type=str, nargs="+", default="")
    parser.add_argument("--model", type=str, default="resnet50", help="model name", choices=["resnet50", "resnet101"])

    return parser.parse_args()


def flip(x, dim):
    if x.is_cuda:
        # dim -> w dimension
        return torch.index_select(x, dim, torch.arange(x.size(dim) - 1, -1, -1).long().cuda())
    else:
        return torch.index_select(x, dim, torch.arange(x.size(dim) - 1, -1, -1).long())


def test():
    args = get_argparse()
    device = torch.device(args.device)
    model = MPVOS(backbone=args.model, pretrained=args.pretrained)

    model.to(device)
    model.load_state_dict(args.weights)

    if isinstance(args.val_datasets, str):
        val_datasets = [args.val_datasets]
    elif isinstance(args.val_datasets, list):
        val_datasets = args.val_datasets
    else:
        raise NotImplementedError

    with torch.no_grad():
        model.eval()
        if isinstance(args.scales, int):
            scales = [args.scales]
        else:
            scales = args.scales

        for val_dataset in val_datasets:
            dataset = create_dataset(args, is_train=False, datasets=val_dataset)
            data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False,
                                     pin_memory=True if args.device == "cuda" else False)
            with tqdm(total=len(data_loader)) as tbar:
                for image1, image2, mask, o_shape, mask_path in data_loader:

                    tbar.set_description(f"Dataset: {val_dataset}")

                    image1, image2 = image1.to(device), image2.to(device)
                    orin_w, orin_h = o_shape

                    scale_images = [(image1, image2)]
                    if args.multi_scale:
                        h, w = image1.shape[-2:]
                        for scale in scales:
                            if scale != 1.0:
                                size = (int(round(h * scale / 32) * 32), int(round(w * scale / 32) * 32))
                                image1_scale = F.interpolate(image1, size=size, mode="bilinear", align_corners=False)
                                image2_scale = F.interpolate(image2, size=size, mode="bilinear", align_corners=False)
                                scale_images.append((image1_scale, image2_scale))

                    predicts = []
                    for scale_image in scale_images:
                        predict = model(scale_image[0], scale_image[1]).sigmoid()
                        predict = F.interpolate(predict, size=(orin_h, orin_w), mode="bilinear", align_corners=False)
                        predicts.append(predict)

                        if args.use_flip:
                            # when have five dimensions tensor choose 4, four dimensions choose 3
                            predict_flip = model(flip(scale_image[0], dim=3), flip(scale_image[1], dim=3))

                            predict_flip = flip(predict_flip, dim=3).sigmoid()
                            predict_flip = F.interpolate(predict_flip, size=(orin_h, orin_w), mode="bilinear", align_corners=False)
                            predicts.append(predict_flip)

                    predict = torch.stack(predicts, dim=0).mean(0)

                    predict = predict.cpu().detach().numpy()

                    predict[predict >= 0.5] = 1
                    predict[predict < 0.5] = 0
                    predict = predict[0, 0, :, :] * 255
                    predict = Image.fromarray(predict).convert("L")
                    save_path = osp.join(args.save_dir, "test", val_dataset, mask_path[0].split("/")[-2])
                    os.makedirs(save_path, exist_ok=True)
                    save_file = osp.join(save_path, mask_path[0].split("/")[-1][:-4] + ".png")
                    predict.save(save_file)

                    tbar.update(1)


if __name__ == '__main__':
    test()
