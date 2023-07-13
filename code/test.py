import torch
import SimpleITK as sitk
import numpy as np
import nibabel as nib
from network import UNet
import torch.nn as nn
import argparse
import os
from dataloader import BaseDataSets
from torch.utils.data import DataLoader
from scipy import ndimage
from segresnet import SegResNet,SegResNetVAE
from val import calculate_metric_percase


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/mntnfs/med_data5/xuzihang/feta2021', help='Name of Experiment')
parser.add_argument('--save_mode_path', type=str,
                    default='/mntnfs/med_data5/xuzihang/feta2021', help='Save path of model')
parser.add_argument('--exp', type=str,
                    default='ASC', help='Name of Experiment')
parser.add_argument('--num_classes', type=int,  default=8,
                    help='output channel of network')
parser.add_argument('--patch_size', type=list,  default=[144,144,144],
                    help='patch size of network input')
parser.add_argument('--zoom', type=int, default=1,
                    help='whether use zoom training')
parser.add_argument('--crop', type=int, default=1,
                    help='whether use crop training')
parser.add_argument('--gpu', type=str, default='0,1,2,3', help='GPU to use')
args = parser.parse_args()

model = SegResNet(in_channels=1, out_channels=args.num_classes, init_filters=32)
if args.exp == 'olva':
    model = SegResNetVAE(in_channels=1, out_channels=args.num_classes, init_filters=32,
                                 input_image_size=args.patch_size, vae_estimate_std=True)

model = nn.DataParallel(model).cuda()
test_save_path = args.save_mode_path[:args.save_mode_path.find('iter')] + 'log'
model.load_state_dict(torch.load(args.save_mode_path))
model.eval()
db_val = BaseDataSets(data_dir=args.root_path, mode="test", list_name='test.list'
                      , patch_size=args.patch_size, crop=args.crop, zoom=args.zoom)
valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

def test():
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)

    def metric_():
        import torchmetrics
        metric_list = torchmetrics.functional.dice(torch.from_numpy(out).cuda(),
                                                   torch.from_numpy(label).cuda(),
                                                   average='none', num_classes=8)
        return metric_list[1:]
    metric_list = 0.0
    metric_list1 = 0.0
    metric_list2 = 0.0
    for i_batch, sampled_batch in enumerate(valloader):
        volume_batch, label, case = \
            sampled_batch['image'], sampled_batch['mask'], sampled_batch['idx']
        label = label.squeeze(1).squeeze(0).cpu().detach().numpy()
        with torch.no_grad():
            out = model(volume_batch.cuda())['seg']
            out = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            if args.crop:
                bbox = sampled_batch['bbox']
                out = ndimage.zoom(out, zoom=
                ((int(bbox[0][1]) - int(bbox[0][0])) / (out.shape[0]),
                 (int(bbox[1][1]) - int(bbox[1][0])) / (out.shape[1]),
                 (int(bbox[2][1]) - int(bbox[2][0])) / (out.shape[2])), order=0)
                out = np.pad(out, ((int(bbox[0][0]), label.shape[0] - int(bbox[0][1])),
                                   (int(bbox[1][0]), label.shape[1] - int(bbox[1][1])),
                                   (int(bbox[2][0]), label.shape[2] - int(bbox[2][1]))))
            else:
                out = ndimage.zoom(out, zoom=
                (label.shape[0] / (out.shape[0]),
                label.shape[1] / (out.shape[1]),
                label.shape[2] / (out.shape[2])), order=0)
        #
        metric_i = metric_()
        with open(test_save_path + '/results.txt', 'a') as f:
            f.write('test_case: ' + str(i_batch) + '\n')
            f.write('mean_dice: ' + str(metric_i.mean()) + '\n')
        print('test_case %d : mean_dice : %f' % (i_batch, metric_i.mean()))
        metric_list += metric_i
        if i_batch < 15:
            metric_list1 += metric_i
        else:
            metric_list2 += metric_i

    metric_list = metric_list / len(db_val)
    metric_list1 = metric_list1 / 15
    metric_list2 = metric_list2 / 25
    performance = metric_list.mean()
    performance1 = metric_list1.mean()
    performance2 = metric_list2.mean()
    with open(test_save_path + '/results.txt', 'a') as f:
        num = 1
        for file in metric_list:
            f.write('class: ' + str(num) + '\n')
            f.write('dice: ' + str(file) + '\n')
            num += 1
        f.write(str(performance) + '\n')
        f.write(str(performance1) + '\n')
        f.write(str(performance2) + '\n')
    print('Total mean_dice : ', performance)
    print('Total mean_dice1 : ', performance1)
    print('Total mean_dice2 : ', performance2)
    print('Finished!')

if __name__ == "__main__":
    test()