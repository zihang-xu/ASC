import argparse
import logging
import os
import random
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from dataloader import BaseDataSets
import torch
import torch.nn as nn
from segresnet import SegResNet,conv_block
from medpy import metric
import numpy as np
import torch.nn.functional as F
import argparse
import logging
import os
import random
import sys
from tqdm import tqdm
import math
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import BaseDataSets
from utils import *
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--root_path_t', type=str,
                    default='/mntnfs/med_data5/xuzihang/feta2021', help='Name of Experiment')
parser.add_argument('--root_path_s', type=str,
                    default='/mntnfs/med_data5/xuzihang/registered0', help='Name of Experiment')
parser.add_argument('--train_data', type=str,
                    default='train.list', help='Name of Dataset')
parser.add_argument('--test_data', type=str,
                    default='test.list', help='Name of Dataset')
parser.add_argument('--exp', type=str,
                    default='scale', help='Name of Experiment')
parser.add_argument('--num_classes', type=int,  default=8,
                    help='output channel of network')
parser.add_argument('--patch_size', type=list,  default=[144,144,144],
                    help='patch size of network input')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0,1,2,3', help='GPU to use')
args = parser.parse_args()


def create_nonzero_mask(data):
    from scipy.ndimage import binary_fill_holes
    assert len(data.shape) == 4 or len(data.shape) == 3, "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    nonzero_mask = np.zeros(data.shape[1:], dtype=bool)
    for c in range(data.shape[0]):
        this_mask = data[c] != 0
        nonzero_mask = nonzero_mask | this_mask
    nonzero_mask = binary_fill_holes(nonzero_mask)
    return nonzero_mask


def get_bbox_from_mask(mask, outside_value=0):
    mask_voxel_coords = np.where(mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]


def crop_to_bbox(image, bbox):
    assert len(image.shape) == 3, "only supports 3d images"
    resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
    return image[resizer]


def train(args, snapshot_path, exp_path):
    num_classes = args.num_classes

    db_train = BaseDataSets(data_dir=args.root_path_s, mode="train", list_name='train.list',
                            crop=True,zoom=True)
    trainloader = DataLoader(db_train, batch_size=1, shuffle=False, num_workers=1)

    model = conv_block(in_channels=1, out_channels=1).cuda()
    model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CosineSimilarity()
    import torch.nn.functional as F

    for i_batch, sampled_batch in enumerate(trainloader):
        label = sampled_batch['mask'].cuda()
        volume = sampled_batch['image'].cuda()
        label = label.squeeze(0).unsqueeze(1)
        volume = volume.squeeze(0).unsqueeze(1)
        if volume.shape[0] == 1:
            continue
        #
        volume_patch = volume.reshape(volume.shape[0]*1728, 1, 12, 12, 12)
        label_patch = label.reshape(volume.shape[0]*1728, 1, 12, 12, 12)
        feature_patch = model(volume_patch)
        feature_patch = nn.functional.normalize(feature_patch, dim=1)
        class_patch = label_patch.reshape(label_patch.shape[0],-1).mode(-1)[0]
        this_classes = torch.unique(class_patch)
        loss = 0.
        for c in this_classes:
            index1 = torch.where(class_patch[:1728] == c)[0]
            index2 = torch.where(class_patch[1728:] == c)[0]
            class_feature1 = torch.index_select(feature_patch, dim=0, index=index1)
            class_feature2 = torch.index_select(feature_patch, dim=0, index=index2)
            perm = min(class_feature1.shape[0],class_feature2.shape[0])
            if perm < 2:
                continue
            loss_c = criterion(class_feature1[:perm],class_feature2[:perm].detach()) + \
                     criterion(class_feature2[:perm],class_feature1[:perm].detach())
            loss += 1 - loss_c.mean()
            # print(perm)
            print(loss)
        loss /= len(this_classes)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print( i_batch)

    model.eval()
    save_mode_path = os.path.join(exp_path, 'iter_num.pth')
    torch.save(model.state_dict(), save_mode_path)
    print("Training Finished!")

    model.eval()
    # save_mode_path = '/mntnfs/med_data5/xuzihang/miccai2023/scale/iter_num.pth'
    model.load_state_dict(torch.load(save_mode_path))

    metric_lists = 0.0
    db_train = BaseDataSets(data_dir=args.root_path_s, mode="test", list_name='test50.list',
                            crop=True,zoom=True)
    db_val = BaseDataSets(data_dir=args.root_path_t, mode="test", list_name='test.list',
                          crop=True,zoom=True)

    trainloader = DataLoader(db_train, batch_size=1, shuffle=False, num_workers=1)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)
    for i_batch, sampled_batch in enumerate(zip(trainloader,valloader)):

        sampled_batch_s, sampled_batch_t = sampled_batch[0], sampled_batch[1]
        label = sampled_batch_t['mask'].squeeze(0).squeeze(0).cpu().detach().numpy()
        registered = sampled_batch_s['mask'].squeeze(0).cpu().detach().numpy()
        volume = sampled_batch_t['image'].cuda()#.squeeze(0).cpu().detach().numpy()
        atlas = sampled_batch_s['image'].cuda()#.squeeze(0).cpu().detach().numpy()
        feature_anchor = model(volume.reshape(1728, 1, 12, 12, 12).float())
        feature_anchor = F.normalize(feature_anchor, dim=1)
        similarity_base = -1
        index = -1
        for i in range(registered.shape[0]):
            feature = model(atlas[:,i].reshape(1728, 1, 12, 12, 12).float())
            feature = F.normalize(feature, dim=1)
            similarity = criterion(feature_anchor,feature).mean()
            if similarity > similarity_base:
                similarity_base = similarity
                index = i
        print(index)
        # index = -1
        registered_label = registered[index]
        def calculate_metric_percase(pred, gt):
            pred[pred > 0] = 1
            gt[gt > 0] = 1
            dice = metric.binary.dc(pred, gt)
            return dice
        metric_list = []
        for i in range(1, num_classes):
            metric_list.append(calculate_metric_percase(
                registered_label == i, label == i))
        print(np.mean(metric_list, axis=0))
        metric_lists += np.array(metric_list)
    metric_lists = metric_lists / 40
    print('------------------------')
    print('Class Dice : ', metric_lists)
    performance = np.mean(metric_lists, axis=0)
    print('------------------------')
    print('Mean Dice : ',performance)

    return "Testing Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    snapshot_path = "/home/xuzihang/miccai2023/" + args.exp
    exp_path = '/mntnfs/med_data5/xuzihang/miccai2023/' + args.exp
    timestamp = str(int(time.time()))
    snapshot_path = os.path.join(snapshot_path, 'log_' + timestamp)
    exp_path = os.path.join(exp_path, 'log_' + timestamp)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    code_path = os.path.join(snapshot_path, 'code')
    if not os.path.exists(code_path):
        os.makedirs(code_path)
    #
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log等级总开关
    # 第二步，创建一个handler，用于写入日志文件
    logfile = snapshot_path + '/log.txt'
    fh = logging.FileHandler(logfile, mode='a')  # open的打开模式这里可以进行参考
    fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
    # 第三步，再创建一个handler，用于输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)  # 输出到console的log等级的开关
    # 第四步，定义handler的输出格式
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # 第五步，将logger添加到handler里面
    logger.addHandler(fh)
    logger.addHandler(ch)

    logging.info(str(args))
    train(args, snapshot_path, exp_path)



