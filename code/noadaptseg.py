import argparse
import logging
import os
import random
from tqdm import tqdm
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from segresnet import SegResNet
from dataloader import BaseDataSets
from val import test_single_volume
import data_augment
import time
import shutil
from scipy import ndimage
import SimpleITK as sitk
from utils import *
import ramps
import torch.nn.functional as F


parser = argparse.ArgumentParser()
parser.add_argument('--root_path_t', type=str,
                    default='/mntnfs/med_data5/xuzihang/feta2021', help='Name of Experiment')
parser.add_argument('--root_path_s', type=str,
                    default='/mntnfs/med_data5/xuzihang/atlases', help='Name of Experiment')
parser.add_argument('--train_data_s', type=str,
                    default='train.list', help='Name of Dataset')
parser.add_argument('--train_data_t', type=str,
                    default='train.list', help='Name of Dataset')
parser.add_argument('--test_data', type=str,
                    default='test.list', help='Name of Dataset')
parser.add_argument('--exp', type=str,
                    default='noadaptseg', help='Name of Experiment')
parser.add_argument('--num_classes', type=int, default=8,
                    help='output channel of network')
parser.add_argument('--max_epoch', type=int,
                    default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=2,
                    help='batch_size per gpu')
parser.add_argument('--patch_size', type=list,  default=[144,144,144],
                    help='patch size of network input')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.0001,
                    help='segmentation network learning rate')
parser.add_argument('--epoch_gap', type=int, default=5,
                    help='choose epoch gap to val model')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0,1,2,3', help='GPU to use')
parser.add_argument('--zoom', type=int, default=1,
                    help='whether use zoom training')
parser.add_argument('--crop', type=int, default=1,
                    help='whether use crop training')
args = parser.parse_args()


def train(args,snapshot_path,exp_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_epoch = args.max_epoch

    def create_model(ema=False):
        # Network definition
        model = SegResNet(in_channels=1, out_channels=args.num_classes, init_filters=32).cuda()

        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = nn.DataParallel(create_model())

    db_train = BaseDataSets(data_dir=args.root_path_s, mode="train", list_name=args.train_data_s,
                            patch_size=args.patch_size, crop=args.crop, zoom=args.zoom,
                            transform=None)
    db_val = BaseDataSets(data_dir=args.root_path_t, mode="test", list_name=args.test_data,
                          patch_size=args.patch_size,crop=args.crop,zoom=args.zoom)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    model.train()

    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    dice_loss = DiceLoss(num_classes)

    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    max_iterations = max_epoch * len(trainloader)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            #
            volume_batch, label_batch = sampled_batch['image'],sampled_batch['mask']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            outputs = model(volume_batch)['seg']
            outputs_soft = torch.softmax(outputs, dim=1)
            loss_dice = dice_loss(outputs_soft, label_batch)
            loss = loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - (iter_num+1) / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            logging.info(
                'iteration %d : loss : %f loss_dice : %f' %
                (iter_num, loss.item(), loss_dice.item()))

            if iter_num % (max_iterations * 0.2) == 0:
                model.eval()
                save_mode_path = os.path.join(exp_path, 'iter_num_{}.pth'.format(iter_num))
                torch.save(model.state_dict(), save_mode_path)
                model.train()

    return "Training Finished!"


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

    snapshot_path = "/home/xuzihang/miccai/" + args.exp
    exp_path = '/mntnfs/med_data5/xuzihang/miccai/' + args.exp
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

    train_name = args.exp + '.py'
    shutil.copy(train_name, code_path + '/' + train_name)
    shutil.copy('segresnet.py', code_path + '/' + 'segresnet.py')
    shutil.copy('utils.py', code_path + '/' + 'utils.py')
    shutil.copy('network.py', code_path + '/' + 'network.py')
    shutil.copy('dataloader.py', code_path + '/' + 'dataloader.py')
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


