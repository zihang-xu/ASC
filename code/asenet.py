import argparse
import logging
import os
import random
import sys
from tqdm import tqdm
import math
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import transforms
from transform import transform
from network import UNet
from network import FCDiscriminator
from segresnet import SegResNet,SegResNetVAE,SegResNet1
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
                    default='asenet', help='Name of Experiment')
parser.add_argument('--num_classes', type=int,  default=8,
                    help='output channel of network')
parser.add_argument('--max_epoch', type=int,
                    default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--patch_size', type=list,  default=[144,144,144],
                    help='patch size of network input')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.0001,
                    help='segmentation network learning rate')
parser.add_argument('--gan_lr', type=float,  default=0.0001,
                    help='discriminator network learning rate')
parser.add_argument('--epoch_gap', type=int, default=5,
                    help='choose epoch gap to val model')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0,1,2,3', help='GPU to use')
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=10, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
parser.add_argument('--zoom', type=int, default=1,
                    help='whether use zoom training')
parser.add_argument('--crop', type=int, default=1,
                    help='whether use crop training')
args = parser.parse_args()


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(args,snapshot_path,exp_path):
    base_lr = args.base_lr
    gan_lr = args.gan_lr
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
    ema_model = nn.DataParallel(create_model(ema=True))
    model_g1= FCDiscriminator(args.num_classes).cuda()
    model_g1 = nn.DataParallel(model_g1)
    model_g2 = FCDiscriminator(args.num_classes).cuda()
    model_g2 = nn.DataParallel(model_g2)

    db_train_t = BaseDataSets(data_dir=args.root_path_t, mode="train", list_name=args.train_data_t,
                              patch_size=args.patch_size, crop=args.crop, zoom=args.zoom,
                              transform=None)
    db_train_s = BaseDataSets(data_dir=args.root_path_s, mode="train", list_name=args.train_data_s,
                              patch_size=args.patch_size, crop=args.crop, zoom=args.zoom,
                              transform=None)
    db_val = BaseDataSets(data_dir=args.root_path_t, mode="test", list_name=args.test_data,
                          patch_size=args.patch_size,crop=args.crop,zoom=args.zoom)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    batch_size_half = int(batch_size / 2)
    trainloader_t = DataLoader(db_train_t, batch_size=batch_size_half, shuffle=True,
                             num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    trainloader_s = DataLoader(db_train_s, batch_size=batch_size_half, shuffle=True,
                             num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    model.train()
    model_g1.train()
    model_g2.train()

    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    optimizer_g1 = optim.Adam(model_g1.parameters(), lr=gan_lr)
    optimizer_g2 = optim.Adam(model_g2.parameters(), lr=gan_lr)
    dice_loss = DiceLoss(num_classes)
    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()

    logging.info("{} iterations per epoch".format(len(trainloader_t)))

    iter_num = 0
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    max_iterations = max_epoch * len(trainloader_t)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(zip(trainloader_s,trainloader_t)):
            # T1 = time.time()
            #
            sampled_batch_s, sampled_batch_t = sampled_batch[0], sampled_batch[1]
            volume_batch, label_batch_s = torch.cat((sampled_batch_s['image'],sampled_batch_t['image'])), \
                                          sampled_batch_s['mask']
            volume_batch, label_batch_s = volume_batch.cuda(), label_batch_s.cuda()
            volume_batch_s = volume_batch[:batch_size_half]
            volume_batch_t = volume_batch[batch_size_half:]
            noise = torch.clamp(torch.randn_like(volume_batch_t) * 0.1, -0.2, 0.2)
            ema_inputs = volume_batch_t + noise

            #
            outputs_soft = torch.softmax(model(volume_batch)['seg'], dim=1)
            loss_dice = dice_loss(outputs_soft[:batch_size_half], label_batch_s)

            #
            with torch.no_grad():
                ema_output = torch.softmax(ema_model(ema_inputs)['seg'], dim=1)
            output_noise = torch.softmax(model(ema_inputs)['seg'], dim=1)
            consistency_loss = mse_loss(ema_output, outputs_soft[batch_size_half:]) + \
                               mse_loss(output_noise, outputs_soft[batch_size_half:]) + \
                               mse_loss(output_noise, ema_output)

            #
            gan_loss1 = model_g1(outputs_soft[batch_size_half:],volume_batch_t)['output']
            gan_loss2 = model_g2(output_noise,volume_batch_t)['output']
            target = torch.ones(batch_size, 1).cuda()
            target[batch_size_half:] = 0
            gan_loss = bce_loss(gan_loss1, target[:batch_size_half]) * 0.5 + \
                       bce_loss(gan_loss2, target[:batch_size_half]) * 0.5

            #
            consistency_weight = get_current_consistency_weight(epoch_num)
            loss = loss_dice + gan_loss * 0.01 + consistency_loss * consistency_weight
            #
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            optimizer_g1.zero_grad()
            with torch.no_grad():
                outputs_soft = torch.softmax(model(volume_batch)['seg'], dim=1)
                ema_output = torch.softmax(ema_model(volume_batch_t)['seg'], dim=1)
                output_noise = torch.softmax(model(ema_inputs)['seg'], dim=1)
            gan_loss1 = model_g1(outputs_soft[:batch_size_half],volume_batch_s)['output']
            gan_loss2 = model_g1(outputs_soft[batch_size_half:],volume_batch_t)['output']
            g_loss = bce_loss(torch.cat((gan_loss1,gan_loss2)), target)
            g_loss.backward()
            optimizer_g1.step()

            optimizer_g2.zero_grad()
            gan_loss1 = model_g2(ema_output,volume_batch_t)['output']
            gan_loss2 = model_g2(output_noise,volume_batch_t)['output']
            g_loss = bce_loss(torch.cat((gan_loss1,gan_loss2)), target)
            g_loss.backward()
            optimizer_g2.step()

            lr_ = base_lr * (1.0 - (iter_num + 1) / max_iterations) ** 0.9
            lr_g = gan_lr * (1.0 - (iter_num + 1) / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer_g1.param_groups:
                param_group['lr'] = lr_g
            for param_group in optimizer_g2.param_groups:
                param_group['lr'] = lr_g

            iter_num = iter_num + 1
            logging.info(
                'iteration %d : loss : %f loss_dice : %f loss_gan : %f loss_con : %f' %
                (iter_num, loss.item(), loss_dice.item(), gan_loss.item(), consistency_loss.item()))

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



