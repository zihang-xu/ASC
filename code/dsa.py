import argparse
import logging
import os
import random
import sys
from tqdm import tqdm
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from transform import transform
from segresnet import SegResNet
from network import *
from dataloader import BaseDataSets
from val import test_single_volume
import time
import shutil
import utils
import ramps
import SimpleITK as sitk
from scipy import ndimage


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
                    default='dsa', help='Name of Experiment')
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
    model_dt = nn.DataParallel(Decoder(in_channels=256).cuda())
    model_ds = nn.DataParallel(Decoder(in_channels=256).cuda())
    model_gt = nn.DataParallel(FCDiscriminator(in_channels=1).cuda())
    model_gs = nn.DataParallel(FCDiscriminator(in_channels=1).cuda())
    model_gp = nn.DataParallel(FCDiscriminator(in_channels=8).cuda())

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
    model_ds.train()
    model_dt.train()
    model_gt.train()
    model_gs.train()
    model_gp.train()

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': model_ds.parameters(), 'lr': 1e-4},
        {'params': model_dt.parameters(), 'lr': 1e-4}], lr=base_lr)
    gan_lr3 = 1e-5
    gan_lr4 = 1e-5
    gan_lr5 = 1e-5
    optimizer3 = optim.Adam(model_gt.parameters(), lr=gan_lr3)
    optimizer4 = optim.Adam(model_gs.parameters(), lr=gan_lr4)
    optimizer5 = optim.Adam(model_gp.parameters(), lr=gan_lr5)

    bce_loss = nn.BCELoss()
    ce_loss = CrossEntropyLoss()
    dice_loss = utils.DiceLoss(num_classes)
    mse_loss = nn.MSELoss()

    logging.info("{} iterations per epoch".format(len(trainloader_t)))

    iter_num = 0
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    max_iterations = max_epoch * len(trainloader_t)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(zip(trainloader_s,trainloader_t)):
            #
            sampled_batch_s, sampled_batch_t = sampled_batch[0], sampled_batch[1]
            volume_batch, label_batch_s = torch.cat((sampled_batch_s['image'],sampled_batch_t['image'])), \
                                          sampled_batch_s['mask']
            volume_batch, label_batch_s = volume_batch.cuda(), label_batch_s.cuda()
            volume_batch_t = volume_batch[batch_size_half:]
            volume_batch_s = volume_batch[:batch_size_half]
            #
            ous = model(volume_batch_s)
            outputs,feature_s = ous['seg'],ous['enco']
            out = model(volume_batch_t)
            outputt,feature_t = out['seg'],out['enco']
            outputs_soft_s = torch.softmax(outputs, dim=1)
            outputs_soft_t = torch.softmax(outputt, dim=1)
            #
            fake_batch_ss,fake_batch_ts = model_ds(feature_s),model_ds(feature_t)
            fake_batch_st,fake_batch_tt = model_dt(feature_s),model_dt(feature_t)
            #
            outputs_fake = model(torch.cat((fake_batch_st,fake_batch_ts)))['seg']
            outputs_soft_fake = torch.softmax(outputs_fake, dim=1)
            outputs_soft_st, outputs_soft_ts = outputs_soft_fake[:batch_size_half], \
                                               outputs_soft_fake[batch_size_half:]
            #
            gan_target = torch.ones(batch_size,1).cuda()
            gan_target[batch_size_half:] = 0
            loss_adv_t = bce_loss(model_gt(fake_batch_st)['output'],gan_target[:batch_size_half])
            loss_adv_s = bce_loss(model_gs(fake_batch_ts)['output'],gan_target[:batch_size_half])
            loss_adv_p = bce_loss(model_gp(outputs_soft_t)['output'],gan_target[:batch_size_half]) + \
                         bce_loss(model_gp(outputs_soft_ts)['output'],gan_target[:batch_size_half])
            loss_re = mse_loss(fake_batch_ss,volume_batch_s) + \
                      mse_loss(fake_batch_tt,volume_batch_t)
            loss_dice = dice_loss(outputs_soft_s, label_batch_s) + \
                        dice_loss(outputs_soft_st, label_batch_s)
            loss = loss_dice + loss_re * 0.01 + loss_adv_t * 0.01 + loss_adv_s * 0.01 + loss_adv_p * 0.01
            #
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #
            with torch.no_grad():
                ous = model(volume_batch_s)
                outputs, feature_s = ous['seg'], ous['enco']
                out = model(volume_batch_t)
                outputt, feature_t = out['seg'], out['enco']
                outputs_soft_s = torch.softmax(outputs, dim=1)
                outputs_soft_t = torch.softmax(outputt, dim=1)
                outputs_soft = torch.cat((outputs_soft_s,outputs_soft_t))
                #
                fake_batch_ts = model_ds(feature_t)
                fake_batch_st = model_dt(feature_s)
                #
                outputs_fake = model(torch.cat((fake_batch_st, fake_batch_ts)))['seg']
                outputs_soft_fake = torch.softmax(outputs_fake, dim=1)
            optimizer3.zero_grad()
            loss_adv_t = bce_loss(model_gt(torch.cat((volume_batch_t,fake_batch_st)))['output'],gan_target)
            loss_adv_t.backward()
            optimizer3.step()
            #
            optimizer4.zero_grad()
            loss_adv_s = bce_loss(model_gs(torch.cat((volume_batch_s,fake_batch_ts)))['output'],gan_target)
            loss_adv_s.backward()
            optimizer4.step()
            #
            optimizer5.zero_grad()
            loss_adv_p = bce_loss(model_gp(outputs_soft)['output'],gan_target) + \
                         bce_loss(model_gp(outputs_soft_fake)['output'],gan_target)
            loss_adv_p.backward()
            optimizer5.step()
            #
            lr_ = base_lr * (1.0 - (epoch_num+1) / max_epoch) ** 0.9
            lr_g3 = gan_lr3 * (1.0 - (iter_num + 1) / max_iterations) ** 0.9
            lr_g4 = gan_lr4 * (1.0 - (iter_num + 1) / max_iterations) ** 0.9
            lr_g5 = gan_lr5 * (1.0 - (iter_num + 1) / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer3.param_groups:
                param_group['lr'] = lr_g3
            for param_group in optimizer4.param_groups:
                param_group['lr'] = lr_g4
            for param_group in optimizer5.param_groups:
                param_group['lr'] = lr_g5

            iter_num = iter_num + 1
            logging.info(
                'iteration %d : loss : %f loss_dice : %f loss_re : %f loss_advt : %f loss_advs : %f loss_advp : %f' %
                (iter_num, loss.item(), loss_dice.item(), loss_re.item(), loss_adv_t.item(), loss_adv_s.item(), loss_adv_p.item()))

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


