import random
from operator import attrgetter
import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import os
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
import skimage


def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def dice_loss1(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def entropy_loss(p, C=2):
    # p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1) / \
        torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent


def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1*torch.sum(p * torch.log(p + 1e-6), dim=1,
                       keepdim=True)/torch.tensor(np.log(C)).cuda()
    return ent


def softmax_mse_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_softmax = torch.sigmoid(input_logits)
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss


def softmax_kl_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_log_softmax = torch.log(torch.sigmoid(input_logits))
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_log_softmax = F.log_softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='mean')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div


def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2)**2)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def entropy_minmization(p):
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1)
    ent = torch.mean(y1)

    return ent


def entropy_map(p):
    ent_map = -1*torch.sum(p * torch.log(p + 1e-6), dim=1,
                           keepdim=True)
    return ent_map


def get_loss(C_fake, C_real, loss_type, net='G'):
    """
    Returns the generator and discriminator losses for the given loss type.
        Relativistic generator losses use discriminator output for the real samples.
        Relativistic average losses use the average of discriminator outputs for both real and fake samples.
        Pre-calculated gradient penalty term is added to the discriminator losses using gradient penalty.
    """

    if(loss_type == "sgan"):

        loss = nn.BCEWithLogitsLoss()
        if net=='D':
            ones = torch.ones_like(C_real)
            zeros = torch.zeros_like(C_fake)
            return (loss(C_real,ones) + loss(C_fake,zeros))
        elif net=='G':
            ones = torch.ones_like(C_fake)
            return loss(C_fake,ones)

    elif(loss_type == "rsgan"):

        loss = nn.BCEWithLogitsLoss()
        if net=='D':
            ones = torch.ones_like(C_fake)
            return loss((C_real-C_fake),ones)
        elif net=='G':
            ones = torch.ones_like(C_fake)
            return loss((C_fake-C_real),ones)

    elif(loss_type == "rasgan"):

        loss = nn.BCEWithLogitsLoss()
        if net=='D':
            ones = torch.ones_like(C_real)
            zeros = torch.zeros_like(C_fake)
            return (loss((C_real-C_fake.mean()),ones) + loss((C_fake-C_real.mean()),zeros))
        elif net=='G':
            ones = torch.ones_like(C_fake)
            zeros = torch.zeros_like(C_real)
            return (loss((C_real-C_fake.mean()),zeros) + loss((C_fake-C_real.mean()),ones))

    elif(loss_type == "lsgan"):

        loss = nn.MSELoss()
        if net=='D':
            ones = torch.ones_like(C_fake)
            zeros = torch.zeros_like(C_real)
            return (loss(C_real, zeros) + loss(C_fake, ones))

        elif net=='G':
            zeros = torch.zeros_like(C_fake)
            return loss(C_fake,zeros)

    elif(loss_type == "ralsgan"):

        loss = nn.MSELoss()

        if net=='D':
            ones = torch.ones_like(C_fake)
            return (loss((C_real-C_fake.mean()), ones) + loss((C_fake-C_real.mean()), -ones))
        elif net=='G':
            ones = torch.ones_like(C_fake)
            return (loss((C_fake-C_real.mean()), ones) + loss((C_real-C_fake.mean()),-ones))

    elif(loss_type == "hingegan"):
        if net=='D':
            ones = torch.ones_like(C_fake)
            return (torch.clamp((ones-C_real), min=0).mean() + torch.clamp((C_fake+ones), min=0).mean())
        elif net=='G':
            return -C_fake.mean()

    elif(loss_type == "rahingegan"):
        if net=='D':
            ones = torch.ones_like(C_fake)
            return (torch.clamp((ones - C_real + C_fake.mean()), min=0).mean() +
                    torch.clamp((ones + C_fake - C_real.mean()), min=0).mean())
        elif net=='G':
            ones = torch.ones_like(C_fake)
            return (torch.clamp((ones - C_fake + C_real.mean()), min=0).mean() +
                    torch.clamp((ones + C_real - C_fake.mean()), min=0).mean())

    elif(loss_type == "wgan"):
        if net=='D':

            return (-C_real.mean() + C_fake.mean())
        elif net=='G':

            return -C_fake.mean()

    elif(loss_type == "rwgan"):
        if net=='D':

            return (-C_real.mean() + C_fake.mean())
        elif net=='G':

            return -(-C_real.mean() + C_fake.mean())


class DistBinaryDiceLoss(nn.Module):
    """
    Distance map penalized Dice loss
    Motivated by: https://openreview.net/forum?id=B1eIcvS45V
    Distance Map Loss Penalty Term for Semantic Segmentation
    """

    def __init__(self, smooth=1e-5):
        super(DistBinaryDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, net_output, gt):
        """
        net_output: (batch_size, 2, x,y,z)
        target: ground truth, shape: (batch_size, 1, x,y,z)
        """
        net_output = softmax_helper(net_output)
        # one hot code for gt
        with torch.no_grad():
            if len(net_output.shape) != len(gt.shape):
                gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(net_output.shape)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, 1)

        gt_temp = gt[:, 0, ...].type(torch.float32)
        with torch.no_grad():
            dist = compute_edts_forPenalizedLoss(gt_temp.cpu().numpy() > 0.5) + 1.0
        # print('dist.shape: ', dist.shape)
        dist = torch.from_numpy(dist)

        if dist.device != net_output.device:
            dist = dist.to(net_output.device).type(torch.float32)

        tp = net_output * y_onehot
        tp = torch.sum(tp[:, 1, ...] * dist, (1, 2, 3))

        dc = (2 * tp + self.smooth) / (torch.sum(net_output[:, 1, ...], (1, 2, 3)) + torch.sum(y_onehot[:, 1, ...],
                                                                                               (1, 2, 3)) + self.smooth)

        dc = dc.mean()

        return -dc


def compute_sdf(img_gt, out_shape):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1,1]
    """

    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]): # batch size
        posmask = img_gt[b].astype(np.bool)
        if posmask.any():
            negmask = ~posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
            sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
            sdf[boundary==1] = 0
            normalized_sdf[b] = sdf
            # assert np.min(sdf) == -1.0, print(np.min(posdis), np.max(posdis), np.min(negdis), np.max(negdis))
            # assert np.max(sdf) ==  1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))

    return normalized_sdf


def get_max_con(mask):
    import cc3d
    batch = mask.shape[0]
    map = np.zeros_like(mask)
    for b in range(batch):
        cc_seg, max_labels_count = cc3d.connected_components((mask[b] > 0).astype("uint8"), return_N=True)
        max_label = 1
        max_counts = 0
        # second_label = 1
        # second_counts = 0
        for label in range(1, max_labels_count + 1):
            curr_sum = np.sum(cc_seg == label)
            if curr_sum > max_counts:
                max_counts = curr_sum
                max_label = label
            # elif curr_sum > second_counts:
            #     second_counts = curr_sum
            #     second_label = label
        # if 2 * second_counts < max_counts:
        map[b] = (cc_seg == max_label).astype("uint8")
        # else:
        #     map[b] = ((cc_seg == max_label) & (cc_seg == second_label)).astype("uint8")
    return map

class MMDLoss(nn.Module):
    '''
    计算源域数据和目标域数据的MMD距离
    Params:
    source: 源域数据（n * len(x))
    target: 目标域数据（m * len(y))
    kernel_mul:
    kernel_num: 取不同高斯核的数量
    fix_sigma: 不同高斯核的sigma值
    Return:
    loss: MMD loss
    '''
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)/len(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        delta = f_of_X.float() - f_of_Y.float()
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss


class MaskGenerator(object):
    """
    Mask Generator
    """

    def generate_params(self, n_masks, mask_shape, rng=None):
        raise NotImplementedError('Abstract')

    def append_to_batch(self, *batch):
        x = batch[0]
        params = self.generate_params(len(x), x.shape[2:4])
        return batch + (params,)

    def torch_masks_from_params(self, t_params, mask_shape, torch_device):
        raise NotImplementedError('Abstract')


def gaussian_kernels(sigma, max_sigma=None, truncate=4.0):
    """
    Generate multiple 1D gaussian convolution kernels
    :param sigma: values for sigma as a `(N,)` array
    :param max_sigma: maximum possible value for sigma or None to compute it; used to compute kernel size
    :param truncate: kernel size truncation factor
    :return: kernels as a `(N, kernel_size)` array
    """
    if max_sigma is None:
        max_sigma = sigma.max()
    sigma = sigma[:, None]
    radius = int(truncate * max_sigma + 0.5)
    sigma2 = sigma * sigma
    x = np.arange(-radius, radius + 1)[None, :]
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum(axis=1, keepdims=True)
    return phi_x


class BoxMaskGenerator(MaskGenerator):
    def __init__(self, prop_range, n_boxes=1, random_aspect_ratio=True, prop_by_area=True, within_bounds=True,
                 invert=False):
        if isinstance(prop_range, float):
            prop_range = (prop_range, prop_range)
        self.prop_range = prop_range
        self.n_boxes = n_boxes
        self.random_aspect_ratio = random_aspect_ratio
        self.prop_by_area = prop_by_area
        self.within_bounds = within_bounds
        self.invert = invert

    def generate_params(self, n_masks, mask_shape, rng=None):
        """
        Box masks can be generated quickly on the CPU so do it there.
        >>> boxmix_gen = BoxMaskGenerator((0.25, 0.25))
        >>> params = boxmix_gen.generate_params(256, (32, 32))
        >>> t_masks = boxmix_gen.torch_masks_from_params(params, (32, 32), 'cuda:0')
        :param n_masks: number of masks to generate (batch size)
        :param mask_shape: Mask shape as a `(height, width)` tuple
        :param rng: [optional] np.random.RandomState instance
        :return: masks: masks as a `(N, 1, H, W)` array
        """
        if rng is None:
            rng = np.random

        if self.prop_by_area:
            # Choose the proportion of each mask that should be above the threshold
            mask_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))

            # Zeros will cause NaNs, so detect and suppres them
            zero_mask = mask_props == 0.0

            if self.random_aspect_ratio:
                y_props = np.exp(rng.uniform(low=0.0, high=1.0, size=(n_masks, self.n_boxes)) * np.log(mask_props))
                x_props = mask_props / y_props
            else:
                z_props = y_props = x_props = np.sqrt(mask_props)
            fac = np.sqrt(1.0 / self.n_boxes)

            z_props *= fac
            y_props *= fac
            x_props *= fac

            z_props[zero_mask] = 0
            y_props[zero_mask] = 0
            x_props[zero_mask] = 0
        else:
            if self.random_aspect_ratio:
                y_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))
                x_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))
            else:
                x_props = y_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))
            fac = np.sqrt(1.0 / self.n_boxes)
            y_props *= fac
            x_props *= fac

        sizes = np.round(np.stack([z_props, y_props, x_props], axis=2) * np.array(mask_shape)[None, None, :])

        if self.within_bounds:
            positions = np.round((np.array(mask_shape) - sizes) * rng.uniform(low=0.0, high=1.0, size=sizes.shape))
            rectangles = np.append(positions, positions + sizes, axis=2)
        else:
            centres = np.round(np.array(mask_shape) * rng.uniform(low=0.0, high=1.0, size=sizes.shape))
            rectangles = np.append(centres - sizes * 0.5, centres + sizes * 0.5, axis=2)

        if self.invert:
            masks = np.zeros((n_masks, 1) + mask_shape)
        else:
            masks = np.ones((n_masks, 1) + mask_shape)
        for i, sample_rectangles in enumerate(rectangles):
            for z0, y0, x0, z1, y1, x1 in sample_rectangles:
                masks[i, 0, int(z0):int(z1), int(y0):int(y1), int(x0):int(x1)] = 1 - masks[i, 0, int(z0):int(z1), int(y0):int(y1), int(x0):int(x1)]
        return masks

    def torch_masks_from_params(self, t_params, mask_shape, torch_device):
        return t_params


class AddMaskParamsToBatch(object):
    """
    We add the cut-and-paste parameters to the mini-batch within the collate function,
    (we pass it as the `batch_aug_fn` parameter to the `SegCollate` constructor)
    as the collate function pads all samples to a common size
    """

    def __init__(self, mask_gen):
        self.mask_gen = mask_gen

    def __call__(self, batch):
        sample = batch[0]
        if 'sample0' in sample:
            sample0 = sample['sample0']
        else:
            sample0 = sample
        mask_size = sample0['image'].shape[1:3]
        params = self.mask_gen.generate_params(len(batch), mask_size)
        for sample, p in zip(batch, params):
            sample['mask_params'] = p.astype(np.float32)
        return batch


def extract_ampl_phase(fft_im):
    # fft_im: size should be b x h x w x d
    fft_amp = torch.abs(fft_im)
    fft_pha = torch.angle(fft_im)
    return fft_amp, fft_pha


def low_freq_mutate(amp_src, amp_trg, L=0.1):
    bs, n, h, w, d = amp_src.size()
    # multiply w by 2 because we have only half the space as rFFT is used
    d *= 2
    # multiply by 0.5 to have the maximum b for L=1 like in the paper
    b = (np.floor(0.5 * np.amin((h, w, d)) * L)).astype(int)     # get b
    if b > 0:
        # When rFFT is used only half of the space needs to be updated
        # because of the symmetry along the last dimension
        # x = np.array(list(range(144))).reshape([144, 1, 1])
        # y = np.array(list(range(144))).reshape([1, 144, 1])
        # z = np.array(list(range(73))).reshape([1, 1, 73])
        # mask = (x - 0) ** 2 + (y - 0) ** 2 + (z - 0) ** 2 <= b ** 2
        # mask += (x - 0) ** 2 + (y - 144) ** 2 + (z - 0) ** 2 <= b ** 2
        # mask += (x - 144) ** 2 + (y - 0) ** 2 + (z - 0) ** 2 <= b ** 2
        # mask += (x - 144) ** 2 + (y - 144) ** 2 + (z - 0) ** 2 <= b ** 2
        # mask = mask.reshape(1, 1, 144, 144, 73).repeat(bs, axis=0)
        # mask = mask + 0
        # mask = torch.from_numpy(mask).cuda()
        # amp_src[:, :, :, :, 0:b] = amp_src[:, :, :, :, 0:b] * (1 - mask[..., 0:b]) + \
        #                            amp_trg[:, :, :, :, 0:b] * mask[..., 0:b]
        amp_src[:, :, 0:b, 0:b, 0:b] = amp_trg[:, :, 0:b, 0:b, 0:b]    # top left
        amp_src[:, :, h-b+1:h, 0:b, 0:b] = amp_trg[:, :, h-b+1:h, 0:b, 0:b]    # bottom left
        amp_src[:, :, 0:b, w-b+1:w, 0:b] = amp_trg[:, :, 0:b, w-b+1:w, 0:b]    # bottom right
        amp_src[:, :, h-b+1:h, w-b+1:w, 0:b] = amp_trg[:, :, h-b+1:h, w-b+1:w, 0:b]    # bottom right
    return amp_src


def FDA_source_to_target(src_img, trg_img, L=0.1):
    # get fft of both source and target
    fft_src = torch.fft.rfftn(src_img.clone(), dim=(-3, -2, -1))
    fft_trg = torch.fft.rfftn(trg_img.clone(), dim=(-3, -2, -1))

    # extract amplitude and phase of both ffts
    amp_src, pha_src = extract_ampl_phase(fft_src.clone())
    amp_trg, pha_trg = extract_ampl_phase(fft_trg.clone())

    # replace the low frequency amplitude part of source with that from target
    amp_src_ = low_freq_mutate(amp_src.clone(), amp_trg.clone(), L=L)

    # recompose fft of source
    real = torch.cos(pha_src.clone()) * amp_src_.clone()
    imag = torch.sin(pha_src.clone()) * amp_src_.clone()
    fft_src_ = torch.complex(real=real, imag=imag)

    # get the recomposed image: source content, target style
    _, _, imgH, imgW, imgD = src_img.size()
    src_in_trg = torch.fft.irfftn(fft_src_, dim=(-3, -2, -1), s=[imgH, imgW, imgD])

    return src_in_trg


class prototype_dist_estimator():
    def __init__(self, feature_num=32,resume=''):
        super(prototype_dist_estimator, self).__init__()

        self.class_num = 8
        self.feature_num = feature_num
        # momentum
        self.use_momentum = False
        self.momentum = 0.9

        # init prototype
        self.init(feature_num=feature_num, resume=resume)

    def init(self, feature_num, resume=""):
        if resume:
            if feature_num == self.class_num:
                resume = os.path.join(resume, 'prototype_out_dist.pth')
            elif feature_num == self.feature_num:
                resume = os.path.join(resume, 'prototype_feat_dist.pth')
            else:
                raise RuntimeError("Feature_num not available: {}".format(feature_num))
            print("Loading checkpoint from {}".format(resume))
            checkpoint = torch.load(resume, map_location=torch.device('cpu'))
            self.Proto = checkpoint['Proto'].cuda(non_blocking=True)
            self.Amount = checkpoint['Amount'].cuda(non_blocking=True)
        else:
            self.Proto = torch.zeros(self.class_num, feature_num).cuda(non_blocking=True)
            self.Amount = torch.zeros(self.class_num).cuda(non_blocking=True)

    def update(self, features, labels):
        if not self.use_momentum:
            N, A = features.size()
            C = self.class_num
            # refer to SDCA for fast implementation
            features = features.view(N, 1, A).expand(N, C, A)
            onehot = torch.zeros(N, C).cuda()
            onehot.scatter_(1, labels.view(-1, 1), 1)
            NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)
            features_by_sort = features.mul(NxCxA_onehot)
            Amount_CXA = NxCxA_onehot.sum(0)
            Amount_CXA[Amount_CXA == 0] = 1
            mean = features_by_sort.sum(0) / Amount_CXA
            sum_weight = onehot.sum(0).view(C, 1).expand(C, A)
            weight = sum_weight.div(
                sum_weight + self.Amount.view(C, 1).expand(C, A)
            )
            weight[sum_weight == 0] = 0
            self.Proto = (self.Proto.mul(1 - weight) + mean.mul(weight)).detach()
            self.Amount = self.Amount + onehot.sum(0)
        else:
            # momentum implementation
            ids_unique = labels.unique()
            for i in ids_unique:
                i = i.item()
                mask_i = (labels == i)
                feature = features[mask_i]
                feature = torch.mean(feature, dim=0)
                self.Amount[i] += len(mask_i)
                self.Proto[i, :] = self.momentum * feature + self.Proto[i, :] * (1 - self.momentum)

    def save(self, name):
        torch.save({'Proto': self.Proto.cpu(),
                    'Amount': self.Amount.cpu()
                    },name)


class PrototypeContrastiveLoss(nn.Module):
    def __init__(self):
        super(PrototypeContrastiveLoss, self).__init__()
        self.TAU = 1

    def forward(self, Proto, feat, labels):
        """
        Args:
            C: NUM_CLASSES A: feat_dim B: batch_size H: feat_high W: feat_width N: number of pixels except IGNORE_LABEL
            Proto: shape: (C, A) the mean representation of each class
            feat: shape (BHW, A) -> (N, A)
            labels: shape (BHW, ) -> (N, )

        Returns:

        """
        assert not Proto.requires_grad
        assert not labels.requires_grad
        assert feat.requires_grad
        assert feat.dim() == 2
        assert labels.dim() == 1
        # remove IGNORE_LABEL pixels
        # mask = (labels != self.cfg.INPUT.IGNORE_LABEL)
        # labels = labels[mask]
        # feat = feat[mask]

        feat = F.normalize(feat, p=2, dim=1)
        Proto = F.normalize(Proto, p=2, dim=1)

        logits = feat.mm(Proto.permute(1, 0).contiguous())
        logits = logits / self.TAU

        ce_criterion = nn.CrossEntropyLoss()
        loss = ce_criterion(logits, labels)

        return loss


class PseudoLabel:
    def __init__(self, OUTPUT_DIR):
        h, w, d = (144,144,144)
        self.prob_tar = np.zeros([1, h, w, d])
        self.label_tar = np.zeros([1, h, w, d])
        self.thres = []
        self.number_class = 8
        self.out_dir = OUTPUT_DIR
        self.iter = 0

    def save_results(self):
        np.save(os.path.join(self.out_dir, 'thres_const.npy'), self.thres)
        print("save done.")

    def update_pseudo_label(self, input):
        input = F.softmax(input.detach(), dim=1)
        prob, label = torch.max(input, dim=1)
        prob_np = prob.cpu().numpy()
        label_np = label.cpu().numpy()
        print(self.iter)
        if self.iter==0:
            self.prob_tar = prob_np
            self.label_tar = label_np
        else:
            self.prob_tar = np.append(self.prob_tar, prob_np, axis=0)
            self.label_tar = np.append(self.label_tar, label_np, axis=0)
        self.iter += 1

    def get_threshold_const(self, thred, percent=0.5):
        for i in range(self.number_class):
            x = self.prob_tar[self.label_tar == i]
            if len(x) == 0:
                self.thres.append(0)
                continue
            x = np.sort(x)
            self.thres.append(x[np.int(np.round(len(x) * percent))])
        self.thres = np.array(self.thres)
        self.thres[self.thres > thred] = thred
        return self.thres


class PixelContrastLoss(nn.Module):
    def __init__(self, ):
        super(PixelContrastLoss, self).__init__()

        self.temperature = 0.07
        self.base_temperature = 0.07
        self.max_samples = 8*50
        self.max_views = 50
        self.ignore_label = 0

    def _hard_anchor_sampling(self, X, y_hat, y):
        batch_size, feat_dim = X.shape[0], X.shape[-1]

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x != self.ignore_label]
            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]

            classes.append(this_classes)
            total_classes += len(this_classes)

        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)

        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()

        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]

            for cls_id in this_classes:
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep

                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)

                X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                y_[X_ptr] = cls_id
                X_ptr += 1

        return X_, y_

    def _sample_negative(self, Q):
        class_num, cache_size, feat_size = Q.shape

        X_ = torch.zeros((class_num * cache_size, feat_size)).float().cuda()
        y_ = torch.zeros((class_num * cache_size, 1)).float().cuda()
        sample_ptr = 0
        for ii in range(class_num):
            if ii == 0: continue
            this_q = Q[ii, :cache_size, :]
            X_[sample_ptr:sample_ptr + cache_size, ...] = this_q
            y_[sample_ptr:sample_ptr + cache_size, ...] = ii
            sample_ptr += cache_size

        return X_, y_

    def _contrastive(self, X_anchor, y_anchor, queue=None):
        anchor_num, n_view = X_anchor.shape[0], X_anchor.shape[1]

        y_anchor = y_anchor.contiguous().view(-1, 1)
        anchor_count = n_view
        anchor_feature = torch.cat(torch.unbind(X_anchor, dim=1), dim=0)

        if queue is not None:
            X_contrast, y_contrast = self._sample_negative(queue)
            y_contrast = y_contrast.contiguous().view(-1, 1)
            contrast_count = 1
            contrast_feature = X_contrast
        else:
            y_contrast = y_anchor
            contrast_count = n_view
            contrast_feature = torch.cat(torch.unbind(X_anchor, dim=1), dim=0)

        mask = torch.eq(y_anchor, y_contrast.T).float().cuda()

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T),
                                        self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask

        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
                                                     0)

        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def forward(self, feats, labels=None, predict=None, queue=None):
        labels = labels.float().clone()

        batch_size = feats.shape[0]

        labels = labels[...,::6,::6,::6]
        predict = predict[...,::6,::6,::6]
        feats = feats[...,::6,::6,::6]

        labels = labels.contiguous().view(batch_size, -1)
        predict = predict.contiguous().view(batch_size, -1)
        feats = feats.permute(0, 2, 3, 4, 1)

        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])

        feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict)

        loss = self._contrastive(feats_, labels_, queue=queue)
        return loss


class PixelContrastLoss1(nn.Module):
    def __init__(self, sample='random'):
        super(PixelContrastLoss1, self).__init__()

        self.temperature = 0.1
        self.base_temperature = 0.1
        self.max_samples = 28 * 50
        self.max_views = 50
        self.ignore_label = 8
        self.sample = sample

    def _hard_anchor_sampling(self, X, y_hat, y):
        batch_size, feat_dim = X.shape[0], X.shape[-1]

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x != self.ignore_label]
            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]

            classes.append(this_classes)
            total_classes += len(this_classes)

        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)

        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()

        X_ptr = 0

        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]

            if self.sample == 'patch':
                step = 6
                num = int(y.shape[-1] / step)
                this_feat = torch.zeros(1,feat_dim).cuda()
                this_label = torch.zeros(1).cuda()
                one = torch.ones(1).cuda()

                for k in range(num):
                    for q in range(num):
                        for p in range(num):
                            # if torch.rand(1) < 0.99:
                            #     continue
                            keys0 = X[ii, k * step:(k + 1) * step, q * step:(q + 1) * step,
                                    p * step:(p + 1) * step, :].reshape(-1, feat_dim)
                            labels0 = this_y_hat[:, k * step:(k + 1) * step, q * step:(q + 1) * step,
                                      p * step:(p + 1) * step].reshape(-1)
                            # pseudo0 = this_y[k * step:(k + 1) * step, q * step:(q + 1) * step,
                            #           p * step:(p + 1) * step].reshape(-1)
                            for j in this_classes:
                                index = torch.where(labels0 == j)[0]
                                if index.shape[0] == 0:
                                    continue
                                else:
                                    class_feature = torch.index_select(keys0, dim=0, index=index).mean(0, keepdim=True)
                                    this_feat = torch.cat((this_feat, class_feature), 0)
                                    this_label = torch.cat((this_label, one * j))

                this_feat = this_feat[1:, :]
                this_label = this_label[1:]

            for cls_id in this_classes:
                if self.sample == 'patch':
                    indices = (this_label == cls_id).nonzero()
                    perm = torch.randperm(indices.shape[0])
                    indices = indices[perm[:n_view]]
                else:
                    hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                    easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

                    num_hard = hard_indices.shape[0]
                    num_easy = easy_indices.shape[0]
                    # print(num_hard)
                    # print(num_easy)
                    # print(cls_id)
                    if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                        num_hard_keep = n_view // 2
                        num_easy_keep = n_view - num_hard_keep
                    elif num_hard >= n_view / 2:
                        num_easy_keep = num_easy
                        num_hard_keep = n_view - num_easy_keep
                    elif num_easy >= n_view / 2:
                        num_hard_keep = num_hard
                        num_easy_keep = n_view - num_hard_keep

                    perm = torch.randperm(num_hard)
                    hard_indices = hard_indices[perm[:num_hard_keep]]
                    perm = torch.randperm(num_easy)
                    easy_indices = easy_indices[perm[:num_easy_keep]]
                    indices = torch.cat((hard_indices, easy_indices), dim=0)

                if self.sample == 'patch':
                    # print(cls_id)
                    # print(this_feat.shape)
                    X_[X_ptr, :, :] = this_feat[indices].squeeze(1)
                else:
                    X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                y_[X_ptr] = cls_id
                X_ptr += 1

        return X_, y_

    def _sample_negative(self, Q):
        class_num, cache_size, feat_size = Q.shape

        X_ = torch.zeros((class_num * cache_size, feat_size)).float().cuda()
        y_ = torch.zeros((class_num * cache_size, 1)).float().cuda()
        sample_ptr = 0
        for ii in range(class_num):
            if ii == 0: continue
            this_q = Q[ii, :cache_size, :]
            X_[sample_ptr:sample_ptr + cache_size, ...] = this_q
            y_[sample_ptr:sample_ptr + cache_size, ...] = ii
            sample_ptr += cache_size

        return X_, y_

    def _contrastive(self, X_anchor, y_anchor, queue=None):
        anchor_num, n_view = X_anchor.shape[0], X_anchor.shape[1]

        y_anchor = y_anchor.contiguous().view(-1, 1)
        anchor_count = n_view
        anchor_feature = torch.cat(torch.unbind(X_anchor, dim=1), dim=0)

        if queue is not None:
            X_pos, y_pos = self._sample_negative(queue[:,:queue.shape[1]//2,:])
            y_pos = y_pos.contiguous().view(-1, 1)
            X_neg, y_neg = self._sample_negative(queue[:,queue.shape[1]//2:,:])
            y_neg = y_neg.contiguous().view(-1, 1)
            contrast_count = 1
        else:
            y_contrast = y_anchor
            contrast_count = n_view
            contrast_feature = torch.cat(torch.unbind(X_anchor, dim=1), dim=0)
        loss = 0
        this_classes = torch.unique(y_anchor)
        for c in this_classes:
            anchor_feature = X_anchor[y_anchor[:,0]==c].reshape(-1,X_anchor.shape[-1])
            pos_feature = X_pos[y_pos[:,0]==c,:]
            neg_feature = X_neg[y_neg[:,0]==c,:]
            anchor_feature = nn.functional.normalize(anchor_feature, p=2, dim=1)
            pos_feature = nn.functional.normalize(pos_feature, p=2, dim=1)
            neg_feature = nn.functional.normalize(neg_feature, p=2, dim=1)
            logits = torch.div(torch.matmul(anchor_feature, pos_feature.T),
                               self.temperature)
            neg_logits = torch.div(torch.matmul(anchor_feature, neg_feature.T),
                               self.temperature)
            neg_logits = torch.exp(neg_logits).sum(1, keepdim=True)
            exp_logits = torch.exp(logits)
            log_prob = - logits + torch.log(exp_logits + neg_logits)
            loss += log_prob.mean()
        loss /= len(this_classes)
        return loss

    def forward(self, feats, labels=None, predict=None, queue=None):
        labels = labels.float().clone()

        batch_size = feats.shape[0]

        if self.sample == 'uniform':
            num = random.randint(0, 10)
            feats = feats[..., num::6, num::6, num::6]
            labels = labels[..., num::6, num::6, num::6]
            predict = predict[..., num::6, num::6, num::6]

        feats = feats.permute(0, 2, 3, 4, 1)
        if self.sample != 'patch':
            labels = labels.contiguous().view(batch_size, -1)
            predict = predict.contiguous().view(batch_size, -1)
            feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])

        feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict)

        loss = self._contrastive(feats_, labels_, queue=queue)
        return loss


class ContrastCELoss(nn.Module):
    def __init__(self,sample):
        super(ContrastCELoss, self).__init__()

        self.contrast_criterion = PixelContrastLoss1(sample)

    def forward(self, seg, target, embedding, segment_queue, pixel_queue):
        # seg = preds['seg']
        # embedding = preds['feat']
        # segment_queue = preds['segment_queue']
        # pixel_queue = preds['pixel_queue']

        queue = torch.cat((segment_queue, pixel_queue), dim=1)

        _, predict = torch.max(seg, 1)
        loss_contrast = self.contrast_criterion(embedding, target, predict, queue)

        return loss_contrast  # just a trick to avoid errors in distributed training


def dequeue_and_enqueue(keys, labels, preds,
                        pixel_queue, pixel_queue_ptr,
                        unreliable_queue, unreliable_queue_ptr,
                        pixel_update_freq=50,
                        label=True,high_entropy_mask=None,sample='random',unlabel=False,atlas=None):
    batch_size = keys.shape[0]
    feat_dim = keys.shape[1]
    pseudo = torch.argmax(preds, dim=1, keepdim=True)
    _,unpseudo = torch.sort(preds, dim=1)
    if label:
        unpseudo = unpseudo[:,-2:,...]
    else:
        unpseudo = unpseudo[:,:2,...]
    if sample=='uniform':
        num = random.randint(0,10)
        keys = keys[...,num::6,num::6,num::6]
        labels = labels[...,num::6,num::6,num::6]
        pseudo = pseudo[...,num::6,num::6,num::6]
        unpseudo = unpseudo[...,num::6,num::6,num::6]
        if atlas is not None:
            atlas = atlas[...,num::6,num::6,num::6]
        if high_entropy_mask is not None:
            high_entropy_mask = high_entropy_mask[...,num::6,num::6,num::6]
    for bs in range(batch_size):
        this_feat = keys[bs].contiguous().view(feat_dim, -1)
        this_label = labels[bs].contiguous().view(-1)
        this_pseudo = pseudo[bs].contiguous().view(-1)
        this_unpseudo = unpseudo[bs].contiguous().view(2,-1)
        if atlas is not None:
            this_atlas = atlas[bs].contiguous().view(-1)
        if high_entropy_mask is not None:
            this_high_entropy_mask = high_entropy_mask[bs].contiguous().view(-1)
        this_label_ids = torch.unique(this_label)
        this_label_ids = [x for x in this_label_ids if x != 8]

        if sample == 'patch':
            step = 6
            num = int(labels.shape[-1] / step)
            this_feat_p = torch.zeros(feat_dim, 1).cuda()
            this_feat_u = torch.zeros(feat_dim, 1).cuda()
            this_label_p = torch.zeros(1).cuda()
            this_label_u = torch.zeros(1).cuda()
            one = torch.ones(1).cuda()
            for k in range(num):
                for q in range(num):
                    for p in range(num):
                        block_labels = labels[bs, :, k * step:(k + 1) * step, q * step:(q + 1) * step,
                                  p * step:(p + 1) * step].reshape(-1)
                        if block_labels.sum() < 1: continue
                        b, c = torch.unique(block_labels, return_counts=True)
                        indexes = torch.where(b != 8)[0]
                        b = b[indexes]
                        c = c[indexes]
                        if len(b) == 0: continue
                        if c.min() < 20: continue
                        if b[0] == 0:
                            if c.shape[0] < 3: continue
                            if random.randint(0, 9) < 8: continue
                        else:
                            # continue
                            if 4 in b or 5 in b or 6 in b or 7 in b:
                                if c.shape[0] < 3: continue
                                if random.randint(0, 9) < 6: continue
                            else:
                                continue

                        keys0 = keys[bs, :, k * step:(k + 1) * step, q * step:(q + 1) * step,
                                p * step:(p + 1) * step].reshape(feat_dim, -1)
                        labels0 = labels[bs, :, k * step:(k + 1) * step, q * step:(q + 1) * step,
                                  p * step:(p + 1) * step].reshape(-1)
                        pseudo0 = pseudo[bs, :, k * step:(k + 1) * step, q * step:(q + 1) * step,
                                  p * step:(p + 1) * step].reshape(-1)
                        unpseudo0 = unpseudo[bs, 0, k * step:(k + 1) * step, q * step:(q + 1) * step,
                                    p * step:(p + 1) * step].reshape(-1)
                        unpseudo1 = unpseudo[bs, 1, k * step:(k + 1) * step, q * step:(q + 1) * step,
                                    p * step:(p + 1) * step].reshape(-1)
                        if high_entropy_mask is not None:
                            high_entropy_mask0 = high_entropy_mask[bs, :, k * step:(k + 1) * step, q * step:(q + 1) * step,
                                                 p * step:(p + 1) * step].reshape(-1)
                        for j in this_label_ids:
                            if label:
                                index = torch.where((labels0 == j) & (pseudo0 == j))[0]
                                unreliable_index = torch.where((((unpseudo0 == j) + (unpseudo1 == j)) &
                                                                (labels0 != j)).nonzero())[0]
                                if index.shape[0] < 10:
                                    continue
                                else:
                                    class_feature = torch.index_select(keys0, dim=-1, index=index).mean(-1, keepdim=True)
                                    this_feat_p = torch.cat((this_feat_p, class_feature), 1)
                                    this_label_p = torch.cat((this_label_p,one*j))
                                if unreliable_index.shape[0] < 10:
                                    continue
                                else:
                                    class_feature = torch.index_select(keys0, dim=-1, index=unreliable_index).mean(-1, keepdim=True)
                                    this_feat_u = torch.cat((this_feat_u, class_feature), 1)
                                    this_label_u = torch.cat((this_label_u,one*j))

                            if high_entropy_mask is not None:
                                unreliable_index = torch.where((((unpseudo0 == j) + (unpseudo1 == j)) & (
                                        high_entropy_mask0 > 0)).nonzero())[0]
                                if unreliable_index.shape[0] < 10:
                                    continue
                                else:
                                    class_feature = torch.index_select(keys0, dim=-1, index=unreliable_index).mean(-1, keepdim=True)
                                    this_feat_u = torch.cat((this_feat_u, class_feature), 1)
                                    this_label_u = torch.cat((this_label_u,one*j))
            this_feat_u = this_feat_u[:, 1:]
            this_feat_p = this_feat_p[:, 1:]
            this_label_u = this_label_u[1:]
            this_label_p = this_label_p[1:]
        for lb in this_label_ids:
            lb = int(lb)
            # pixel enqueue and dequeue
            if unlabel:
                if sample == 'patch':
                    idxs = (this_label_p == lb).nonzero()
                else:
                    idxs = (this_label == lb & (this_atlas == lb)).nonzero()
                num_pixel = idxs.shape[0]
                perm = torch.randperm(num_pixel)
                K = min(num_pixel, pixel_update_freq)
                if sample == 'patch':
                    feat = this_feat_p[:, perm[:K]]
                else:
                    feat = this_feat[:, perm[:K]]
                feat = torch.transpose(feat, 0, 1)
                ptr = int(pixel_queue_ptr[lb])
                if ptr + K >= pixel_queue.shape[1]:
                    pixel_queue[lb, -K:, :] = nn.functional.normalize(feat, p=2, dim=1)
                    pixel_queue_ptr[lb] = 0
                else:
                    pixel_queue[lb, ptr:ptr + K, :] = nn.functional.normalize(feat, p=2, dim=1)
                    pixel_queue_ptr[lb] = (pixel_queue_ptr[lb] + 1) % pixel_queue.shape[1]
            elif label:
                if sample == 'patch':
                    idxs = (this_label_p == lb).nonzero()
                else:
                    idxs = (this_label == lb & (this_pseudo == lb)).nonzero()
                num_pixel = idxs.shape[0]
                perm = torch.randperm(num_pixel)
                K = min(num_pixel, pixel_update_freq)
                if sample == 'patch':
                    feat = this_feat_p[:, perm[:K]]
                else:
                    feat = this_feat[:, perm[:K]]
                feat = torch.transpose(feat, 0, 1)
                ptr = int(pixel_queue_ptr[lb])
                if ptr + K >= pixel_queue.shape[1]:
                    pixel_queue[lb, -K:, :] = nn.functional.normalize(feat, p=2, dim=1)
                    pixel_queue_ptr[lb] = 0
                else:
                    pixel_queue[lb, ptr:ptr + K, :] = nn.functional.normalize(feat, p=2, dim=1)
                    pixel_queue_ptr[lb] = (pixel_queue_ptr[lb] + 1) % pixel_queue.shape[1]

            # unreliable pixel enqueue and dequeue
            if label:
                idxs = (((this_unpseudo[0] == lb) + (this_unpseudo[1] == lb)) & (this_label != lb)).nonzero()
            if high_entropy_mask is not None:
                idxs = (((this_unpseudo[0] == lb) + (this_unpseudo[1] == lb)) & (this_high_entropy_mask>0)).nonzero()
            if sample == 'patch':
                idxs = (this_label_u == lb).nonzero()
            num_pixel = idxs.shape[0]
            perm = torch.randperm(num_pixel)
            K = min(num_pixel, pixel_update_freq)
            if sample == 'patch':
                feat = this_feat_u[:, perm[:K]]
            else:
                feat = this_feat[:, perm[:K]]
            feat = torch.transpose(feat, 0, 1)

            ptr = int(unreliable_queue_ptr[lb])
            if ptr + K >= unreliable_queue.shape[1]:
                unreliable_queue[lb, -K:, :] = nn.functional.normalize(feat, p=2, dim=1)
                unreliable_queue_ptr[lb] = 0
            else:
                unreliable_queue[lb, ptr:ptr + K, :] = nn.functional.normalize(feat, p=2, dim=1)
                unreliable_queue_ptr[lb] = (unreliable_queue_ptr[lb] + 1) % unreliable_queue.shape[1]

    return pixel_queue, pixel_queue_ptr, unreliable_queue, unreliable_queue_ptr


class PixelContrastLoss2(nn.Module):
    def __init__(self, sample='random'):
        super(PixelContrastLoss2, self).__init__()

        self.temperature = 0.1
        self.base_temperature = 0.1
        self.max_samples = 28 * 50
        self.max_views = 50
        self.ignore_label = 0
        self.sample = sample
        self.current_class_threshold = 0.3

    def _hard_anchor_sampling(self, X, y_hat, y):
        feat_dim = X.shape[-1]

        classes = []
        total_classes = 0

        this_y = y_hat
        this_classes = torch.unique(this_y)
        this_classes = [x for x in this_classes if x != self.ignore_label]
        this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]

        total_classes += len(this_classes)

        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)

        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()

        X_ptr = 0

        this_y_hat = y_hat
        this_y = y

        for cls_id in this_classes:
            hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
            easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

            num_hard = hard_indices.shape[0]
            num_easy = easy_indices.shape[0]

            if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                num_hard_keep = n_view // 2
                num_easy_keep = n_view - num_hard_keep
            elif num_hard >= n_view / 2:
                num_easy_keep = num_easy
                num_hard_keep = n_view - num_easy_keep
            elif num_easy >= n_view / 2:
                num_hard_keep = num_hard
                num_easy_keep = n_view - num_hard_keep

            perm = torch.randperm(num_hard)
            hard_indices = hard_indices[perm[:num_hard_keep]]
            perm = torch.randperm(num_easy)
            easy_indices = easy_indices[perm[:num_easy_keep]]
            indices = torch.cat((hard_indices, easy_indices), dim=0)

            X_[X_ptr, :, :] = X[indices, :].squeeze(1)
            y_[X_ptr] = cls_id
            X_ptr += 1

        return X_, y_

    def _sample_negative(self, Q):
        class_num, cache_size, feat_size = Q.shape

        X_ = torch.zeros((class_num * cache_size, feat_size)).float().cuda()
        y_ = torch.zeros((class_num * cache_size, 1)).float().cuda()
        sample_ptr = 0
        for ii in range(class_num):
            if ii == 0: continue
            this_q = Q[ii, :cache_size, :]
            X_[sample_ptr:sample_ptr + cache_size, ...] = this_q
            y_[sample_ptr:sample_ptr + cache_size, ...] = ii
            sample_ptr += cache_size

        return X_, y_

    def _contrastive(self, X_anchor, y_anchor, queue=None, positive_queue=None):
        anchor_num, n_view = X_anchor.shape[0], X_anchor.shape[1]
        y_anchor = y_anchor.contiguous().view(-1, 1)

        if queue is not None:
            X_neg, y_neg = self._sample_negative(queue[:,queue.shape[1]//2:,:])
            y_neg = y_neg.contiguous().view(-1, 1)
            contrast_count = 1
        else:
            y_contrast = y_anchor
            contrast_count = n_view
            contrast_feature = torch.cat(torch.unbind(X_anchor, dim=1), dim=0)
        loss = 0
        this_classes = torch.unique(y_anchor)
        for c in this_classes:
            c = int(c)
            anchor_feature = X_anchor[y_anchor[:,0]==c].reshape(-1,X_anchor.shape[-1])
            # print(positive_queue.shape)
            # print(c)
            pos_feature = positive_queue[c].unsqueeze(0)
            # print(positive_queue.shape)
            neg_feature = X_neg[y_neg[:,0]==c,:]
            anchor_feature = nn.functional.normalize(anchor_feature, p=2, dim=1)
            pos_feature = nn.functional.normalize(pos_feature, p=2, dim=1)
            neg_feature = nn.functional.normalize(neg_feature, p=2, dim=1)
            logits = torch.div(torch.matmul(anchor_feature, pos_feature.T),
                               self.temperature)
            neg_logits = torch.div(torch.matmul(anchor_feature, neg_feature.T),
                               self.temperature)
            neg_logits = torch.exp(neg_logits).sum(1, keepdim=True)
            exp_logits = torch.exp(logits)
            log_prob = - logits + torch.log(exp_logits + neg_logits)
            loss += log_prob.mean()
        loss /= len(this_classes)
        return loss

    def forward(self, feats, labels=None, predict=None, probability=None, queue=None):
        labels = labels.float().clone()

        batch_size = feats.shape[0]

        if self.sample == 'uniform':
            num = random.randint(0, 10)
            feats = feats[..., num::6, num::6, num::6]
            labels = labels[..., num::6, num::6, num::6]
            predict = predict[..., num::6, num::6, num::6]
            probability = probability[..., num::6, num::6, num::6]

        feats = feats.permute(0, 2, 3, 4, 1)
        labels = labels.contiguous().view(batch_size, -1)  # (B,N)
        predict = predict.contiguous().view(batch_size, -1)  # (B,N)
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])  # (B,N,C)
        probability = probability.contiguous().view(batch_size, -1)  # (B,N)
        labels = labels[probability>self.current_class_threshold]
        feats = feats[probability>self.current_class_threshold]
        predict = predict[probability>self.current_class_threshold]

        feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict)
        positive_queue = torch.zeros(1,feats.shape[-1]).cuda()
        for ii in torch.unique(labels):
            # print(ii,torch.unique(labels))
            a = feats[labels==ii].mean(0,keepdim=True)
            positive_queue = torch.cat((positive_queue,a))
            # print(positive_queue.shape)
        positive_queue = positive_queue[1:]
        loss = self._contrastive(feats_, labels_, queue=queue, positive_queue=positive_queue)
        return loss


class ContrastCELoss_u2pl(nn.Module):
    def __init__(self,sample):
        super(ContrastCELoss_u2pl, self).__init__()

        self.contrast_criterion = PixelContrastLoss2(sample)

    def forward(self, seg, target, embedding, pixel_queue):
        probability, predict = torch.max(seg, 1)
        loss_contrast = self.contrast_criterion(embedding, target, predict, probability, pixel_queue)

        return loss_contrast  # just a trick to avoid errors in distributed training


def dequeue_and_enqueue_ur(keys, labels, preds,
                           unreliable_queue, unreliable_queue_ptr,
                           pixel_update_freq=50,
                           label=True,high_entropy_mask=None,sample='random'):
    batch_size = keys.shape[0]
    feat_dim = keys.shape[1]
    pseudo = torch.argmax(preds, dim=1, keepdim=True)
    _,unpseudo = torch.sort(preds, dim=1)
    if label:
        unpseudo = unpseudo[:,-2:,...]
    else:
        unpseudo = unpseudo[:,:2,...]
    if sample=='uniform':
        num = random.randint(0,10)
        keys = keys[...,num::6,num::6,num::6]
        labels = labels[...,num::6,num::6,num::6]
        pseudo = pseudo[...,num::6,num::6,num::6]
        unpseudo = unpseudo[...,num::6,num::6,num::6]
        if high_entropy_mask is not None:
            high_entropy_mask = high_entropy_mask[...,num::6,num::6,num::6]
    for bs in range(batch_size):
        this_feat = keys[bs].contiguous().view(feat_dim, -1)
        this_label = labels[bs].contiguous().view(-1)
        this_unpseudo = unpseudo[bs].contiguous().view(2,-1)
        if high_entropy_mask is not None:
            this_high_entropy_mask = high_entropy_mask[bs].contiguous().view(-1)
        this_label_ids = torch.unique(this_label)
        this_label_ids = [x for x in this_label_ids if x > 0]

        for lb in this_label_ids:
            lb = int(lb)
            # unreliable pixel enqueue and dequeue
            if label:
                idxs = (((this_unpseudo[0] == lb) + (this_unpseudo[1] == lb)) & (this_label != lb)).nonzero()
            if high_entropy_mask is not None:
                idxs = (((this_unpseudo[0] == lb) + (this_unpseudo[1] == lb)) & (this_high_entropy_mask>0)).nonzero()
            num_pixel = idxs.shape[0]
            perm = torch.randperm(num_pixel)
            K = min(num_pixel, pixel_update_freq)
            feat = this_feat[:, perm[:K]]
            feat = torch.transpose(feat, 0, 1)

            ptr = int(unreliable_queue_ptr[lb])
            if ptr + K >= unreliable_queue.shape[1]:
                unreliable_queue[lb, -K:, :] = nn.functional.normalize(feat, p=2, dim=1)
                unreliable_queue_ptr[lb] = 0
            else:
                unreliable_queue[lb, ptr:ptr + K, :] = nn.functional.normalize(feat, p=2, dim=1)
                unreliable_queue_ptr[lb] = (unreliable_queue_ptr[lb] + 1) % unreliable_queue.shape[1]

    return unreliable_queue, unreliable_queue_ptr


class SupConLoss(nn.Module):
    """modified supcon loss for segmentation application, the main difference is that the label for different view
    could be different if after spatial transformation"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.ignore_label = 8

    def forward(self, features, labels=None):
        labels = labels.reshape(-1)
        features = features.permute(0, 2, 3, 4, 1)
        features = features.reshape(-1,features.shape[-1])
        this_classes, this_num = torch.unique(labels, return_counts=True)
        indexes = torch.where(this_classes != self.ignore_label)[0]
        this_classes = this_classes[indexes]
        this_num = this_num[indexes]
        # if this_classes[0] == 0:
        #     this_num = this_num[1:]
        #     this_classes = this_classes[1:]
        n_view = this_num.min()
        total_classes = this_classes.shape[0]
        feat_dim = features.shape[1]
        X_anchor = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
        y_anchor = torch.zeros(total_classes, dtype=torch.float).cuda()
        X_ptr = 0
        for cls_id in this_classes:
            indices = (labels == cls_id).nonzero()
            perm = torch.randperm(n_view)
            indices = indices[perm[:n_view]]
            X_anchor[X_ptr, :, :] = features[indices, :].squeeze(1)
            y_anchor[X_ptr] = cls_id
            X_ptr+=1

        anchor_num, n_view = X_anchor.shape[0], X_anchor.shape[1]
        y_anchor = y_anchor.contiguous().view(-1, 1)
        anchor_count = n_view
        anchor_feature = torch.cat(torch.unbind(X_anchor, dim=1), dim=0)

        y_contrast = y_anchor
        contrast_count = n_view
        contrast_feature = torch.cat(torch.unbind(X_anchor, dim=1), dim=0)

        mask = torch.eq(y_anchor, y_contrast.T).float().cuda()

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T),
                                        self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask

        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
                                                     0)
        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        return loss


class BlockConLoss(nn.Module):
    def __init__(self, temperature=0.5, block_size=6):
        super(BlockConLoss, self).__init__()
        self.block_size = block_size
        self.supconloss = SupConLoss(temperature=temperature,base_temperature=temperature)
        self.ignore_label = 8

    def forward(self, features, labels):
        # input features: [bsz, c, h ,w, d], h & w & d are the image size
        shape = features.shape
        img_size = shape[-1]
        div_num = img_size // self.block_size
        num = 0
        loss = []
        for i in range(div_num):
            for j in range(div_num):
                for k in range(div_num):
                    block_labels = labels[:, :, i*self.block_size:(i+1)*self.block_size,
                                   j*self.block_size:(j+1)*self.block_size,
                                   k*self.block_size:(k+1)*self.block_size]
                    if block_labels.sum() < 1:continue
                    b, c = torch.unique(block_labels, return_counts=True)
                    indexes = torch.where(b != self.ignore_label)[0]
                    # print(b,c)
                    # print(torch.where(b != self.ignore_label))
                    b = b[indexes]
                    c = c[indexes]
                    # print(b,c)
                    # print('__________')
                    if len(b) == 0:continue
                    if c.min() < 20:continue
                    if b[0] == 0:
                        if c.shape[0] < 3:continue
                        if random.randint(0, 9)<8:continue
                    else:
                        # continue
                        if 4 in b or 5 in b or 6 in b or 7 in b:
                            if c.shape[0] < 3:continue
                            if random.randint(0, 9)<6:continue
                        else:continue
                    num+=1
                    block_features = features[:, :, i * self.block_size:(i + 1) * self.block_size,
                                     j * self.block_size:(j + 1) * self.block_size,
                                     k * self.block_size:(k + 1) * self.block_size]
                    tmp_loss = self.supconloss(block_features, block_labels)
                    loss.append(tmp_loss)
        print(num)
        if len(loss) == 0:
            loss = torch.tensor(0).float().cuda()
            return loss
        loss = torch.stack(loss).mean()
        return loss


def dequeue_and_enqueue3(keys, labels,
                         pixel_queue, pixel_queue_ptr,
                         pixel_update_freq=50, step=6):
    batch_size = keys.shape[0]
    feat_dim = keys.shape[1]

    for bs in range(batch_size):
        this_label = labels[bs].contiguous().view(-1)
        this_label_ids = torch.unique(this_label)
        this_label_ids = [x for x in this_label_ids if x != 8]

        num = int(labels.shape[-1] / step)
        this_feat = torch.zeros(feat_dim, 1).cuda()
        this_label = torch.zeros(1).cuda()
        one = torch.ones(1).cuda()
        for k in range(num):
            for q in range(num):
                for p in range(num):
                    labels0 = labels[bs, :, k * step:(k + 1) * step, q * step:(q + 1) * step,
                              p * step:(p + 1) * step].reshape(-1)
                    if labels0.sum() < 1: continue
                    b, c = torch.unique(labels0, return_counts=True)
                    indexes = torch.where(b != 8)[0]
                    b = b[indexes]
                    c = c[indexes]
                    if len(b) == 0: continue
                    if c.min() < 20: continue
                    if b[0] == 0:
                        if c.shape[0] < 3: continue
                        # if random.randint(0, 9) < 9: continue
                    else:
                        if 4 in b or 5 in b or 6 in b or 7 in b:
                            if c.shape[0] < 3: continue
                            if random.randint(0, 9) < 10: continue
                        else:
                            continue
                    print(b,c)
                    keys0 = keys[bs, :, k * step:(k + 1) * step, q * step:(q + 1) * step,
                            p * step:(p + 1) * step].reshape(feat_dim, -1)
                    for j in b:
                        index = torch.where(labels0 == j)[0]
                        class_feature = torch.index_select(keys0, dim=-1, index=index)
                        this_feat = torch.cat((this_feat, class_feature), 1)
                        this_label = torch.cat((this_label, one*j))
        this_feat = this_feat[:, 1:]
        this_label = this_label[1:]
        print(this_label.shape)
        for lb in this_label_ids:
            lb = int(lb)
            # pixel enqueue and dequeue
            idxs = (this_label == lb).nonzero()
            num_pixel = idxs.shape[0]
            perm = torch.randperm(num_pixel)
            K = min(num_pixel, pixel_update_freq)
            feat = this_feat[:, perm[:K]]
            feat = torch.transpose(feat, 0, 1)
            ptr = int(pixel_queue_ptr[lb])
            if ptr + K >= pixel_queue.shape[1]:
                pixel_queue[lb, -K:, :] = nn.functional.normalize(feat, p=2, dim=1)
                pixel_queue_ptr[lb] = 0
            else:
                pixel_queue[lb, ptr:ptr + K, :] = nn.functional.normalize(feat, p=2, dim=1)
                pixel_queue_ptr[lb] = (pixel_queue_ptr[lb] + 1) % pixel_queue.shape[1]
    return pixel_queue, pixel_queue_ptr


class PixelContrastLoss3(nn.Module):
    def __init__(self, ):
        super(PixelContrastLoss3, self).__init__()

        self.temperature = 0.07
        self.max_samples = 8*50
        self.max_views = 50
        self.ignore_label = 8
        self.block_size = 6

    def _sample_negative(self, Q):
        class_num, cache_size, feat_size = Q.shape

        X_ = torch.zeros((class_num * cache_size, feat_size)).float().cuda()
        y_ = torch.zeros((class_num * cache_size, 1)).float().cuda()
        sample_ptr = 0
        for ii in range(class_num):
            this_q = Q[ii, :cache_size, :]
            X_[sample_ptr:sample_ptr + cache_size, ...] = this_q
            y_[sample_ptr:sample_ptr + cache_size, ...] = ii
            sample_ptr += cache_size

        return X_, y_

    def forward(self, feats, labels, queue=None):
        batch_size = feats.shape[0]
        feat_dim = feats.shape[1]
        all_loss = []
        for bs in range(batch_size):
            num = int(labels.shape[-1] / self.block_size)
            this_feat = torch.zeros(feat_dim, 1).cuda()
            this_label = torch.zeros(1).cuda()
            one = torch.ones(1).cuda()
            for k in range(num):
                for q in range(num):
                    for p in range(num):
                        labels0 = labels[bs, :, k * self.block_size:(k + 1) * self.block_size, q * self.block_size:(q + 1) * self.block_size,
                                  p * self.block_size:(p + 1) * self.block_size].reshape(-1)
                        if labels0.sum() < 1: continue
                        b, c = torch.unique(labels0, return_counts=True)
                        indexes = torch.where(b != 8)[0]
                        b = b[indexes]
                        c = c[indexes]
                        if len(b) == 0: continue
                        if c.min() < 20: continue
                        if b[0] == 0:
                            if c.shape[0] < 3: continue
                            # if random.randint(0, 9) < 9: continue
                        else:
                            if 4 in b or 5 in b or 6 in b or 7 in b:
                                if c.shape[0] < 3: continue
                                if random.randint(0, 9) < 10: continue
                            else:
                                continue
                        print(b, c)
                        keys0 = feats[bs, :, k * self.block_size:(k + 1) * self.block_size, q * self.block_size:(q + 1) * self.block_size,
                                p * self.block_size:(p + 1) * self.block_size].reshape(feat_dim, -1)
                        for j in b:
                            index = torch.where(labels0 == j)[0]
                            class_feature = torch.index_select(keys0, dim=-1, index=index)
                            this_feat = torch.cat((this_feat, class_feature), 1)
                            this_label = torch.cat((this_label, one * j))
            this_feat = this_feat[:, 1:]
            this_label = this_label[1:]
            print(this_label.shape)
            #
            this_feat = this_feat.reshape(-1, feat_dim)
            this_classes, this_num = torch.unique(this_label, return_counts=True)

            n_view = self.max_views
            total_classes = this_classes.shape[0]
            X_anchor = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
            y_anchor = torch.zeros(total_classes, dtype=torch.float).cuda()
            X_ptr = 0
            for cls_id in this_classes:
                indices = (this_label == cls_id).nonzero()
                perm = torch.randperm(n_view)
                indices = indices[perm[:n_view]]
                X_anchor[X_ptr, :, :] = this_feat[indices, :].squeeze(1)
                y_anchor[X_ptr] = cls_id
                X_ptr += 1
            #
            anchor_num, n_view = X_anchor.shape[0], X_anchor.shape[1]
            y_anchor = y_anchor.contiguous().view(-1, 1)
            anchor_count = n_view
            anchor_feature = torch.cat(torch.unbind(X_anchor, dim=1), dim=0)

            if queue is not None:
                X_contrast, y_contrast = self._sample_negative(queue)
                y_contrast = y_contrast.contiguous().view(-1, 1)
                contrast_count = 1
                contrast_feature = X_contrast
            else:
                y_contrast = y_anchor
                contrast_count = n_view
                contrast_feature = torch.cat(torch.unbind(X_anchor, dim=1), dim=0)

            mask = torch.eq(y_anchor, y_contrast.T).float().cuda()

            anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T),
                                            self.temperature)
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()

            mask = mask.repeat(anchor_count, contrast_count)
            neg_mask = 1 - mask

            logits_mask = torch.ones_like(mask).scatter_(1,
                                                         torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
                                                         0)
            mask = mask * logits_mask

            neg_logits = torch.exp(logits) * neg_mask
            neg_logits = neg_logits.sum(1, keepdim=True)

            exp_logits = torch.exp(logits)

            log_prob = logits - torch.log(exp_logits + neg_logits)

            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = loss.mean()
            all_loss.append(loss)
        all_loss = torch.stack(all_loss).mean()
        return all_loss

