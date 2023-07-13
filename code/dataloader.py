import os
import torch
import random
import logging
import numpy as np
from glob import glob
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import nibabel as nib
from scipy.ndimage import binary_fill_holes


class BaseDataSets(Dataset):
    def __init__(self, data_dir=None, mode='train', list_name='train.list',
                 transform=None, patch_size=[144, 144, 144], crop=None, zoom=None, atlas=None, data_dir_s=None, registered=None):
        self._data_dir = data_dir
        self.sample_list = []
        self.mode = mode
        self.list_name = list_name
        self.transform = transform
        self.patch_size = patch_size
        self.crop = crop
        self.zoom = zoom
        self.atlas = atlas
        self.registered = registered
        self.data_dir_s = data_dir_s
        self.illness = ['sub-070','sub-052','sub-066','sub-005','sub-017','sub-016','sub-042','sub-023','sub-025','sub-008','sub-020','sub-021','sub-014','sub-080','sub-073','sub-003','sub-013','sub-064','sub-048','sub-002','sub-009','sub-075','sub-024','sub-050']
        self.illness_t = ['sub-022','sub-065','sub-063','sub-071','sub-007','sub-043','sub-074','sub-015','sub-078','sub-006','sub-004','sub-012','sub-056','sub-077','sub-055','sub-010','sub-069','sub-011','sub-001','sub-047','sub-018','sub-067','sub-019','sub-049','sub-054']
        self.week = [27.9,
               28.2,
               27.4,
               25.5,
               22.6,
               24.9,
               22.8,
               25.2,
               29,
               27.3,
               27.6,
               25.9,
               27.5,
               26.7,
               23.7,
               23.3,
               22.8,
               28.5,
               29.2,
               25.8,
               26.1,
               20,
               23.7,
               30.4,
               24.2,
               27.8,
               26.5,
               31.1,
               32.5,
               33.4,
               31.4,
               32.3,
               30,
               28.7,
               32.8,
               22.7,
               23.4,
               26.9,
               24.3,
               27.3,
               34.8,
               23.6,
               22.9,
               27.9,
               24.7,
               23.9,
               28.1,
               27.9,
               31.1,
               33.1,
               29.6,
               21.2,
               30.3,
               33.1,
               27.1,
               26.6,
               28.2,
               29.2,
               34.8,
               31.7,
               33,
               24.4,
               21.7,
               27.8,
               20.9,
               21.8,
               29,
               31.5,
               27.4,
               20.1,
               22.4,
               25.9,
               27.2,
               23.3,
               29,
               23.2,
               26.9,
               24,
               29.1,
               26.9
               ]
        count = 1
        self.dict = {}  # Empty dictionary to add values into
        for i in self.week:
            self.dict['sub-' + str(count).zfill(3)] = i
            count += 1
        list_path = os.path.join(self._data_dir,self.list_name)
        with open(list_path, "r") as f:
            for line in f.readlines():
                line = line.strip('\n')  # 去掉列表中每一个元素的换行符
                self.sample_list.append(line)
        logging.info(f'Creating total {self.mode} dataset with {len(self.sample_list)} examples')

    def __len__(self):
        return len(self.sample_list)

    def __sampleList__(self):
        return self.sample_list

    def dataloadering_atlas(self,case):
        import SimpleITK as sitk
        from scipy import ndimage
        img_np_path = os.path.join(self.data_dir_s, 'image/{}_srr.nii'.format(case))
        mask_np_path = self.data_dir_s + '/label/' + case + '_parcellation.nii'
        vol = sitk.ReadImage(img_np_path, sitk.sitkFloat32)
        image = sitk.GetArrayFromImage(vol)
        if self.crop:
            image1 = np.expand_dims(image, 0)
            nonzero_mask = create_nonzero_mask(image1)
            bbox = get_bbox_from_mask(nonzero_mask, 0)
            image = crop_to_bbox(image, bbox)
        image = normalization(image)
        if self.zoom:
            image = ndimage.zoom(image, zoom=
            (self.patch_size[0] / (image.shape[0]),
             self.patch_size[1] / (image.shape[1]),
             self.patch_size[2] / (image.shape[2])), order=3)
        image = np.expand_dims(image, 0)

        mask = sitk.ReadImage(mask_np_path, sitk.sitkUInt8)
        mask = sitk.GetArrayFromImage(mask)
        mask[mask == 4] = 11
        mask[mask == 5] = 12
        mask[mask == 3] = 15
        mask[mask == 1] = 13
        mask[mask == 2] = 14
        mask[mask == 8] = 12
        mask[mask == 11] = 1
        mask[mask == 12] = 2
        mask[mask == 15] = 5
        mask[mask == 13] = 3
        mask[mask == 14] = 4
        if self.crop:
            mask = crop_to_bbox(mask, bbox)
        if self.zoom:
            mask = ndimage.zoom(mask, zoom=
            (self.patch_size[0] / (mask.shape[0]),
             self.patch_size[1] / (mask.shape[1]),
             self.patch_size[2] / (mask.shape[2])), order=0)
        mask = np.expand_dims(mask, 0)

        sample = {'image_source': image.copy(), 'mask_source': mask.copy()}
        return sample

    def __getitem__(self, idx):
        if "feta" in self._data_dir:
            case = self.sample_list[idx]
            self.age = round(self.dict[case[:7]])
            if case[:7] in self.illness or case[:7] in self.illness_t:
                self.pathological = 1
            else:
                self.pathological = 0
            img_np_path = os.path.join(self._data_dir, 'image/{}'.format(case))
            mask_np_path = self._data_dir + '/label/' + case[:-7] + 'dseg.nii'

            import SimpleITK as sitk
            from scipy import ndimage
            vol = sitk.ReadImage(img_np_path, sitk.sitkFloat32)
            image = sitk.GetArrayFromImage(vol)
            mask = sitk.ReadImage(mask_np_path, sitk.sitkUInt8)
            mask = sitk.GetArrayFromImage(mask)
            map = mask > 0
            image = image*map
            if self.crop:
                image1 = np.expand_dims(image, 0)
                nonzero_mask = create_nonzero_mask(image1)
                bbox = get_bbox_from_mask(nonzero_mask, 0)
                image = crop_to_bbox(image, bbox)
            image = normalization(image)
            if self.zoom:
                image = ndimage.zoom(image, zoom=
                (self.patch_size[0]/(image.shape[0]),
                 self.patch_size[1]/(image.shape[1]),
                 self.patch_size[2]/(image.shape[2])), order=3)
            image = np.expand_dims(image, 0)
            # if random.uniform(0,1) > 0.5:
            #     image = image[...,::-1]
            if 'train' in self.mode:
                if self.crop:
                    mask = crop_to_bbox(mask, bbox)
                if self.zoom:
                    mask = ndimage.zoom(mask, zoom=
                    (self.patch_size[0] / (mask.shape[0]),
                     self.patch_size[1] / (mask.shape[1]),
                     self.patch_size[2] / (mask.shape[2])), order=0)
            mask = np.expand_dims(mask, 0)
            sample = {'image': image.copy(), 'mask': mask.copy(), 'idx': case}
            if self.transform:
                sample = self.transform(sample)
                image0 = sample['image']
                # mask0 = sample['mask']
                # mas0k = mask0.astype(np.uint8)
                sample = {'image': image.copy(), 'mask': mask.copy(), 'idx': case}
                sample['image0'] = image0
            sample['age'] = self.age
            sample['pathological'] = self.pathological
            if self.crop:
                sample['bbox'] = bbox
            if self.atlas:
                if self.pathological:
                    case_age = self.age - 6
                    if case_age < 15:
                        case_age = 15
                    if 29 > case_age > 19:
                        case_age += 1
                    sample3 = self.dataloadering_atlas(str(case_age).zfill(3))
                    sample['image_source'] = sample3['image_source']
                    sample['mask_source'] = sample3['mask_source']
                    if 30 > case_age > 20:
                        case_age -= 1
                    case_age += 6
                    sample['case_age'] = case_age
                else:
                    randn_num = random.randint(0, 1)
                    if randn_num:
                        case_age = self.age - 21
                        if case_age < 1:
                            case_age = 1
                        sample2 = self.dataloadering_atlas(str(case_age).zfill(3))
                        sample['image_source'] = sample2['image_source']
                        sample['mask_source'] = sample2['mask_source']
                        case_age += 21
                        sample['case_age'] = case_age
                    else:
                        case_age = self.age + 9
                        if case_age < 30:
                            case_age = 30
                        sample4 = self.dataloadering_atlas(str(case_age).zfill(3))
                        sample['image_source'] = sample4['image_source']
                        sample['mask_source'] = sample4['mask_source']
                        case_age -= 9
                        sample['case_age'] = case_age
            if self.registered:
                if self.pathological:
                    mask_np_path = self._data_dir + '/registered/' + case[:-7] + 'dseg_2.nii'
                    if not os.path.exists(mask_np_path):
                        mask_np_path = self._data_dir + '/registered/' + case[:-7] + 'dseg_3.nii'
                else:
                    mask_np_path = self._data_dir + '/registered/' + case[:-7] + 'dseg_3.nii'
                mask = sitk.ReadImage(mask_np_path, sitk.sitkUInt8)
                mask = sitk.GetArrayFromImage(mask)
                if self.crop:
                    mask = crop_to_bbox(mask, bbox)
                if self.zoom:
                    mask = ndimage.zoom(mask, zoom=
                    (self.patch_size[0] / (mask.shape[0]),
                     self.patch_size[1] / (mask.shape[1]),
                     self.patch_size[2] / (mask.shape[2])), order=0)
                sample['registered'] = mask.copy()
            return sample
        elif "atlases" in self._data_dir:
            case = self.sample_list[idx]
            img_np_path = os.path.join(self._data_dir, 'image/{}'.format(case))
            mask_np_path = self._data_dir + '/label/' + case[:-7] + 'parcellation.nii'
            self.age = int(case[:-8])
            if 14 < self.age < 30:
                self.pathological = 1
            else:
                self.pathological = 0
            if self.age < 15:
                self.age += 21
            elif self.age < 30:
                if self.age < 20:
                    self.age += 6
                else:
                    self.age += 5
            else:
                self.age -= 9
            import SimpleITK as sitk
            from scipy import ndimage
            vol = sitk.ReadImage(img_np_path, sitk.sitkFloat32)
            image = sitk.GetArrayFromImage(vol)
            if self.crop:
                image1 = np.expand_dims(image, 0)
                nonzero_mask = create_nonzero_mask(image1)
                bbox = get_bbox_from_mask(nonzero_mask, 0)
                image = crop_to_bbox(image, bbox)
            image = normalization(image)
            if self.zoom:
                image = ndimage.zoom(image, zoom=
                (self.patch_size[0] / (image.shape[0]),
                 self.patch_size[1] / (image.shape[1]),
                 self.patch_size[2] / (image.shape[2])), order=3)
            image = np.expand_dims(image, 0)

            mask = sitk.ReadImage(mask_np_path, sitk.sitkUInt8)
            mask = sitk.GetArrayFromImage(mask)
            mask[mask == 4] = 11
            mask[mask == 5] = 12
            mask[mask == 3] = 15
            mask[mask == 1] = 13
            mask[mask == 2] = 14
            mask[mask == 8] = 12
            mask[mask == 11] = 1
            mask[mask == 12] = 2
            mask[mask == 15] = 5
            mask[mask == 13] = 3
            mask[mask == 14] = 4
            if self.crop:
                mask = crop_to_bbox(mask, bbox)
            if self.zoom:
                mask = ndimage.zoom(mask, zoom=
                (self.patch_size[0] / (mask.shape[0]),
                 self.patch_size[1] / (mask.shape[1]),
                 self.patch_size[2] / (mask.shape[2])), order=0)
            mask = np.expand_dims(mask, 0)

            sample = {'image': image.copy(), 'mask': mask.copy(), 'idx': case}
            if self.transform:
                sample = self.transform(sample)
                image = sample['image']
                image = image.astype(np.float32)
                mask = sample['mask']
                mask = mask.astype(np.uint8)
                sample = {'image': image.copy(), 'mask': mask.copy(), 'idx': case}
            if self.crop:
                sample = {'image': image.copy(), 'mask': mask.copy(), 'idx': case, 'bbox': bbox, 'age': self.age, 'pathological': self.pathological}
            return sample
        elif "registered" in self._data_dir:
            case = self.sample_list[idx]
            def sampling(num,case):
                img_np_path = os.path.join(self._data_dir, 'image/{}'.format(case[:-7])) +'T2w_'+str(num)+'.nii.gz'
                mask_np_path = self._data_dir + '/label/' + case[:-7] + 'dseg_'+str(num)+'.nii.gz'
                import SimpleITK as sitk
                from scipy import ndimage
                vol = sitk.ReadImage(img_np_path, sitk.sitkFloat32)
                image = sitk.GetArrayFromImage(vol)
                mask = sitk.ReadImage(mask_np_path, sitk.sitkUInt8)
                mask = sitk.GetArrayFromImage(mask)
                if self.crop:
                    image1 = np.expand_dims(image, 0)
                    nonzero_mask = create_nonzero_mask(image1)
                    bbox = get_bbox_from_mask(nonzero_mask, 0)
                    image = crop_to_bbox(image, bbox)
                    if 'train' in self.mode:
                        mask = crop_to_bbox(mask, bbox)
                image = normalization(image)
                if self.zoom:
                    image = ndimage.zoom(image, zoom=
                    (self.patch_size[0] / (image.shape[0]),
                     self.patch_size[1] / (image.shape[1]),
                     self.patch_size[2] / (image.shape[2])), order=3)
                    if 'train' in self.mode:
                        mask = ndimage.zoom(mask, zoom=
                        (self.patch_size[0] / (mask.shape[0]),
                         self.patch_size[1] / (mask.shape[1]),
                         self.patch_size[2] / (mask.shape[2])), order=0)
                image = np.expand_dims(image, 0)
                mask = np.expand_dims(mask, 0)
                return image,mask

            image, mask = sampling(3, case)
            # print(image.shape)
            sample = {'image': image.copy(), 'mask': mask.copy(), 'idx': case}
            img_np_path = os.path.join(self._data_dir, 'image/{}'.format(case[:-7])) + 'T2w_1.nii.gz'
            if os.path.exists(img_np_path):
                image1, mask1 = sampling(1, case)
                image = np.concatenate((image1, image))
                # print(image.shape)

                mask = np.concatenate((mask1, mask))
                sample = {'image': image.copy(), 'mask': mask.copy(), 'idx': case}
            if self.transform:
                sample = self.transform(sample)
                image = sample['image']
                image = image.astype(np.float32)
                mask = sample['mask']
                mask = mask.astype(np.uint8)
                sample = {'image': image.copy(), 'mask': mask.copy(), 'idx': case}
            # if self.crop:
            #     sample = {'image': image.copy(), 'mask': mask.copy(), 'idx': case, 'bbox': bbox, 'age': self.age, 'pathological': self.pathological}
            return sample


def normalization(image):
    pixels = image
    mean = pixels.mean()
    std = pixels.std()
    image -= mean
    image /= std
    return image


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


