import argparse
import os
import time
from typing import Optional, Callable

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from skimage import morphology, restoration
from torch import optim
from torch.utils.data import ConcatDataset
from torchvision import datasets, transforms
import torch.utils.data
from torchvision.io import read_image
import tifffile as tiff
from matplotlib import image as plt_image


class INSAR(datasets.VisionDataset):
    def __init__(self, root, transform: Optional[Callable] = None,
                 is_valid_file: Optional[Callable[[str], bool]] = None):
        super().__init__(root, transform)
        self.root = root
        if is_valid_file is not None:
            files = np.array(os.listdir(self.root))
            self.samples = files[np.vectorize(is_valid_file)(files)]
        else:
            self.samples = np.array(os.listdir(self.root))

        if self.samples.size == 0:
            raise ValueError("Found no samples in the folder")

        self.samples = np.vectorize(lambda x: self.root + '/' + x)(self.samples)

        if transform is None:
            self.transform = transforms.Compose([])
        else:
            self.transform = transform

        if 'png' in self.samples[0] or 'jpeg' in self.samples[0]:
            self.loader = read_image
            # self.loader = plt_image.imread
        elif 'tif' in self.samples[0]:
            self.loader = tiff.imread
            self.transform = transforms.Compose([transforms.ToTensor(), self.transform])
        else:
            raise ValueError("File type not supported")

        self.len = self.samples.size

        # self.turb = AddTurbulance()
        # self.deform = AddDeform()

    def __getitem__(self, index):
        sample = self.loader(self.samples[index])
        label = 0
        # sample, label = self.deform(self.turb(sample))

        sample = self.transform(sample)
        # return sample, self.samples[index], label
        return sample, self.samples[index], label

    def __len__(self):
        return self.len


class SetRange(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        return img * 2 - 1.

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class PatchesCrop(torch.nn.Module):
    def __init__(self, image_size):
        super().__init__()
        self.image_size = image_size

    def forward(self, img):
        channels = img.shape[0]
        return img\
            .unfold(1, self.image_size, self.image_size)\
            .unfold(2, self.image_size, self.image_size)\
            .reshape(-1, channels, self.image_size, self.image_size)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(image_size={self.image_size})"


class OverlappingPatchesCrop(torch.nn.Module):
    def __init__(self, image_size, stride):
        super().__init__()
        self.image_size = image_size
        self.stride = stride

    def forward(self, img):
        channels = img.shape[0]
        return img\
            .unfold(1, self.image_size, self.stride)\
            .unfold(2, self.image_size, self.stride)\
            .reshape(-1, channels, self.image_size, self.image_size)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(image_size={self.image_size}, stride={self.stride})"


class Normalize(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        return img - torch.nanmean(img)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Clamp(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, img):
        return torch.clamp(img, min=self.min, max=self.max)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(min={self.min}, max={self.max})"


class BringToInterval(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.min = min
        self.max = max
        self.range = max - min

    def forward(self, img):
        return (img - self.min) / self.range

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(min={self.min}, max={self.max})"


class RemoveNAN(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        return torch.nan_to_num(img)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

class Interpolate(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.water = tiff.imread('datasets/taal/taal_032D_07536_111313_mask_water.tif').astype(bool)

    def forward(self, img):
        image = img.copy()

        badarea = (image == 0) + np.isnan(image) + self.water
        imageblur = cv2.medianBlur(image.copy(), 5)
        image[image > 200] = imageblur[image > 200]
        image[image < -200] = imageblur[image < -200]
        mask = morphology.binary_closing(badarea > 0, morphology.disk(5))
        mask = morphology.binary_dilation(mask, morphology.disk(2))
        # shift with bias
        refimg = image.copy()
        refimg[mask + (image > 200) + (image < -200)] = np.nan
        # refimg[mask] = np.nan
        refval = np.nanmean(refimg)
        image = image - refval
        image = np.round((image + 50) / 100 * 255)
        image[image < 0] = 0
        image[image > 255] = 255
        mask = np.array((mask) * 255, dtype='uint8')
        image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        image = cv2.medianBlur(image, 3)
        mask = mask > 0
        # badpixel = morphology.binary_closing((imagecrop == 0) + np.isnan(imagecrop), morphology.disk(2))
        # if (np.count_nonzero(badpixel) > 0):
        #     # image = cv2.imread(img_name, -1)
        #     image = img.copy()
        #     refimg = image.copy()
        #     refimg[mask + (image > 200) + (image < -200)] = np.nan
        #     # refimg[mask] = np.nan
        #     refval = np.nanmean(refimg)
        #     image = image - refval
        #     image = np.round((image + 50) / 100 * 255)
        #     image[image < 0] = 0
        #     image[image > 255] = 255
        #     cnum = 0
        #     while (np.count_nonzero(badpixel) > 0) and (cnum < 10):
        #         badarea = (image == 0) + np.isnan(image)
        #         mask = morphology.binary_closing(badarea > 0, morphology.disk(3))
        #         mask = morphology.binary_dilation(mask, morphology.disk(2))
        #         mask = np.array((mask) * 255, dtype='uint8')
        #         image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        #         imagecrop = image[250 - 113:250 + 114, 250 - 113:250 + 114]
        #         badpixel = morphology.binary_closing((imagecrop == 0) + np.isnan(imagecrop), morphology.disk(2))
        #         cnum = cnum + 1
        #     image = cv2.medianBlur(image, 3)

        mask = morphology.binary_dilation(badarea > 0, morphology.disk(2))
        mask = mask.astype(np.bool)
        img[mask] = image[mask]
        return np.stack([img, mask], axis=2)

    # def forward(self, img):
    #     # img = img.view(500, 500).cpu().numpy()
    #     # mask = (np.isnan(img) + img == 0).astype(np.bool)
    #     mask = np.isnan(img).astype(bool) | (img == 0).astype(bool) | self.water
    #     mask = (mask > 0).astype(np.uint8)
    #
    #     # mask = morphology.binary_dilation(mask, morphology.disk(2))
    #     # mask = morphology.binary_closing(mask, morphology.disk(5))
    #
    #     # # mask = mask.astype(np.uint8)
    #     # mask = morphology.binary_dilation(mask, morphology.disk(2))
    #     # mask = morphology.binary_closing(mask, morphology.disk(5))
    #     mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=1)
    #     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    #     dst = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    #     # dst = cv2.medianBlur(dst, 3)
    #     # dst = restoration.inpaint_biharmonic(img, mask)
    #
    #     mask = np.isnan(img).astype(bool) | (img == 0).astype(bool) | self.water
    #     mask = morphology.binary_dilation(mask.astype(np.uint8), morphology.disk(2))
    #     mask = mask.astype(bool)
    #     # mask = morphology.binary_closing(mask, morphology.disk(5))
    #     img[mask > 0] = dst[mask > 0]
    #
    #     img = (img + 30) / 50
    #     img[img > 1] = 1
    #     img[img < 0] = 0
    #
    #     return np.stack([img, mask], axis=2)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(interpolate_biharmonic)"


class GenerateWrap(torch.nn.Module):
    def __init__(self, random=True):
        super().__init__()
        self.random = random
        self.sea = tiff.imread('datasets/agung/agung_156A_09814_081406_mask_sea.tif') == 0

    def forward(self, img):
        img = torch.nan_to_num(img)
        if self.random:
            rand_number = 2 * torch.pi * torch.rand(1).item()
        else:
            rand_number = 0
        img = (img + rand_number) % (2 * torch.pi) - torch.pi
        img[0, :][self.sea] = 0
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(random={self.random})"

class AddTurbulance(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.sea = tiff.imread('datasets/agung/agung_156A_09814_081406_mask_sea.tif') == 0
        self.folder = 'datasets/generated/set1/unwrap/turbulent/'
        self.turbs = os.listdir(self.folder)
        self.size = len(self.turbs)
        self.c = 0

    def forward(self, img):
        turb = loadmat(self.folder + self.turbs[self.c])['curTur']
        self.c += 1
        if self.c >= self.size:
            self.c = 0
        mask = img == 0
        turb = turb / 60

        img += turb

        img[mask] = 0
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


class AddDeform(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.sea = tiff.imread('datasets/agung/agung_156A_09814_081406_mask_sea.tif') == 0
        self.folder = 'datasets/generated/set1/unwrap/deform/'
        self.loss = os.listdir(self.folder)
        self.size = len(self.loss)
        self.c = 0

    def forward(self, img):
        if 0.5 < torch.rand(1):
            los = loadmat(self.folder + self.loss[self.c])['los_grid']
            self.c += 1
            if self.c >= self.size:
                self.c = 0
            mask = img == 0
            los = los / 60

            img += los

            img[mask] = 0
            return img, 1
        else:
            return img, 0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


class InSARNormalize(torch.nn.Module):
    def __init__(self, min=-30, range=60):
        super().__init__()
        self.sea = tiff.imread('datasets/agung/agung_156A_09814_081406_mask_sea.tif') == 0
        self.min = min
        self.diff = range

    def forward(self, img):
        mask = self.sea + np.isnan(img)

        img -= np.nanmean(img[~mask])

        img = (img - self.min) / self.diff
        img[img > 1] = 1
        img[img < 0] = 0

        img[mask] = 0

        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


class Crop(torch.nn.Module):
    def __init__(self, x1=89, x2=217, y1=201, y2=329):
        super().__init__()
        # defaults for Agung
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def forward(self, img):
        return img[:, self.x1:self.x2, self.y1:self.y2]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(x1={self.x1}, x2={self.x2}, y1={self.y1}, y2={self.y2})"


def get_transforms(image_size, logger=None, test=False):
    # for wrap interval -pi + pi/ rezise / set range
    if test:
        transform_obj = transforms.Compose([
            # InSARNormalize(),
            # transforms.ToTensor(),
            # RemoveNAN(),
            # Normalize(),/
            # Clamp(-30, 30),
            # BringToInterval(-30, 30),
            # BringToInterval(-torch.pi, torch.pi),
            # SetRange(),
            # Interpolate(),
            # transforms.Grayscale(),
            # BringToInterval(0, 255),
            # transforms.ToTensor(),
            # transforms.Resize((image_size, image_size)),
            # transforms.Pad([14, 14, 15, 15]),
            # GenerateWrap(random=False),
            # transforms.Grayscale(3),
            transforms.ConvertImageDtype(torch.float),
            # transforms.Resize((image_size, image_size)),
            transforms.Pad(6, padding_mode='edge'),
            # Crop(),
            # RemoveNAN(),
            # PatchesCrop(image_size),
            # OverlappingPatchesCrop(image_size, int(image_size/2)),
            # transforms.CenterCrop(image_size),
            # transforms.Resize((227, 227)),
            # transforms.ToTensor(),
            # transforms.ConvertImageDtype(torch.float),
            # BringToInterval(0, 255),
            # Normalize(),
            # bring_to_interval,
            SetRange(),
        ])
    else:
        transform_obj = transforms.Compose([
            # InSARNormalize(),
            # transforms.ToTensor(),
            # RemoveNAN(),
            # Normalize(),
            # Clamp(-30, 30),
            # BringToInterval(-30, 30),
            # BringToInterval(-torch.pi, torch.pi),
            # SetRange(),
            # Interpolate(),
            # transforms.ToTensor(),
            # Clamp(0, 1),
            # transforms.Grayscale(),
            # BringToInterval(0, 255),
            # transforms.Pad([14, 14, 15, 15]),
            # GenerateWrap(),
            # transforms.Grayscale(3),
            transforms.ConvertImageDtype(torch.float),
            # AddTurbulance(),
            # AddDeform(),
            # transforms.Pad(6),
            # RemoveNAN(),
            transforms.RandomCrop(image_size),
            # transforms.RandomRotation(180),
            # transforms.RandomAdjustSharpness(2),
            # transforms.RandomAdjustSharpness(0),
            # transforms.RandomAutocontrast(),
            # transforms.RandomErasing(),
            # Crop(),
            # transforms.Resize((image_size, image_size)),
            # transforms.Resize((227, 227)),
            # transforms.ToTensor(),
            # transforms.ConvertImageDtype(torch.float),
            # Normalize(),
            # bring_to_interval,
            # BringToInterval(0, 255),
            SetRange(),
        ])

    if logger is not None:
        trs = str(transform_obj).split('\n')[1:-1]
        log = ''
        for tr in trs:
            log += tr.split('    ')[-1]
            log += '->'
        log = log[:-2]
        logger['preprocessing'] = log
    return transform_obj


def load_transforms(image_size, logger, test=False):
    trs = []
    for tr in logger['preprocessing'].split('->'):
        if 'Grayscale' in tr:
            trs.append(transforms.Grayscale())
        elif 'BringToInterval' in tr:
            string_of_args = tr.split('(')[1].split(')')[0]
            args = dict(e.split('=') for e in string_of_args.split(', '))
            args = {k: int(v) for k, v in args.items()}
            trs.append(BringToInterval(**args))
        elif 'Resize' in tr:
            trs.append(transforms.Resize((image_size, image_size)))
        elif 'CenterCrop' in tr:
            str_size = tr.split('(')[1].split(')')[0]
            trs.append(transforms.CenterCrop(int(str_size)))
        elif 'SetRange' in tr:
            trs.append(SetRange())
        elif 'Normalize' in tr:
            trs.append(Normalize())
        elif 'RandomCrop' in tr:
            if test:
                trs.append(PatchesCrop(image_size))
            else:
                trs.append(transforms.RandomCrop(image_size))
        elif 'ConvertImageDtype' in tr:
            trs.append(transforms.ConvertImageDtype(torch.float))
        elif 'ToTensor' in tr:
            trs.append(transforms.ToTensor())
        elif 'Clamp' in tr:
            string_of_args = tr.split('(')[1].split(')')[0]
            args = dict(e.split('=') for e in string_of_args.split(', '))
            args = {k: int(v) for k, v in args.items()}
            trs.append(Clamp(**args))
        elif 'Pad' in tr:
            trs.append(transforms.Pad([6, 6, 6, 6]))
            #    TODO make this one more flexible
        elif 'Interpolate' in tr:
            trs.append(Interpolate())
        #    TODO make this one more flexible
        elif 'GenerateWrap' in tr:
            arg = tr.split('=')[1] == 'True'
            trs.append(GenerateWrap(arg))
        elif 'RemoveNAN' in tr:
            trs.append(RemoveNAN())
        elif 'Crop' in tr:
            string_of_args = tr.split('(')[1].split(')')[0]
            args = dict(e.split('=') for e in string_of_args.split(', '))
            args = {k: int(v) for k, v in args.items()}
            trs.append(Crop(**args))
        elif 'Random' in tr:
            pass
        else:
            raise ValueError(f'Unknown transform {tr}')
    return transforms.Compose(trs)


def load_dataset(data_path, file_filter=None, num_workers=0, batch_size=128, transform_obj=None, shuffle=True):
    dataset = INSAR(
        root=data_path, transform=transform_obj, is_valid_file=file_filter
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle
    )

    return loader


def _get_datasets(train_dir, test_normal_dir, test_abnormal_dir, file_filter=None,
                  image_size=128, num_workers=0, batch_size=128, logger=None, load_from_log=False):
    if load_from_log:
        train_transform = load_transforms(image_size, logger=logger, test=False)
        test_transform = load_transforms(image_size, logger=logger, test=True)
    else:
        # logger['info'] += 'short_interpolate/'
        train_transform = get_transforms(image_size, logger=logger, test=False)
        test_transform = get_transforms(image_size, logger=None, test=True)

    train_dataset = load_dataset(train_dir, file_filter, num_workers=num_workers, batch_size=batch_size,
                                 transform_obj=train_transform, shuffle=True)
    test_normal = load_dataset(test_normal_dir, file_filter, num_workers=num_workers, batch_size=1,
                               transform_obj=test_transform, shuffle=False)
    test_abnormal = load_dataset(test_abnormal_dir, file_filter, num_workers=num_workers, batch_size=1,
                                 transform_obj=test_transform, shuffle=False)
    return train_dataset, test_normal, test_abnormal


def _get_train_dataset_for_testing(train_dir, file_filter=None, image_size=128, num_workers=0, logger=None, load_from_log=False):
    if load_from_log:
        train_transform = load_transforms(image_size, logger=logger, test=True)
    else:
        train_transform = get_transforms(image_size, logger=None, test=True)

    train_dataset = load_dataset(train_dir, file_filter, num_workers=num_workers, batch_size=1,
                                 transform_obj=train_transform, shuffle=False)
    return train_dataset


def _get_full_dataset(train_dir, test_normal_dir, test_abnormal_dir, file_filter=None,
                  image_size=128, num_workers=0, batch_size=128):

    transform_obj = get_transforms(image_size, logger=None, test=False)

    train = INSAR(
        root=train_dir, transform=transform_obj, is_valid_file=file_filter
    )

    test_normal = INSAR(
        root=test_normal_dir, transform=transform_obj, is_valid_file=file_filter
    )

    test_abnormal = INSAR(
        root=test_abnormal_dir, transform=transform_obj, is_valid_file=file_filter
    )

    loader = torch.utils.data.DataLoader(
        ConcatDataset([train, test_normal, test_abnormal]),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )

    return loader


def default_filename_sanitizer(filename: str):
    date1 = filename.split('/')[-1].split('_')[1]
    date2 = filename.split('/')[-1].split('_')[2]
    date1 = date1[:4] + ':' + date1[4:6] + ':' + date1[6:8]
    date2 = date2[:4] + ':' + date2[4:6] + ':' + date2[6:8]
    return date1 + '-' + date2


def unwrap_taal_filename_sanitizer(filename: str):
    date1 = filename.split('/')[3].split('_')[6]
    date2 = filename.split('/')[3].split('_')[7]
    date1 = date1[:4] + ':' + date1[4:6] + ':' + date1[6:8]
    date2 = date2[:4] + ':' + date2[4:6] + ':' + date2[6:8]
    return date1 + '-' + date2


def wrap_taal_filename_sanitizer(filename: str):
    return default_filename_sanitizer(filename)


def unwrap_agung_filename_sanitizer(filename: str):
    return default_filename_sanitizer(filename)


def get_datasets(args, logger=None, load_from_logger=False):
    # dataset = datasets_info[args.dataset]['path']
    dataset = 'datasets/' + args.dataset
    subfolder = args.subfolder

    filter = CCFilter('ccs.csv', args.cc)
    return _get_datasets(dataset + "/train/" + subfolder, dataset + "/test_normal/" + subfolder,
                         dataset + "/test_abnormal/" + subfolder, filter, args.image_size, args.num_workers,
                         args.batch_size, logger, load_from_log=load_from_logger)


def get_train_dataset_for_testing(args, logger=None, load_from_logger=False):
    dataset = 'datasets/' + args.dataset
    subfolder = args.subfolder

    return _get_train_dataset_for_testing(dataset + "/train/" + subfolder, None, args.image_size, args.num_workers,
                                          logger, load_from_log=load_from_logger)


def get_full_dataset(args):
    dataset = 'datasets/' + args.dataset
    subfolder = args.subfolder

    filter = CCFilter('ccs.csv', args.cc)

    return _get_full_dataset(dataset + "/train/" + subfolder, dataset + "/test_normal/" + subfolder,
                         dataset + "/test_abnormal/" + subfolder, filter, args.image_size, args.num_workers,
                         args.batch_size)


# Filter based on coherence
class CCFilter(torch.nn.Module):
    def __init__(self, cc_file, threshold):
        super().__init__()
        self.cc_file = pd.read_csv(cc_file)
        self.threshold = threshold

    def forward(self, img_name):
        name = img_name.split('/')[-1].split('.')[0]
        finds = self.cc_file.loc[self.cc_file['names'] == name].to_dict(orient='records')
        if len(finds) == 0:
            return True
        else:
            return finds[0]['cc'] > self.threshold
