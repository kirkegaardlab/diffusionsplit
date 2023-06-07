import numpy as np
import torch
import glob
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import torchvision.transforms.functional as F
from data.utils import load_data, to_tensors, rotated_rect_with_max_area

dir_path = os.path.dirname(os.path.realpath(__file__))

class CellposeOverlap(Dataset):
    def __init__(self, max_n=None, test=False, device='cuda', augment=True,
                 direc=os.path.join(dir_path, 'overlapdataset'), mask_identifier='_mask',
                 train_crop_size=192):

        if test:
            direc = os.path.join(direc, 'test/')
        else:
            direc = os.path.join(direc, 'train/')


        self.ims = []
        self.all_labels = []
        self.test = test

        self.augment = augment
        self.max_labels = 0

        if test:
            self.crop_size = None
        else:
            self.crop_size = train_crop_size
        self.warning_printed = False

        fnames = sorted([x for x in glob.glob(direc + '*.png') if mask_identifier not in x])
        for fname in tqdm(fnames, desc='Loading data'):
            im, labels = to_tensors(load_data(fname))

            if len(im.shape) == 3:
                assert im.shape[2] < im.shape[0] and im.shape[2] < im.shape[1]
                im = torch.swapaxes(torch.swapaxes(im, 1, 2), 0, 1)

            self.ims.append(im.to(device))
            self.all_labels.append(labels.to(device))
            self.max_labels = max(self.max_labels, labels.shape[2])

            if len(self.ims) == max_n:
                break

        if len(self.ims) == 0:
            print('warning: no images were found')

    def __len__(self):
        return len(self.ims)

    def transform(self, im, labels):
        labels = torch.permute(labels, [2, 0, 1])

        if self.augment:
            if np.random.random() < 0.8:
                # Rotation
                theta = 360 * np.random.random()
                im = F.rotate(im, theta, interpolation=F.InterpolationMode.BILINEAR)
                labels = F.rotate(labels, theta)

                # Rotation crop
                d = rotated_rect_with_max_area(im.shape[1], im.shape[2], theta)
                im = im[:, d['y_min']:d['y_max'], d['x_min']:d['x_max']]
                labels = labels[:, d['y_min']:d['y_max'], d['x_min']:d['x_max']]

        # Crop
        if self.crop_size is not None:
            ux = np.random.randint(0, im.shape[1] - self.crop_size)
            uy = np.random.randint(0, im.shape[2] - self.crop_size)

            im = im[:, ux:(ux + self.crop_size), uy:(uy + self.crop_size)]
            labels = labels[:, ux:(ux + self.crop_size), uy:(uy + self.crop_size)]

        if self.augment:
            # Flip
            if np.random.random() < 0.5:
                im = torch.flip(im, dims=(1,))
                labels = torch.flip(labels, dims=(1,))

            if np.random.random() < 0.5:
                im = torch.flip(im, dims=(2,))
                labels = torch.flip(labels, dims=(2,))

        _, n, m = im.shape
        if n % 16 != 0 or m % 16 != 0:
            nn = n - (n % 16)
            nm = m - (m % 16)
            if not self.warning_printed:
                print(f'warning: image shape not divisble by 16 - cropping to {nn} x {nm}')
                self.warning_printed = True

            im = im[:, :nn, :nm]
            labels = labels[:, :nn, :nm]

        labels = torch.permute(labels, [1, 2, 0])

        return im, labels

    def __getitem__(self, idx):
        im = self.ims[idx]
        labels = self.all_labels[idx]

        im, labels = self.transform(im, labels)

        # I think this is unnecessarily slow:
        labels = torch.nn.functional.pad(labels, (0, self.max_labels - labels.shape[2]))
        # segmentation = labels.max(dim=2)[0]

        if len(im.shape) == 2:
            im = im[None]

        return im, labels


if __name__ == '__main__':
    data = CellposeOverlap(max_n=5, test=False, augment=True)
    print(data[0][0].shape, data[0][1].shape)

    data = CellposeOverlap(max_n=5, test=True, augment=False)
    print(data[0][0].shape, data[0][1].shape)

