import sys, os

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os


# Add this function from FedDG
# Assume inputs are 28x28x3 source, 3x28x28 amplitude
def source_to_target_freq(src_img, amp_trg, L=0.1, ratio=0):
    src_img = src_img.transpose((2, 0, 1))

    fft_src = np.fft.fft2(src_img, axes=(-2, -1))

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src), np.angle(fft_src)

    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np(amp_src, amp_trg, L, ratio)

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp(1j * pha_src)

    # get the mutated image
    src_in_trg = np.fft.ifft2(fft_src_, axes=(-2, -1))
    src_in_trg = np.real(src_in_trg)

    src_in_trg = src_in_trg.transpose((1, 2, 0))
    return src_in_trg


# Add this function from FedDG
# TODO assume input is (3, 28, 28)
def low_freq_mutate_np(amp_src, amp_trg, L=0.1, ratio=0):
    if amp_src.shape != amp_trg.shape:
        raise ValueError("src shape {} and trg shape {} is not same.".format(amp_src.shape, amp_trg.shape))

    a_src = np.fft.fftshift(amp_src, axes=(-2, -1))
    a_trg = np.fft.fftshift(amp_trg, axes=(-2, -1))

    h, w, _ = a_src.shape
    b = (np.floor(np.amin((h, w)) * L)).astype(int)
    c_h = np.floor(h / 2.0).astype(int)
    c_w = np.floor(w / 2.0).astype(int)

    h1 = c_h - b
    h2 = c_h + b + 1
    w1 = c_w - b
    w2 = c_w + b + 1

    a_src[h1:h2, w1:w2, :] = a_src[h1:h2, w1:w2, :] * ratio + a_trg[h1:h2, w1:w2, :] * (1 - ratio)
    a_src = np.fft.ifftshift(a_src, axes=(-2, -1))
    return a_src


digits_domain_list = ["MNIST", "MNIST_M", "SVHN", "SynthDigits", "USPS"]


class DigitsDataset(Dataset):
    def extract_freqs_from_image(self, image): # assume input is PIL.Image of 28x28x3
        image = np.asarray(image, np.float32) # convert to array
        image = image.transpose((2, 0, 1)) # transpose to 3x28x28

        fft_image = np.fft.fft2(image, axes=(-2, -1))
        fft_abs, _ = np.abs(fft_image), np.angle(fft_image)

        return fft_abs

    def blend_with_cross_domain_freq(self, image):
        number_of_domains = len(digits_domain_list)
        target_domain = np.random.choice(number_of_domains) # select domain
        target_images = self.cross_domain_trainsets[target_domain].get_images() # get images

        target_index = np.random.choice(target_images.shape[0]) # select index
        target_sample = target_images[target_index] # get image array
        target_sample = self.cross_domain_trainsets[target_domain].transform_image(target_sample) # input array, return transformed image
        target_sample = transforms.ToPILImage()(target_sample)  # convert to PIL.Image
        target_sample = np.asarray(target_sample, np.float32)  # convert to array

        target_freq = self.extract_freqs_from_image(target_sample)

        image = source_to_target_freq(image, target_freq, L=0.1, ratio=0.5)
        image = np.clip(image, 0, 255)

        return image

    def get_channels(self):
        return self.channels

    def get_images(self):
        return self.images

    def set_trainsets(self, cross_domain_trainsets):
        self.cross_domain_trainsets = cross_domain_trainsets

    def __init__(self, data_path, channels, percent=0.1, filename=None, train=True, transform=None):
        self.is_train = train

        if filename is None:
            if train:
                if percent >= 0.1:
                    for part in range(int(percent * 10)):
                        if part == 0:
                            self.images, self.labels = np.load(
                                os.path.join(data_path, 'partitions/train_part{}.pkl'.format(part)), allow_pickle=True)
                            # TODO add load freq here
                            # self.freqs, _ = np.load(os.path.join(data_path, 'freq_amp_npy/train_part{}.pkl'.format(part)), allow_pickle=True)
                            # self.freqs = self.extract_freqs_from_image(self.images)
                        else:
                            images, labels = np.load(
                                os.path.join(data_path, 'partitions/train_part{}.pkl'.format(part)), allow_pickle=True)
                            # TODO add load freq here
                            # freqs, _ = np.load(os.path.join(data_path, 'freq_amp_npy/train_part{}.pkl'.format(part)), allow_pickle=True)
                            # freqs = self.extract_freqs_from_image(images)

                            self.images = np.concatenate([self.images, images], axis=0)
                            self.labels = np.concatenate([self.labels, labels], axis=0)
                            # TODO concatenate freq here
                            # self.freqs, _ = np.concatenate([self.freqs, freqs], axis=0)
                else:
                    self.images, self.labels = np.load(os.path.join(data_path, 'partitions/train_part0.pkl'),
                                                       allow_pickle=True)
                    # TODO add load freq here
                    # self.freqs, _ = np.load(os.path.join(data_path, 'freq_amp_npy/train_part0.pkl'), allow_pickle=True)
                    # self.freqs = self.extract_freqs_from_image(self.images)

                    data_len = int(self.images.shape[0] * percent * 10)
                    self.images = self.images[:data_len]
                    self.labels = self.labels[:data_len]
                    # TODO cut freqs by using data_len here
                    # self.freqs, _ = self.freqs[:data_len]
            else:
                self.images, self.labels = np.load(os.path.join(data_path, 'test.pkl'), allow_pickle=True)
                # self.freqs = []
        else:
            self.images, self.labels = np.load(os.path.join(data_path, filename), allow_pickle=True)
            # TODO add load freq here, may have problems here
            # TODO replace "partitions/XXX" with "freq_amp_npy/XXX"
            # filename = filename.replace("partitions", "freq_amp_npy")
            # self.freqs, _ = np.load(os.path.join(data_path, filename), allow_pickle=True)
            # self.freqs = self.extract_freqs_from_image(self.images)

        self.transform = transform
        self.channels = channels
        self.labels = self.labels.astype(np.long).squeeze()
        # self.trainsets = []

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        image = self.transform_image(image)

        # TODO load a random freq here, and blend with image
        if self.is_train:  # only do this in training phase, not in test
            image = transforms.ToPILImage()(image) # convert to PIL.Image
            image = np.asarray(image, np.float32) # convert to array
            image = self.blend_with_cross_domain_freq(image) # input array, return array
            image = self.transform_image(image)

        return image, label

    def transform_image(self, image): # input array, output transformed image
        if self.channels == 1:
            if len(image.shape) == 2:
                image = Image.fromarray(image, mode='L')
            else:
                image = image[:, :, 0]
                image = Image.fromarray(image, mode='L')
        elif self.channels == 3:
            image = Image.fromarray(image, mode='RGB')
        else:
            raise ValueError("{} channel is not allowed.".format(self.channels))

        if self.transform is not None:
            image = self.transform(image)

        return image


class OfficeDataset(Dataset):
    def __init__(self, base_path, site, train=True, transform=None):
        if train:
            self.paths, self.text_labels = np.load('../data/office_caltech_10/{}_train.pkl'.format(site),
                                                   allow_pickle=True)
        else:
            self.paths, self.text_labels = np.load('../data/office_caltech_10/{}_test.pkl'.format(site),
                                                   allow_pickle=True)

        label_dict = {'back_pack': 0, 'bike': 1, 'calculator': 2, 'headphones': 3, 'keyboard': 4, 'laptop_computer': 5,
                      'monitor': 6, 'mouse': 7, 'mug': 8, 'projector': 9}
        self.labels = [label_dict[text] for text in self.text_labels]
        self.transform = transform
        self.base_path = base_path if base_path is not None else '../data'

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.labels[idx]
        image = Image.open(img_path)

        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class DomainNetDataset(Dataset):
    def __init__(self, base_path, site, train=True, transform=None):
        if train:
            self.paths, self.text_labels = np.load('../data/DomainNet/{}_train.pkl'.format(site), allow_pickle=True)
        else:
            self.paths, self.text_labels = np.load('../data/DomainNet/{}_test.pkl'.format(site), allow_pickle=True)

        label_dict = {'bird': 0, 'feather': 1, 'headphones': 2, 'ice_cream': 3, 'teapot': 4, 'tiger': 5, 'whale': 6,
                      'windmill': 7, 'wine_glass': 8, 'zebra': 9}

        self.labels = [label_dict[text] for text in self.text_labels]
        self.transform = transform
        self.base_path = base_path if base_path is not None else '../data'

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.labels[idx]
        image = Image.open(img_path)

        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label
