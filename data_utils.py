import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os

import random # for the code from FedDG

# Add this function from FedDG
# TODO
def source_to_target_freq( src_img, amp_trg, L=0.1 ):
    # TODO convert PIL.Image to nparray
    src_img = np.asarray(src_img)
    amp_trg = np.asarray(amp_trg)

    # exchange magnitude
    # input: src_img, trg_img
    # src_img = src_img.transpose((2, 0, 1))
    src_img_np = src_img #.cpu().numpy()
    fft_src_np = np.fft.fft2( src_img_np, axes=(-2, -1) )

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)

    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np( amp_src, amp_trg, L=L )

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp( 1j * pha_src )

    # get the mutated image
    src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
    src_in_trg = np.real(src_in_trg)

    src_in_trg = src_in_trg.transpose(1, 2, 0)

    src_in_trg = Image.fromarray(src_in_trg, mode="RGB")
    return src_in_trg

# Add this function from FedDG
# TODO
def low_freq_mutate_np( amp_src, amp_trg, L=0.1 ):
    a_src = np.fft.fftshift( amp_src, axes=(-2, -1) )
    a_trg = np.fft.fftshift( amp_trg, axes=(-2, -1) )

    _, h, w = a_src.shape
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)
    # print (b)
    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    ratio = random.randint(1,10)/10

    if a_src.shape != a_trg.shape:
        a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
    # a_src[:,h1:h2,w1:w2] = a_src[:,h1:h2,w1:w2] * ratio + a_trg[:,h1:h2,w1:w2] * (1- ratio)
    # a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
    a_src = np.fft.ifftshift( a_src, axes=(-2, -1) )
    # a_trg[:,h1:h2,w1:w2] = a_src[:,h1:h2,w1:w2]
    # a_trg = np.fft.ifftshift( a_trg, axes=(-2, -1) )
    return a_src

digits_domain_list = ["MNIST", "MNIST_M", "SVHN", "SynthDigits", "USPS"]

class DigitsDataset(Dataset):
    def extract_freqs_from_image(self, images):
        fft = np.fft.fft2(images)
        freqs = np.abs(fft)
        return freqs

    def blend_with_cross_domain_freq(self, image):
        # TODO get # of domains
        number_of_domains = len(digits_domain_list)

        # TODO randomly choose k target domains
        k = 1
        for target_domain in np.random.choice(number_of_domains, k):
            # TODO for domain i, randomly choose a freq sample
            target_freq_index = np.random.choice(self.freqs[target_domain].shape[0])
            target_freq_sample = self.freqs[target_domain][target_freq_index]

            if self.trainsets[target_domain].get_channels() == 1:
                target_freq_sample = Image.fromarray(target_freq_sample, mode='L')
            elif self.trainsets[target_domain].get_channels() == 3:
                target_freq_sample = Image.fromarray(target_freq_sample, mode='RGB')
            else:
                raise ValueError("{} channel is not allowed.".format(self.trainsets[target_domain].get_channels()))

            if self.transform is not None:
                target_freq_sample = self.trainsets[target_domain].transform(target_freq_sample)

            # TODO blend freq sample with image
            if image.shape != target_freq_sample.shape:
                print(target_freq_sample.shape)
            image = source_to_target_freq(image, target_freq_sample, L=0.5)
            image = np.clip(image, 0, 255)

        return image

    def get_channels(self):
        return self.channels

    def get_freqs(self):
        return self.freqs

    def set_freqs(self, freqs):
        self.freqs = freqs

    def set_trainsets(self, cross_domain_trainsets):
        self.trainsets = cross_domain_trainsets

    def __init__(self, data_path, channels, percent=0.1, filename=None, train=True, transform=None):
        if filename is None:
            if train:
                if percent >= 0.1:
                    for part in range(int(percent*10)):
                        if part == 0:
                            self.images, self.labels = np.load(os.path.join(data_path, 'partitions/train_part{}.pkl'.format(part)), allow_pickle=True)
                            # TODO add load freq here
                            #self.freqs, _ = np.load(os.path.join(data_path, 'freq_amp_npy/train_part{}.pkl'.format(part)), allow_pickle=True)
                            self.freqs = self.extract_freqs_from_image(self.images)
                        else:
                            images, labels = np.load(os.path.join(data_path, 'partitions/train_part{}.pkl'.format(part)), allow_pickle=True)
                            # TODO add load freq here
                            #freqs, _ = np.load(os.path.join(data_path, 'freq_amp_npy/train_part{}.pkl'.format(part)), allow_pickle=True)
                            freqs = self.extract_freqs_from_image(images)

                            self.images = np.concatenate([self.images,images], axis=0)
                            self.labels = np.concatenate([self.labels,labels], axis=0)
                            # TODO concatenate freq here
                            self.freqs, _ = np.concatenate([self.freqs, freqs], axis=0)
                else:
                    self.images, self.labels = np.load(os.path.join(data_path, 'partitions/train_part0.pkl'), allow_pickle=True)
                    # TODO add load freq here
                    #self.freqs, _ = np.load(os.path.join(data_path, 'freq_amp_npy/train_part0.pkl'), allow_pickle=True)
                    self.freqs = self.extract_freqs_from_image(self.images)

                    data_len = int(self.images.shape[0] * percent*10)
                    self.images = self.images[:data_len]
                    self.labels = self.labels[:data_len]
                    # TODO cut freqs by using data_len here
                    self.freqs, _ = self.freqs[:data_len]
            else:
                self.images, self.labels = np.load(os.path.join(data_path, 'test.pkl'), allow_pickle=True)
                self.freqs = []
        else:
            self.images, self.labels = np.load(os.path.join(data_path, filename), allow_pickle=True)
            # TODO add load freq here, may have problems here
            # TODO replace "partitions/XXX" with "freq_amp_npy/XXX"
            #filename = filename.replace("partitions", "freq_amp_npy")
            #self.freqs, _ = np.load(os.path.join(data_path, filename), allow_pickle=True)
            self.freqs = self.extract_freqs_from_image(self.images)

        self.transform = transform
        self.channels = channels
        self.labels = self.labels.astype(np.long).squeeze()
        #self.trainsets = []

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.channels == 1:
            image = Image.fromarray(image, mode='L')
        elif self.channels == 3:
            image = Image.fromarray(image, mode='RGB')
        else:
            raise ValueError("{} channel is not allowed.".format(self.channels))

        if self.transform is not None:
            image = self.transform(image)

        # TODO load a random freq here, and blend with image
        if len(self.freqs) != 0: # only do this in training phase, not in test
            image = self.blend_with_cross_domain_freq(image)
            image = image.transpose((2, 0, 1))

        return image, label


class OfficeDataset(Dataset):
    def __init__(self, base_path, site, train=True, transform=None):
        if train:
            self.paths, self.text_labels = np.load('../data/office_caltech_10/{}_train.pkl'.format(site), allow_pickle=True)
        else:
            self.paths, self.text_labels = np.load('../data/office_caltech_10/{}_test.pkl'.format(site), allow_pickle=True)
            
        label_dict={'back_pack':0, 'bike':1, 'calculator':2, 'headphones':3, 'keyboard':4, 'laptop_computer':5, 'monitor':6, 'mouse':7, 'mug':8, 'projector':9}
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
            
        label_dict = {'bird':0, 'feather':1, 'headphones':2, 'ice_cream':3, 'teapot':4, 'tiger':5, 'whale':6, 'windmill':7, 'wine_glass':8, 'zebra':9}     
        
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
