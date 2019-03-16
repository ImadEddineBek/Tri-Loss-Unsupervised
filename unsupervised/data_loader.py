from torchvision import datasets
import gzip
import os
import pickle
import urllib

import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms


class USPS(data.Dataset):
    """USPS Dataset.
    Args:
        root (string): Root directory of dataset where dataset file exist.
        train (bool, optional): If True, resample from dataset randomly.
        download (bool, optional): If true, downloads the dataset
            from the internet and puts it in root directory.
            If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that takes in
            an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
    """

    url = "https://github.com/mingyuliutw/CoGAN/raw/master/cogan_pytorch/data/uspssample/usps_28x28.pkl"

    def __init__(self, root, train=True, transform=None, download=True):
        """Init USPS dataset."""
        # init params
        self.root = os.path.expanduser(root)
        self.filename = "usps_28x28.pkl"
        self.train = train
        # Num of Train = 7438, Num ot Test 1860
        self.transform = transform
        self.dataset_size = None

        # download dataset.
        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError("Dataset not found." +
                               " You can use download=True to download it")

        self.train_data, self.train_labels = self.load_samples()
        if self.train:
            total_num_samples = self.train_labels.shape[0]
            indices = np.arange(total_num_samples)
            np.random.shuffle(indices)
            self.train_data = self.train_data[indices[0:self.dataset_size], ::]
            self.train_labels = self.train_labels[indices[0:self.dataset_size]]
        self.train_data *= 255.0
        self.train_data = self.train_data.transpose(
            (0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, label = self.train_data[index, ::], self.train_labels[index]
        if self.transform is not None:
            img = self.transform(img)
        label = torch.LongTensor([np.int64(label).item()])
        # label = torch.FloatTensor([label.item()])
        return img, label

    def __len__(self):
        """Return size of dataset."""
        return self.dataset_size

    def _check_exists(self):
        """Check if dataset is download and in right place."""
        return os.path.exists(os.path.join(self.root, self.filename))

    def download(self):
        """Download dataset."""
        filename = os.path.join(self.root, self.filename)
        dirname = os.path.dirname(filename)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        if os.path.isfile(filename):
            return
        print("Download %s to %s" % (self.url, os.path.abspath(filename)))
        urllib.request.urlretrieve(self.url, filename)
        print("[DONE]")
        return

    def load_samples(self):
        """Load sample images from dataset."""
        filename = os.path.join(self.root, self.filename)
        f = gzip.open(filename, "rb")
        data_set = pickle.load(f, encoding="bytes")
        f.close()
        if self.train:
            images = data_set[0][0]
            labels = data_set[0][1]
            self.dataset_size = labels.shape[0]
        else:
            images = data_set[1][0]
            labels = data_set[1][1]
            self.dataset_size = labels.shape[0]
        return images, labels


class OFFICE(data.Dataset):
    """OFFICE 31 Dataset.
    Args:
        root (string): Root directory of dataset where dataset file exist.
        train (bool, optional): If True, resample from dataset randomly.
        domain (str, optional): amazon, dslr, webcam
        transform (callable, optional): A function/transform that takes in
            an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
    """

    url = ""

    def __init__(self, root, train=True, transform=None, domain="amazon"):
        """Init USPS dataset."""
        # init params
        self.root = os.path.expanduser(root)
        self.filename = "usps_28x28.pkl"
        self.train = train
        # Num of Train = 7438, Num ot Test 1860
        self.transform = transform
        self.dataset_size = None
        self.domain = domain

        if not self._check_exists():
            raise RuntimeError("Dataset not found." +
                               " You can use download=True to download it")

        self.train_data, self.train_labels = self.load_samples()
        if self.train:
            total_num_samples = self.train_labels.shape[0]
            indices = np.arange(total_num_samples)
            np.random.shuffle(indices)
            self.train_data = self.train_data[indices[0:self.dataset_size], ::]
            self.train_labels = self.train_labels[indices[0:self.dataset_size]]
        self.train_data *= 255.0
        self.train_data = self.train_data.transpose(
            (0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, label = self.train_data[index, ::], self.train_labels[index]
        if self.transform is not None:
            img = self.transform(img)
        label = torch.LongTensor([np.int64(label).item()])
        # label = torch.FloatTensor([label.item()])
        return img, label

    def __len__(self):
        """Return size of dataset."""
        return self.dataset_size

    def _check_exists(self):
        """Check if dataset is download and in right place."""
        return os.path.exists(os.path.join(self.root, self.filename))

    def load_samples(self):
        """Load sample images from dataset."""
        filename = os.path.join(self.root, self.filename)
        f = gzip.open(filename, "rb")
        data_set = pickle.load(f, encoding="bytes")
        f.close()
        if self.train:
            images = data_set[0][0]
            labels = data_set[0][1]
            self.dataset_size = labels.shape[0]
        else:
            images = data_set[1][0]
            labels = data_set[1][1]
            self.dataset_size = labels.shape[0]
        return images, labels


def get_loader(config):
    svhn_transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    mnist_transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.RandomCrop(config.image_size, pad_if_needed=True),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomRotation(40),
        transforms.ColorJitter(brightness=30, contrast=30, saturation=30, hue=0.5),
        transforms.ToTensor(),
        # transforms.RandomApply([transforms.Lambda(lambda x: x.mul(-1)), transforms.Lambda(lambda x: x.mul(1))]),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    usps_transfor = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(config.image_size),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    svhn = datasets.SVHN(root=config.svhn_path, download=True, transform=svhn_transform)
    svhn_val = datasets.SVHN(root=config.svhn_path, download=True, split='test', transform=svhn_transform)
    svhn_extra = datasets.SVHN(root=config.svhn_path, download=True, split='extra', transform=svhn_transform)

    mnist = datasets.MNIST(root=config.mnist_path, download=True, transform=mnist_transform)
    mnist_val = datasets.MNIST(root=config.mnist_path, train=False, download=True, transform=mnist_transform)

    usps = USPS(root=config.mnist_path, train=True, transform=usps_transfor, download=True)
    usps_val = USPS(root=config.mnist_path, train=False, transform=usps_transfor, download=True)

    svhn_loader = torch.utils.data.DataLoader(dataset=svhn,
                                              batch_size=config.batch_size,
                                              shuffle=True,
                                              num_workers=config.num_workers)
    svhn_val_loader = torch.utils.data.DataLoader(dataset=svhn_val,
                                                  batch_size=500,
                                                  shuffle=False,
                                                  num_workers=config.num_workers)
    svhn_extra_loader = torch.utils.data.DataLoader(dataset=svhn_extra,
                                                    batch_size=config.batch_size,
                                                    shuffle=True,
                                                    num_workers=config.num_workers)

    mnist_loader = torch.utils.data.DataLoader(dataset=mnist,
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               num_workers=config.num_workers)
    mnist_val_loader = torch.utils.data.DataLoader(dataset=mnist_val,
                                                   batch_size=500,
                                                   shuffle=False,
                                                   num_workers=config.num_workers)

    usps_loader = torch.utils.data.DataLoader(dataset=usps,
                                              batch_size=config.batch_size,
                                              shuffle=False,
                                              num_workers=config.num_workers)
    usps_val_loader = torch.utils.data.DataLoader(dataset=usps_val,
                                                  batch_size=500,
                                                  shuffle=False,
                                                  num_workers=config.num_workers)

    if config.source == "mnist":
        if config.target  == "svhn":
            return mnist_loader, svhn_loader, svhn_val_loader
        elif config.target  == "svhn_extra":
            return mnist_loader, svhn_extra_loader, svhn_val_loader
        elif config.target  == "usps":
            return mnist_loader, usps_loader, usps_val_loader
        else:
            raise RuntimeError("Not yet implemented")
    elif config.source == "svhn":
        if config.target  == "mnist":
            return svhn_loader, mnist_loader, mnist_val_loader
        elif config.target  == "usps":
            return svhn_loader, usps_loader, usps_val_loader
    elif config.source == "svhn_extra":
        if config.target  == "mnist":
            return svhn_extra_loader, mnist_loader, mnist_val_loader
    elif config.source == "usps":
        return usps_loader, mnist_loader, mnist_val_loader
