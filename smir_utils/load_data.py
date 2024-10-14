
import numpy as np
import torch
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
# import cv2
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset, Subset
import os
import pickle
from tqdm import tqdm

def load_data(dataset, path_to_data, batch_size=1, To_Tensor=False, return_dataset=False,
              return_dataloader=False, show_imgs=False, save_imgs=False):
    print('\nload_data')
    if dataset == 'fashion_mnist':
        trainset = torchvision.datasets.FashionMNIST(root=os.path.join(path_to_data, "fashion_mnist/train"), train=True,
                                                     download=True, transform=transforms.ToTensor())
        testset = torchvision.datasets.FashionMNIST(root=os.path.join(path_to_data, "fashion_mnist/test"), train=False,
                                                    download=True, transform=transforms.ToTensor())
    elif dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root=os.path.join(path_to_data, "mnist/train"), train=True,
                                                     download=True, transform=transforms.ToTensor())
        testset = torchvision.datasets.MNIST(root=os.path.join(path_to_data, "mnist/test"), train=False,
                                                    download=True, transform=transforms.ToTensor())
    elif dataset == 'cifar_10':
        trainset = torchvision.datasets.CIFAR10(root=os.path.join(path_to_data, "cifar_10/train"), train=True,
                                                download=True, transform=transforms.ToTensor())
        testset = torchvision.datasets.CIFAR10(root=os.path.join(path_to_data, "cifar_10/test"), train=False,
                                               download=True, transform=transforms.ToTensor())
    elif dataset == 'cifar_100':
        trainset = torchvision.datasets.CIFAR100(root=os.path.join(path_to_data, "cifar_100/train"), train=True,
                                                download=True, transform=transforms.ToTensor())
        testset = torchvision.datasets.CIFAR100(root=os.path.join(path_to_data, "cifar_100/test"), train=False,
                                               download=True, transform=transforms.ToTensor())
    elif dataset == 'svhn':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),  # Commonly used standardized parameters
        ])

        trainset = torchvision.datasets.SVHN(root=os.path.join(path_to_data, "svhn/train"), split='train',
                                             download=True, transform=transform)
        testset = torchvision.datasets.SVHN(root=os.path.join(path_to_data, "svhn/test"), split='test',
                                            download=True, transform=transform)
        trainset.class_to_idx = {str(i): i for i in range(10)}
        testset.class_to_idx = {str(i): i for i in range(10)}
    elif dataset == 'NEU-CLS-64':
        # preprocessing
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
        ])
        dataset_all = ImageFolder(root=os.path.join(path_to_data, "neu-cls-64"), transform=transform)

        class_indices = {cls: [] for cls in range(len(dataset_all.classes))}
        for idx, (path, label) in enumerate(dataset_all.imgs):
            class_indices[label].append(idx)
        split_ratio_training = 0.7
        train_indices, test_indices = split_indices_by_ratio(class_indices, split_ratio_training)

        trainset = Subset(dataset_all, train_indices)
        testset = Subset(dataset_all, test_indices)
        trainset.class_to_idx = {'cr': 0, 'in': 1, 'pa': 2, 'ps': 3, 'rs': 4, 'sc': 5}
        trainset.class_with_decre_severity = ['in', 'ps', 'cr', 'rs', 'sc', 'pa']

    elif dataset == 'APTOS2019':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # The file path can be modified by yourself
        csv_file = '/root/autodl-tmp/Projects/lhr_data/aptos2019/train.csv'
        orig_img_dir = '/root/autodl-tmp/Projects/lhr_data/aptos2019/train_images'
        processed_img_dir = '/root/autodl-tmp/Projects/lhr_data/aptos2019/processed_images_224^2'

        dataset_all = APTOSDataset(csv_file=csv_file, orig_img_dir=orig_img_dir,
                               processed_img_dir=processed_img_dir, transform=transform)
        show_imgs = False
        if show_imgs:
            dataset_all.show_orig_imgs()
            dataset_all.show_processed_imgs()
        class_indices = {cls: [] for cls in range(len(dataset_all.classes))}
        for idx in range(len(dataset_all)):
            _, label = dataset_all[idx]
            class_indices[label].append(idx)

        split_ratio_training = 0.8
        train_indices = []
        test_indices = []
        for class_label, indices in class_indices.items():
            np.random.seed(42)
            np.random.shuffle(indices)
            split = int(split_ratio_training * len(indices))
            train_indices.extend(indices[:split])
            test_indices.extend(indices[split:])

        trainset = Subset(dataset_all, train_indices)
        testset = Subset(dataset_all, test_indices)
        trainset.class_to_idx = {'No DR': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3, 'Proliferative DR': 4}
        trainset.class_with_decre_severity = ['Proliferative DR', 'Severe', 'Moderate', 'Mild', 'No DR']

    if return_dataset == True:
        print('The total training set is: %d' % len(trainset))
        print('The total testing set is: %d' % len(testset))
        return (trainset, testset)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    if dataset == 'APTOS2019':
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=4)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    # # save_imgs = True
    # if save_imgs:
    #     class_to_idx = trainset.class_to_idx
    #     train_save = os.path.join(path_to_data, f"{dataset}/images/train")
    #     test_save = os.path.join(path_to_data, f"{dataset}/images/test")
    #     save_images_by_class(trainloader, class_to_idx, train_save)
    #     save_images_by_class(testloader, class_to_idx, test_save)
    #     return None

    if return_dataloader == True:
        return (trainloader, testloader)

    if To_Tensor == False:
        train_data = [[], []]
        for data in trainloader:
            images, labels = data
            if dataset in ['mnist', 'fashion_mnist']:
                image_temp = images.numpy()[0].reshape(1, -1)
                train_data[0].append(image_temp)

            elif dataset in ['cifar_10', 'svhn']:
                image_temp = images.numpy()[0]
                train_data[0].append(image_temp)
            elif dataset == 'NEU-CLS-64':
                image_temp = images.numpy()[0].transpose(1, 2, 0)
                train_data[0].append(image_temp)
            elif dataset == 'APTOS2019':
                image_temp = images.numpy()[0].transpose(1, 2, 0)
                train_data[0].append(image_temp)

            train_data[1].append(labels.item())

        train_data[0] = np.asarray(train_data[0])
        train_data[1] = np.asarray(train_data[1])

        test_data = [[], []]
        for data in testloader:
            images, labels = data
            if dataset in ['mnist', 'fashion_mnist']:
                test_data[0].append(images.numpy()[0].reshape(1, -1))
            elif dataset in ['cifar_10', 'svhn']:
                test_data[0].append(images.numpy()[0])
            elif dataset == 'NEU-CLS-64':
                image_temp = images.numpy()[0].transpose(1, 2, 0)
                test_data[0].append(image_temp)
            elif dataset == 'APTOS2019':
                image_temp = images.numpy()[0].transpose(1, 2, 0)
                test_data[0].append(image_temp)
            test_data[1].append(labels.item())

        test_data[0] = np.asarray(test_data[0])
        test_data[1] = np.asarray(test_data[1])

        return (train_data, test_data)

    elif To_Tensor == True:
        train_data = [[], []]
        for data in trainloader:
            images, labels = data
            train_data[0].append(images)
            train_data[1].append(labels)

        test_data = [[], []]
        for data in testloader:
            images, labels = data
            test_data[0].append(images)
            test_data[1].append(labels)
        return (train_data, test_data)


class ToFloat32:
    def __call__(self, tensor):
        return tensor.to(torch.float32)

class APTOSDataset(Dataset):
    '''
    APTOSDataset
    '''
    def __init__(self, csv_file, orig_img_dir, processed_img_dir, transform=None):
        self.labels = pd.read_csv(csv_file)
        self.orig_img_dir = orig_img_dir
        self.processed_img_dir = processed_img_dir
        self.transform = transform

        np.random.seed(42)
        self.selected_indices = np.random.choice(len(self), 25, replace=False)
        self.classes = sorted(self.labels['diagnosis'].unique().tolist())
        self.class_to_idx = {'No DR': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3, 'Proliferative DR': 4}
        self.class_with_decre_severity = ['Proliferative DR', 'Severe', 'Moderate', 'Mild', 'No DR']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx, preprocess=True, transformed=True):
        import time
        # time1 = time.time();
        if not os.path.exists(self.processed_img_dir):
            os.makedirs(self.processed_img_dir)
        processed_img_name = os.path.join(self.processed_img_dir, self.labels.iloc[idx, 0] + '.png')
        if not os.path.exists(processed_img_name):
            orig_img_name = os.path.join(self.orig_img_dir, self.labels.iloc[idx, 0] + '.png')
            image = Image.open(orig_img_name)
            image = APTOS_circle_crop(image, sigmaX=10)
            image = Image.fromarray(np.array(image))
            image.save(processed_img_name)
        else:
            image = Image.open(processed_img_name)

        if self.transform and transformed:
            image = self.transform(image)
        img_id = self.labels.iloc[idx, 0]
        label = self.labels.loc[self.labels['id_code'] == img_id, 'diagnosis'].values[0]
        # time4 = time.time();print(f'\ntime3:{time4-time3}')
        return image, label

    def preprocess_images(self):
        if not os.path.exists(self.processed_img_dir):
            os.makedirs(self.processed_img_dir)

        for idx in tqdm(range(len(self.labels))):
            img_name = os.path.join(self.processed_img_dir, self.labels.iloc[idx, 0] + '.png')
            if not os.path.exists(img_name):
                orig_img_name = os.path.join(self.orig_img_dir, self.labels.iloc[idx, 0] + '.png')
                image = Image.open(orig_img_name)
                image = APTOS_circle_crop(image, sigmaX=10)
                image = Image.fromarray(image)
                image = image.resize((224, 224), Image.LANCZOS)
                image = Image.fromarray(np.array(image))
                image.save(img_name)

    def show_processed_imgs(self):
        fig = plt.figure(figsize=(30, 30))
        for i, idx in enumerate(self.selected_indices):
            ax = fig.add_subplot(5, 5, i + 1, xticks=[], yticks=[])
            image, label = self.__getitem__(idx, preprocess=True, transformed=False)
            w, h = image.size
            ax.imshow(image)
            ax.set_title(f'Severity: {label}, shape: {w} * {h}')
        plt.show()

    def show_orig_imgs(self):
        fig = plt.figure(figsize=(30, 30))
        for i, idx in enumerate(self.selected_indices):
            img_name = os.path.join(self.orig_img_dir, self.labels.iloc[idx, 0] + '.png')
            ax = fig.add_subplot(5, 5, i + 1, xticks=[], yticks=[])
            image = Image.open(img_name)
            w, h = image.size
            ax.imshow(image)
            lab = self.labels.loc[self.labels['id_code'] == self.labels.iloc[idx, 0], 'diagnosis'].values[0]
            ax.set_title('Severity: %s, shape: %s * %s' % (lab, w, h))
        plt.show()

def APTOS_crop_image_from_gray(img, tol=7):
    img = np.array(img)
    if img.ndim == 2:
        mask = img > tol
        img = img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = np.array(Image.fromarray(img).convert('L'))
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)

    h, w = img.shape[:2]
    if h > w:
        start = (h - w) // 2
        img = img[start:start + w, :]
    elif w > h:
        start = (w - h) // 2
        img = img[:, start:start + h]

    return img

def APTOS_circle_crop(img, sigmaX=10):

    img = np.array(img)
    img = APTOS_crop_image_from_gray(img, tol=7)
    img = Image.fromarray(img)

    width, height = img.size
    x = int(width / 2)
    y = int(height / 2)
    r = np.amin([x, y])

    circle_img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(circle_img)
    draw.ellipse((x - r, y - r, x + r, y + r), fill=1)

    img = np.array(img)
    circle_img = np.array(circle_img)
    img = np.multiply(img, np.stack([circle_img]*3, axis=-1))

    img = Image.fromarray(img)
    img = img.filter(ImageFilter.GaussianBlur(sigmaX))

    img = Image.blend(img, img.filter(ImageFilter.GaussianBlur(sigmaX)), -4)

    img = np.uint8(np.clip(np.array(img), 0, 255))

    return img

def split_indices_by_ratio(class_indices, split_ratio_training):
    train_indices = []
    test_indices = []
    for cls, indices in class_indices.items():
        split_point = int(split_ratio_training * len(indices))
        train_indices.extend(indices[:split_point])
        test_indices.extend(indices[split_point:])

    return train_indices, test_indices

def show_samples(dataset, num_samples=10):
    fig = plt.figure(figsize=(10, 10))
    for i in range(num_samples):
        ax = fig.add_subplot(1, num_samples, i + 1, xticks=[], yticks=[])
        img, label = dataset[i]

        img = img / 2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.axis('off')
        ax.set_title(str(label))
    plt.show()

def compute_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images_count = 0

    for data in loader:
        images, _ = data
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count
    mean = tuple(mean.numpy())
    std = tuple(std.numpy())
    print(f'mean and std:{(mean, std)}')
    return mean, std

def save_images_by_class(dataloader, class_to_idx, output_dir):
    from torchvision.utils import save_image

    for class_name in class_to_idx.keys():
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)


    for batch_idx, (images, labels) in enumerate(dataloader):
        for i in range(images.size(0)):
            image = images[i]
            label = labels[i].item()
            class_name = list(class_to_idx.keys())[list(class_to_idx.values()).index(label)]
            # 保存图像
            image_filename = os.path.join(output_dir, class_name, f"img_{batch_idx * len(images) + i}.png")
            save_image(image, image_filename)