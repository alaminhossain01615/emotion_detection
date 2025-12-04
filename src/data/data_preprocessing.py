import torch
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random


class DataPreprocessing:

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.mean=0
        self.std=0

    def calculate_mean_and_std(self,DATASET_DIR=None,grayscale=1):
        count=0
        if DATASET_DIR is None:
            DATASET_DIR=self.root_dir

        transforms = [v2.ToImage()]
        if grayscale == 1:
            transforms.append(v2.Grayscale(num_output_channels=1))
        transforms.append(v2.ToDtype(torch.float32, scale=True))

        initial_transformation = v2.Compose(transforms)

        dataset=ImageFolder(DATASET_DIR,transform=initial_transformation)
        loader=DataLoader(dataset,batch_size=512,shuffle=False)

        for images,_ in loader:
            batch_size=images.size(0)
            self.mean = self.mean+images.mean(dim=[0,2,3])*batch_size
            self.std = self.std+images.std(dim=[0,2,3])*batch_size
            count=count+batch_size

        self.mean = self.mean/count
        self.std=self.std/count
        return self.mean,self.std

    def augment_and_transform_data(self,DATASET_DIR=None,grayscale=1, list_of_transform_pipeline=[],mean=None,std=None, plot_original_vs_augmented = 1,num_examples=5):
        if DATASET_DIR is None:
            DATASET_DIR=self.root_dir
        if mean is None:
            mean=self.mean
        if std is None:
            std=self.std
        if len(list_of_transform_pipeline)==0:
            transforms = [v2.ToImage()]
            if grayscale == 1:
                transforms.append(v2.Grayscale(num_output_channels=1))
            transforms.append(v2.RandomHorizontalFlip(p=0.2))
            transforms.append(v2.RandomRotation(20))
            transforms.append(v2.ToDtype(torch.float32, scale=True))
            transforms.append(v2.Normalize(mean=mean, std=std))
        else:
            transforms=list_of_transform_pipeline

        transformations=v2.Compose(transforms)

        if plot_original_vs_augmented == 1:
            plt.figure(figsize=(10, 4))
            transforms = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])    
            dataset=ImageFolder(DATASET_DIR,transform=transforms)
            indices = random.sample(range(len(dataset)), num_examples)
            for i, idx in enumerate(indices):
                original_img, label = dataset[idx]
                augmented_img = transformations(original_img)

                plt.subplot(2, num_examples, i + 1)
                self.show_image(original_img, title=f"Original {label}",unnormalize=False)

                plt.subplot(2, num_examples, i + 1 + num_examples)
                self.show_image(augmented_img, title=f"Augmented {label}")

            plt.tight_layout()
            plt.show()

        return transformations

    def show_image(self, tensor_img, title=None, unnormalize=True,mean=None,std=None):
        if mean is None:
            mean=self.mean
        if std is None:
            std=self.std
        img = tensor_img.clone()
        print(tensor_img.size())
        if unnormalize:
            img=img * std[:, None, None] + mean[:, None, None]
            img = img.clamp(0, 1)
        img = img.permute(1, 2, 0).numpy()
        plt.imshow(img, cmap='gray' if img.shape[2]==1 else None)
        if title:
            plt.title(title)
        plt.axis('off')

    # def plot_original_vs_augmented_examples(self,num_examples=5,DATASET_DIR=None,list_of_transform_pipeline=[]):
    #     plt.figure(figsize=(10, 4))
    #     if DATASET_DIR is None:
    #         DATASET_DIR=self.root_dir
    #     transforms = [v2.ToImage()]
        
    #     dataset=ImageFolder(DATASET_DIR,transform=None)
    #     transformations=self.augment_and_transform_data()
    #     indices = random.sample(range(len(dataset)), num_examples)
    #     for i, idx in enumerate(indices):
    #         original_img, label = dataset[idx]
    #         augmented_img = train_transform(original_img)

    #         plt.subplot(2, num_examples, i + 1)
    #         self.show_image(original_img, title=f"Original {label}",unnormalize=False)

    #         plt.subplot(2, num_examples, i + 1 + num_examples)
    #         self.show_image(augmented_img, title=f"Augmented {label}")

    #     plt.tight_layout()
    #     plt.show()