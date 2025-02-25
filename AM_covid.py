from __future__ import print_function
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torchvision.transforms as T
import PIL.Image as Image


class VGG16(nn.Module):
    def __init__(self, num_classes=3):
        super(VGG16, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 7 * 7, 4096),  
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)
        out5 = self.block5(out4)
        out = out5.view(out5.size(0), -1)
        out = self.classifier(out)
        # return out, [out1, out2, out3, out4, out5]
        return out
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.15244901180267334, 0.1534290313720703, 0.15778332948684692],
    #                      std=[0.15354399383068085, 0.1540035903453827, 0.155692458152771])  #  标准化
])

# 加载数据集
dataset_path = "D:/VGG_CIFAR10/image_dataset"
train_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
n = 302# 第几张图片
target_class = 0#图片种类                              #covid类型第300个和论文中的类似250个b线明显
count = 0
target_image = None
for image, label in train_dataset:
    if label == target_class:
        count += 1
        if count == n:
            target_image = image.unsqueeze(0)
            break

input_image = target_image.clone().detach().requires_grad_(True)
print(input_image.size())

original_image = input_image.detach().squeeze().permute(1, 2, 0).numpy()
original_image = (original_image).clip(0, 1)
plt.figure(figsize=(10, 10))
plt.imshow(original_image)
plt.title(f"Original Image for Class {target_class} ")
plt.axis('off')
plt.show()

model = VGG16()
model.load_state_dict(torch.load('vgg16_covid19.pth'))
model.eval()
optimizer = torch.optim.Adam([input_image], lr=0.1)

epsilon_1 = 0.1
epsilon_2 = 0.1

def total_variation(image):
    diff1 = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]).mean()
    diff2 = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]).mean()
    return diff1 + diff2

for step in range(500):
    optimizer.zero_grad()
    output = model(input_image)

    if isinstance(output, tuple):
        output = output[0]
    target_activation = output[0, target_class]
    activation_gradient = torch.autograd.grad(target_activation, input_image)[0]    # 计算激活值的梯度
    tv_gradient = torch.autograd.grad(total_variation(input_image), input_image)[0]    # 计算总变差的梯度
    input_image.data = input_image.data + epsilon_1 * activation_gradient - epsilon_2 * tv_gradient
    input_image.data.clamp_(0, 1)
    loss = -target_activation + epsilon_2 / epsilon_1 * total_variation(input_image)
    if step % 20 == 0:
        print(f"Step {step}, Loss: {loss.item()}")

optimized_image = input_image.detach().squeeze().permute(1, 2, 0).numpy()
optimized_image = (optimized_image * 0.5 + 0.5).clip(0, 1)
plt.imshow(optimized_image)
plt.title(f"Optimized Image for Class {target_class} ")
plt.axis('off')
plt.show()