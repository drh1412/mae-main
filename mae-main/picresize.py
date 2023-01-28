import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# img = Image.open("/media/drh/DATA/calcium/mae-main/dataoftest/train/1/225654.png")
# print(img.size)
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
def getStat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())


transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize(a)
])



if __name__ == '__main__':
    dataset_train =ImageFolder('/media/drh/DATA/calcium/mae-main/dataoftest/train',transform=transform_train)
    print(getStat(dataset_train))
# img=dataset_train.__getitem__(0)
# print(img[0].max())
#
# plt.imshow(img[0].permute(1,2,0))
# plt.show()

# img = img.resize((256, 256))
#
# img = np.array(img)
#
# img=np.expand_dims(img,2).repeat(3,axis=2)
# img = cv2.resize(img,dsize=(224,224))
# cv2.imwrite("D:\MAE\mae-main\mae-main\demo\data2.png",img)