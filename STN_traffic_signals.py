# -*- coding: utf-8 -*-

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import csv
from PIL import Image

plt.ion()   # interactive mode

######################################################################
# Loading the data
# ----------------
#
# In this post we experiment with the classic MNIST dataset. Using a
# standard convolutional network augmented with a spatial transformer
# network.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path_train = './data/train/GTSRB/Final_Training/Images'
path_test = "./data/test/GTSRB/Final_Test/Images"




class TrafficSignals(Dataset):
    """Dataset wrapping images and target labels for Kaggle - Planet Amazon from Space competition.

    Arguments:
        A CSV file path
        Path to image folder
        Extension of images
        PIL transforms
    """

    def __init__(self, path_train, path_test, train=False, transform=None):

        self.path_train = path_train
        self.path_test = path_test
        self.transform = transform
        self.train = train

        if train:
            [self.X_train, self.y_train] = self.readTrafficSigns(path_train, train=True)

        else:
            [self.X_train, self.y_train] = self.readTrafficSigns(path_test, train=False)

        # print(self.y_train)
        # print(self.X_train)

    def __getitem__(self, index):
        img = Image.open(self.X_train[index]).resize((28,28), resample=0)
        if self.transform is not None:
             img = self.transform(img)
        label = int(self.y_train[index])

        return img, label

    def __len__(self):
        return len(self.X_train)

    def readTrafficSigns(self, rootpath, train=False):
        '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

        Arguments: path to the traffic sign data, for example './GTSRB/Training'
        Returns:   list of images, list of corresponding labels'''
        images = []  # images
        labels = []  # corresponding labels

        if train:

        # loop over all 42 classes
            for c in range(0, 43):
                # prefix = rootpath + '/'  # subdirectory for class
                prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
                gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
                gtReader = csv.reader(gtFile, delimiter=';')  # csv parser for annotations file
                next(gtReader)  # skip header
                # loop over all images in current annotations file
                for row in gtReader:
                    images.append(prefix + row[0])  # the 1th column is the filename
                    # images.append(plt.imread(prefix + row[0]))  # the 1th column is the filename
                    labels.append(row[7])  # the 8th column is the label
                gtFile.close()

        else:
            prefix = rootpath + '/' # subdirectory for class
            gtFile = open(prefix + 'GT-final_test.csv') # annotations file
            gtReader = csv.reader(gtFile, delimiter=';')  # csv parser for annotations file
            next(gtReader)  # skip header
            # loop over all images in current annotations file
            for row in gtReader:
                images.append(prefix + row[0])  # the 1th column is the filename
                # images.append(plt.imread(prefix + row[0]))  # the 1th column is the filename
                labels.append(row[7])  # the 8th column is the label
            gtFile.close()
        return images, labels

#train dataset loader
dset_train = TrafficSignals(path_train, path_test, train=True, transform=transforms.Compose([
                       transforms.Grayscale(),
                       transforms.ToTensor(),
                   ]))
print(dset_train.__getitem__(1))
train_loader = torch.utils.data.DataLoader(dset_train,
                          batch_size=64,
                          shuffle=True,
                          num_workers=4
                         # pin_memory=True # CUDA only
                         )

#test dataset loader
dset_test = TrafficSignals(path_train, path_test, transform=transforms.Compose([
                       transforms.Grayscale(),
                       transforms.ToTensor(),
                   ]))
test_loader = torch.utils.data.DataLoader(dset_test,
                          batch_size=64,
                          shuffle=True,
                          num_workers=4
                         # pin_memory=True # CUDA only
                         )


print('train_loader enumerate: ',enumerate(train_loader))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        # xs = xs.view(xs.size(0), -1)
        xs = xs.view(-1, 90)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model = Net().to(device)

######################################################################
# Training the model
# ------------------
#
# Now, let's use the SGD algorithm to train the model. The network is
# learning the classification task in a supervised way. In the same time
# the model is learning STN automatically in an end-to-end fashion.


optimizer = optim.SGD(model.parameters(), lr=0.01)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        print('object:', data)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

    # for batch_idx, (img, label) in enumerate(train_loader):
    #     print('object:', img)
    #     data, target = img.to(device), label.to(device)
    #     optimizer.zero_grad()
    #     output = model(data)

        loss = F.nll_loss(output, label)
        loss.backward()
        optimizer.step()
        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
#
# A simple test procedure to measure STN the performances on MNIST.
#


def test():
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # sum up batch loss
            test_loss += F.nll_loss(output, target, size_average=False).item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
              .format(test_loss, correct, len(test_loader.dataset),
                      100. * correct / len(test_loader.dataset)))

######################################################################
# Visualizing the STN results
# ---------------------------
#
# Now, we will inspect the results of our learned visual attention
# mechanism.
#
# We define a small helper function in order to visualize the
# transformations while training.


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

# We want to visualize the output of the spatial transformers layer
# after the training, we visualize a batch of input images and
# the corresponding transformed batch using STN.


def visualize_stn():
    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(test_loader))[0].to(device)

        input_tensor = data.cpu()
        transformed_input_tensor = model.stn(data).cpu()

        in_grid = convert_image_np(
            torchvision.utils.make_grid(input_tensor))

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor))

        scipy.misc.imsave('in_grid.png', in_grid)
        scipy.misc.imsave('out_grid.png', out_grid)
        torchvision.utils.save_image(input_tensor, "in_grid_img.png", nrow=8, padding=2, normalize=False, range=None, scale_each=False,
                                     pad_value=0)

        torchvision.utils.save_image(transformed_input_tensor, "out_grid_img.png", nrow=8, padding=2, normalize=False, range=None,
                                     scale_each=False,
                                     pad_value=0)


        # # Plot the results side-by-side
        # f, axarr = plt.subplots(1, 2)
        # axarr[0].imshow(in_grid)
        # axarr[0].set_title('Dataset Images')
        #
        # axarr[1].imshow(out_grid)
        # axarr[1].set_title('Transformed Images')


for epoch in range(1, 20 + 1):
    train(epoch)
    test()


# Visualize the STN transformation on some input batch
visualize_stn()

# plt.ioff()
# plt.show()


## ___________LOADER 2__________________
# def readTrafficSigns(rootpath, train_set=False):
#     '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.
#
#     Arguments: path to the traffic sign data, for example './GTSRB/Training'
#     Returns:   list of images, list of corresponding labels'''
#     images = [] # images
#     labels = [] # corresponding labels
#     # loop over all 42 classes
#     if train_set:
#
#         for c in range(0,43):
#             prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
#             gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
#             gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
#             next(gtReader) # skip header
#             print(gtReader)
#         # loop over all images in current annotations file
#             for row in gtReader:
#                 images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
#                 labels.append(row[7]) # the 8th column is the label
#             gtFile.close()
#
#     else:
#         prefix = rootpath + '/' # subdirectory for class
#         gtFile = open(prefix + 'GT-final_test.csv') # annotations file
#         gtReader = csv.reader(gtFile, delimiter=';')  # csv parser for annotations file
#         next(gtReader) # skip header
#         # loop over all images in current annotations file
#         for row in gtReader:
#             images.append(plt.imread(prefix + row[0]))  # the 1th column is the filename
#             labels.append(row[7])  # the 8th column is the label
#         gtFile.close()
#         return images, labels
#
# class TrafficSignsDataset(Dataset):
#     def __init__(self, images, labels, transform=None):
#         self.images = torch.from_numpy(images)
#         self.images = self.images.permute(0, 3, 1, 2)
#         self.labels = torch.LongTensor(labels.argmax(1))
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.images)
#
#     def __getitem__(self, index):
#         return self.images[index], self.labels[index]
#
# ##
#
#
# X_train_np, y_train_np = readTrafficSigns(path_train, train_set=True)
# [X_test_np, y_test_np] = readTrafficSigns(path_test)
#
# #train dataset loader
# dset_train = TrafficSignsDataset(X_train_np, y_train_np, transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ]))
# train_loader = DataLoader(dset_train,
#                           batch_size=64,
#                           shuffle=True,
#                           num_workers=4
#                          # pin_memory=True # CUDA only
#                          )
# #test dataset loader
# dset_test = TrafficSignsDataset(X_test_np, y_test_np, transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ]))
# test_loader = DataLoader(dset_test,
#                           batch_size=64,
#                           shuffle=True,
#                           num_workers=4
#                          # pin_memory=True # CUDA only
#                          )
