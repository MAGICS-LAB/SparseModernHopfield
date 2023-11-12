import numpy as np 
import matplotlib.pyplot as plt 
import torch
import torchvision 
from torchvision import transforms
from copy import deepcopy
import torch.nn.functional as F
import time
import scipy.ndimage as nd
import imageio

def load_mnist(batch_size):
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root='./mnist_data', train=True,
                                            download=True, transform=transform)
    print("trainset: ", trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True)
    print("trainloader: ", trainloader)
    trainset = list(iter(trainloader))

    testset = torchvision.datasets.MNIST(root='./mnist_data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=True)
    testset = list(iter(testloader))
    return trainset, testset

def get_cifar10(batch_size):
    transform = transforms.Compose([transforms.ToTensor()])#, transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./cifar_data', train=True,
                                            download=True,transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                      shuffle=True)
    train_data = list(iter(trainloader))
    testset = torchvision.datasets.CIFAR10(root='./cifar_data', train=False,
                                            download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                      shuffle=True)
    test_data = list(iter(testloader))
    return train_data, test_data

def get_id_dictionary(path):
    id_dict = {}
    for i, line in enumerate(open( path + 'wnids.txt', 'r')):
        id_dict[line.replace('\n', '')] = i
    return id_dict
  
def get_class_to_id_dict(path):
    id_dict = get_id_dictionary()
    all_classes = {}
    result = {}
    for i, line in enumerate(open( path + 'words.txt', 'r')):
        n_id, word = line.split('\t')[:2]
        all_classes[n_id] = word
    for key, value in id_dict.items():
        result[value] = (key, all_classes[key])      
    return result

def get_data(path,id_dict):
    print('starting loading data')
    train_data, test_data = [], []
    train_labels, test_labels = [], []
    t = time.time()
    for key, value in id_dict.items():
        train_data += [plt.imread( path + 'train/{}/images/{}_{}.JPEG'.format(key, key, str(i)), format='RGB') for i in range(500)]
        train_labels_ = np.array([[0]*200]*500)
        train_labels_[:, value] = 1
        train_labels += train_labels_.tolist()

    for line in open( path + 'val/val_annotations.txt'):
        img_name, class_id = line.split('\t')[:2]
        test_data.append(plt.imread( path + 'val/images/{}'.format(img_name) ,format='RGB'))
        test_labels_ = np.array([[0]*200])
        test_labels_[0, id_dict[class_id]] = 1
        test_labels += test_labels_.tolist()

    print('finished loading data, in {} seconds'.format(time.time() - t))
    return train_data, train_labels, test_data, test_labels

def parse_train_data(train_images, train_labels,N_imgs):
    images = torch.zeros((N_imgs, 64,64,3))
    for i,(img,label) in enumerate(zip(train_images, train_labels)):
        if i >= N_imgs:
            break
        print(i)
        if len(img.shape) == 3:
            images[i,:,:,:] = torch.tensor(img,dtype=torch.float) / 255.0 # normalize
    return torch.tensor(images)

def load_tiny_imagenet(N_imgs):
    path = 'IMagenet/tiny-imagenet-200/'
    train_data, train_labels, test_data, test_labels = get_data(path,get_id_dictionary(path))
    images = parse_train_data(train_data, train_labels,N_imgs)
    return images

def load_synthetic(N_imgs, signal=10):
    image_width = 10
    image_length = 10
    image_list = []

    for i in range(N_imgs):
        indices = np.random.choice(100, signal, replace=False)  # Generate 10 random indices from 0 to 99
        rows, cols = np.unravel_index(indices, (10, 10))  # Convert indices to 2D coordinates
        img_array = np.random.normal(0, 0., (image_width, image_length))

        for i in range(signal):
            img_array[rows[i], cols[i]] = np.random.choice([-1, 1])
        image_list.append(np.expand_dims(img_array, axis=-1))
    return torch.clamp(((torch.tensor(image_list) + 1) / 2), 0, 1)