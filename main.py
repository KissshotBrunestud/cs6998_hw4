# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import pandas as pd

cudnn.benchmark = True
plt.ion()   # interactive mode


# Transformations for the training set and the validation set
BATCHSIZE = 128
DATA_DIR = '~/data/cifar10'

data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]),
}

# Load the data
def load_data(num_workers=2):
    train_set = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True,
                                            download=True, transform=data_transforms['train'])

    val_set = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False,
                                        download=True, transform=data_transforms['val'])

    image_datasets = {'train': train_set, 'val': val_set}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCHSIZE,
                                                shuffle=True, num_workers=num_workers)
                for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes


    print('Dataset sizes:{}'.format(dataset_sizes))
    print('Class names:{}'.format(class_names))
    return dataloaders, dataset_sizes, class_names


def train_model(model, device, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=350, acc=0.92):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    history = []
    training_iter = 0
    
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode


            epoch_start = time.time()
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                iter_start = time.time()
                
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                        iter_end = time.time()
                        
                        history.append({'epoch': epoch, 
                                       'iteration': training_iter, 
                                       'loss': loss.item(), 
                                       'iter_time': iter_end - iter_start
                                        })
                        training_iter += 1

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_end = time.time()

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Time: {epoch_end - epoch_start:.4f}s')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    
    history_DF = pd.DataFrame(history)
    
    return model, history_DF



def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # from torchvision.models import resnet50, ResNet50_Weights
    # Fetch model
    #model = torch.hub.load('pytorch/vision:v0.13.0', 'resnet50', pretrained=True)
    
    model = models.resnet50(pretrained=True)

    model = model.to(device)

    # Set up criterion
    criterion = nn.CrossEntropyLoss()

    T_max = 200


    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

    # Send model to device
    model = model.to(device)

    dataloaders, dataset_sizes, class_names = load_data(num_workers=2)

    # Train
    model, history = train_model(model, 
                                device = device, 
                                dataloaders = dataloaders, 
                                dataset_sizes = dataset_sizes,
                                criterion = criterion, 
                                optimizer = optimizer, 
                                scheduler = scheduler, 
                                num_epochs = 350,
                                acc=0.92)


    # save model
    output_file = 'resnet50.csv'
    history.to_csv(output_file, index=False)
    
    
if __name__ == "__main__":
    main()
