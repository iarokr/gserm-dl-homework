# import dependencies
import os
from datetime import datetime
import numpy as np
from time import perf_counter

import pandas as pd
import torch
import torchvision
from torch import nn, optim

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score

# Functions for training and evaluation process
# function for the training process


def model_training(in_model, device, learning_rate, num_epochs, minibatch_size, train_data, models_directory):
    model = in_model.to(device)

    # print the class name of the model
    print('[LOG] Training {}...'.format(str(model.__class__.__name__)))

    # print the initialized architectures
    print('[LOG] Model architecture:\n\n{}\n'.format(model))

    # init the number of model parameters
    num_params = 0

    # iterate over the distinct parameters
    for param in model.parameters():

        # collect number of parameters
        num_params += param.numel()

    # print the number of model parameters
    print(f'[LOG] Number of to be trained {str(model.__class__.__name__)} model parameters: {num_params}, epochs={num_epochs}, learning rate={learning_rate}, mini batch size={minibatch_size}\n')

    # define the optimization criterion / loss function
    nll_loss = nn.NLLLoss()
    nll_loss = nll_loss.to(device)

    # define learning rate and optimization strategy
    learning_rate = learning_rate
    optimizer = optim.SGD(params=model.parameters(), lr=learning_rate)
    # specify the training parameters
    num_epochs = num_epochs  # number of training epochs
    mini_batch_size = minibatch_size  # size of the mini-batches
    fashion_mnist_train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=mini_batch_size, shuffle=True)

    # init collection of training epoch losses
    train_epoch_losses = []

    # set the model in training mode
    model.train()

    start_time = perf_counter()
    # train the FashionMNISTNet model
    for epoch in range(num_epochs):
        # init collection of mini-batch losses
        train_mini_batch_losses = []

        # iterate over all-mini batches
        for i, (images, labels) in enumerate(fashion_mnist_train_dataloader):

            # push mini-batch data to computation device
            images = images.to(device)
            labels = labels.to(device)

            # run forward pass through the network
            output = model(images)

            # reset graph gradients
            model.zero_grad()

            # determine classification loss
            loss = nll_loss(output, labels)

            # run backward pass
            loss.backward()

            # update network parameters
            optimizer.step()

            # collect mini-batch reconstruction loss
            train_mini_batch_losses.append(loss.data.item())

        # determine mean min-batch loss of epoch
        train_epoch_loss = np.mean(train_mini_batch_losses)

        # print epoch loss
        now = datetime.utcnow().strftime("%Y%m%d-%H:%M:%S")
        print('[LOG {}] epoch: {} train-loss: {}'.format(str(now), str(epoch), str(train_epoch_loss)))

        # set filename of actual model
        model_name = f'{str(model.__class__.__name__)}_model_e{num_epochs}_mb{mini_batch_size}_lr{learning_rate}_epoch_{epoch}.pth'

        # save model to local directory
        torch.save(model.state_dict(), os.path.join(models_directory, model_name))

        # determine mean min-batch loss of epoch
        train_epoch_losses.append(train_epoch_loss)

    # assign the index of the epoch with the lowest loss
    min_epoch = np.argmin(train_epoch_losses)
    min_loss = np.min(train_epoch_losses)

    end_time = perf_counter()

    tpe = (end_time-start_time)/num_epochs

    return str(model.__class__.__name__), train_epoch_losses, min_epoch, min_loss, tpe

# function for evaluating the trained model
def model_evaluation(model, path_to_best_model: str, eval_data, batch_size: int):
    best_model = model

    # load pre-trained models
    best_model.load_state_dict(torch.load(path_to_best_model, map_location=torch.device('cpu')))

    # set model in evaluation mode
    best_model.eval()

    # activate DataLoader for evaluation dataset
    fashion_mnist_eval_dataloader = torch.utils.data.DataLoader(eval_data, batch_size=batch_size, shuffle=False)

    # init collection of mini-batch losses
    eval_mini_batch_losses = []

    device = torch.device('cpu').type
    best_model.to(device)

    # iterate over all-mini batches
    for i, (images, labels) in enumerate(fashion_mnist_eval_dataloader):

        # push mini-batch data to computation device
        images = images.to(device)
        labels = labels.to(device)

        # run forward pass through the network
        output = best_model(images)

        # determine classification loss
        nll_loss = nn.NLLLoss()
        loss = nll_loss(output, labels)

        # collect mini-batch reconstruction loss
        eval_mini_batch_losses.append(loss.data.item())

    # determine mean min-batch loss of epoch
    eval_loss = np.mean(eval_mini_batch_losses)

    # print epoch loss
    now = datetime.utcnow().strftime("%Y%m%d-%H:%M:%S")
    print('[LOG {}] eval-loss: {}'.format(str(now), str(eval_loss)))

    predictions = torch.argmax(best_model(next(iter(fashion_mnist_eval_dataloader))[0]), dim=1)

    # determine classification matrix of the predicted and target classes
    mat = confusion_matrix(eval_data.targets, predictions.detach())

    return eval_loss, mat, accuracy_score(eval_data.targets, predictions.detach())

def main():
    # Environment setup
    print('[LOG] running locally')

    # create the data sub-directory
    data_directory = './data_fmnist'
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    # create the models sub-directory
    models_directory = './models_fmnist'
    if not os.path.exists(models_directory):
        os.makedirs(models_directory)

    # initialize the seed
    seed = 42
    np.random.seed(seed)

    # select the device to be used for training
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu').type

    # init seed for every device
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu

    # log type of device enabled
    print('[LOG] notebook with {} computation enabled'.format(str(device)))

    # specify training path for loading
    train_path = data_directory + '/train_fmnist'

    # define pytorch transformation into tensor format
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    # download and transform training images
    fashion_mnist_train_data = torchvision.datasets.FashionMNIST(root=train_path, train=True, download=True,
                                                                 transform=transform)

    eval_path = data_directory + '/eval_fmnist'

    # define pytorch transformation into tensor format
    transf = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    # download and transform training images
    fashion_mnist_eval_data = torchvision.datasets.FashionMNIST(root=eval_path, train=False, transform=transf,
                                                                download=True)

    # define fashion mnist classes
    fashion_classes = {
        0: 'T-shirt/top',
        1: 'Trouser',
        2: 'Pullover',
        3: 'Dress',
        4: 'Coat',
        5: 'Sandal',
        6: 'Shirt',
        7: 'Sneaker',
        8: 'Bag',
        9: 'Ankle boot'}

    eval_path = data_directory + '/eval_fmnist'

    # Model 4
    class IaroHasNoFashionNet4(nn.Module):

        # define the class constructor
        def __init__(self):

            # call super class constructor
            super(IaroHasNoFashionNet4, self).__init__()

            # specify convolution layer 1
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3, stride=1, padding=0)

            # define max-pooling layer 1
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

            # specify convolution layer 2
            self.conv2 = nn.Conv2d(in_channels=20, out_channels=60, kernel_size=3, stride=1, padding=1)

            # specify fully-connected (fc) layer 1 - in 8*8*20, out 640
            self.linear1 = nn.Linear(6*6*60, 1000, bias=True)  # the linearity W*x+b
            self.relu1 = nn.ReLU(inplace=True)  # the non-linearity

            # specify fc layer 2 - in 640, out 100
            self.linear2 = nn.Linear(1000, 100, bias=True)  # the linearity W*x+b

            # specify fc layer 2 - in 100, out 10
            self.linear3 = nn.Linear(100, 10, bias=True)  # the linearity W*x+b

            # add a softmax to the last layer
            self.logsoftmax = nn.LogSoftmax(dim=1)  # the softmax

        # define network forward pass
        def forward(self, images):
            # high-level feature learning via convolutional layers

            # define conv layer 1 forward pass
            x = self.pool1(self.relu1(self.conv1(images)))

            # define conv layer 2 forward pass
            x = self.pool1(self.relu1(self.conv2(x)))

            # reshape image pixels
            x = x.view(-1, 6*6*60)

            # define fc layer 1 forward pass
            x = self.relu1(self.linear1(x))

            # define fc layer 2 forward pass
            x = self.relu1(self.linear2(x))

            # define layer 3 forward pass
            x = self.logsoftmax(self.linear3(x))

            # return forward pass result
            return x


    # Model 4 Training
    m4_results = []
    batch_size = 10000
    for lr in [0.0005, 0.001, 0.005, 0.01]:
        for mini_batch_size in [8, 16, 32, 64, 128, 256, 512]:
            num_epochs = 100

            in_model3 = IaroHasNoFashionNet4()

            m4, train_epoch_losses4, min_epoch4, min_loss4, tpe = model_training(in_model3, device, lr, num_epochs, mini_batch_size, fashion_mnist_train_data, models_directory)


            # evaluate the trained model
            bm = f'{m4}_model_e{num_epochs}_mb{mini_batch_size}_lr{lr}_epoch_{min_epoch4}.pth'
            eval_loss, mat, acc = model_evaluation(IaroHasNoFashionNet4(),'./models_fmnist/'+bm, fashion_mnist_eval_data, batch_size)
            m4_results.append([m4, num_epochs, lr, mini_batch_size, train_epoch_losses4, min_epoch4, min_loss4, tpe, eval_loss, mat, acc])

    results = pd.DataFrame(m4_results, columns=['model', 'num_epochs', 'lr', 'mini_batch_size', 'train_epoch_losses', 'min_epoch', 'min_loss', 'time_per_epoch', 'eval_loss', 'conf_matrix', 'accuracy'])
    results.to_pickle(models_directory + '/m4_training_results.pkl')

    pass


main()
