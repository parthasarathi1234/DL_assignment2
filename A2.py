import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SubsetRandomSampler
# !pip install wandb
import wandb
import matplotlib.pyplot as plt
import numpy as np
from wandb.keras import WandbCallback
import socket
import argparse
socket.setdefaulttimeout(30)
wandb.login()
wandb.init(project="cs23m035_DL_Assignment2")

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class SmallCNN(nn.Module):
    def __init__(self, num_filters=64, activation='ReLU',  data_augmentation='Yes',  batch_normalization='No',  dense_neurons=256,  dropout=0.3 ,filter_size=3):
        # print("parthu")
        self.activation=activation
        self.num_filters=num_filters
        super(SmallCNN, self).__init__()

        # 1st convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=num_filters, kernel_size=filter_size, padding=1)
        # 2nd convolutional layer
        self.conv2 = nn.Conv2d(num_filters, num_filters * 2, kernel_size=filter_size, padding=1)
        # 3rd convolutional layer
        self.conv3 = nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=filter_size, padding=1)
        # 4th convolutional layer
        self.conv4 = nn.Conv2d(num_filters * 4, num_filters * 8, kernel_size=filter_size, padding=1)
        # 5th convolutional layer
        self.conv5 = nn.Conv2d(num_filters * 8, num_filters * 16, kernel_size=filter_size, padding=1)
        # max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # dense layer
        self.fc1 = nn.Linear(num_filters * 16 * 4*4, dense_neurons)
        # output dense layer
        self.fc2 = nn.Linear(dense_neurons, 10)  # Output layer with 10 neurons for classification

        self.dropout=nn.Dropout(dropout)
        self.batch_norm=nn.BatchNorm2d(num_filters) if batch_normalization else None

    def forward(self, x):
        if(self.activation=='ReLU'):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = self.pool(F.relu(self.conv4(x)))
            x = self.pool(F.relu(self.conv5(x)))
        elif(self.activation=='sigmoid'):
            x = self.pool(F.sigmoid(self.conv1(x)))
            x = self.pool(F.sigmoid(self.conv2(x)))
            x = self.pool(F.sigmoid(self.conv3(x)))
            x = self.pool(F.sigmoid(self.conv4(x)))
            x = self.pool(F.sigmoid(self.conv5(x)))
        elif(self.activation=='tanh'):
            x = self.pool(F.tanh(self.conv1(x)))
            x = self.pool(F.tanh(self.conv2(x)))
            x = self.pool(F.tanh(self.conv3(x)))
            x = self.pool(F.tanh(self.conv4(x)))
            x = self.pool(F.tanh(self.conv5(x)))

        elif (self.activation == 'GELU'):
            x = self.pool(F.gelu(self.conv1(x)))
            x = self.pool(F.gelu(self.conv2(x)))
            x = self.pool(F.gelu(self.conv3(x)))
            x = self.pool(F.gelu(self.conv4(x)))
            x = self.pool(F.gelu(self.conv5(x)))
        elif (self.activation == 'SiLU'):
            x = self.pool(F.silu(self.conv1(x)))
            x = self.pool(F.silu(self.conv2(x)))
            x = self.pool(F.silu(self.conv3(x)))
            x = self.pool(F.silu(self.conv4(x)))
            x = self.pool(F.silu(self.conv5(x)))
        elif (self.activation == 'Mish'):
            x = self.pool(F.mish(self.conv1(x)))
            x = self.pool(F.mish(self.conv2(x)))
            x = self.pool(F.mish(self.conv3(x)))
            x = self.pool(F.mish(self.conv4(x)))
            x = self.pool(F.mish(self.conv5(x)))

        x = x.view(-1, self.num_filters * 16 * 4*4)

        if(self.activation=='ReLU'):
            x = self.dropout(F.relu(self.fc1(x)))
        elif(self.activation=='sigmoid'):
            x = self.dropout(F.sigmoid(self.fc1(x)))
        elif(self.activation=='tanh'):
            x = self.dropout(F.tanh(self.fc1(x)))

        elif(self.activation=='GELU'):
            x = self.dropout(F.gelu(self.fc1(x)))
        elif(self.activation=='SiLU'):
            x = self.dropout(F.silu(self.fc1(x)))
        elif(self.activation=='Mish'):
            x = self.dropout(F.mish(self.fc1(x)))

        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



def train_network(num_filters, activation, data_augmentation, batch_normalization, dense_neurons, dropout, learning_rate,epochs,optimizer,filter_size):
    # with wandb.init() as run:
    #     config=wandb.config
    # print("hell")
    cnn_model = SmallCNN(num_filters, activation, data_augmentation, batch_normalization, dense_neurons, dropout,filter_size).to(device)

    criterion = nn.CrossEntropyLoss()

    if(optimizer=='adam'):
        optimizer=optim.Adam(cnn_model.parameters(),lr=learning_rate)
    else:
        optimizer=optim.Nadam(cnn_model.parameters(),lr=learning_rate)

    for i in range(epochs):
        cnn_model.train()
        train_loss=0.0
        train_correct=0
        train_total=0
        for image,label in train_loader:
            image=image.to(device=device)  # image moving to cpu/gpu
            label=label.to(device=device)  # label is moving to cpu/gpu

            optimizer.zero_grad()   # clear the gradients of all optimizers tensors
            scores=cnn_model(image) # images are passed through the CNN to obtain the raw output scores(representing the predicted class scores)
            loss=criterion(scores,label) # predicted scores and the true label are used to compute the loss

            loss.backward()  #  computes the gradients of the loss with respect to all model parameters
            #gradient descent or adam step
            optimizer.step() # updates the parameters of the model using the computed gradients and the chosen optimization alg(SGD, Adam)

            train_loss+=loss.item()  # loss value computed for the current batch
            _,predicted=scores.max(1) # returns a tuple containing the maximum value and its corresponding index along the specified dimension('dim=1').
                                    # '_' to discard the maximum values, 'predicted' captures the predicted class labels for each image in the batch
            train_total+=label.size(0) # this increament by the numbers of samples in the current batch
                                    # label.size(0) return the batch size, which corresponds to the no of samples
            train_correct+=predicted.eq(label).sum().item()
                                    # we calculate the number of correctly predicted samples in the current batch
                                    # predicted.eq(labels) performs element wise comparison between predicted and true labels, resulting in a tensor of boolean values indicating whether each prediction is correct.
                                    # .sum().item() calculates the total no of correct predictions and adds it to the train_correct
        train_loss=train_loss/len(train_loader)
        train_accuracy=100*train_correct/train_total


        num_correct=0
        num_loss=0
        total=0
        cnn_model.eval()
        with torch.no_grad():
            for x,y in val_loader:
                x=x.to(device=device)
                y=y.to(device=device)
                scores=cnn_model(x)
                loss=criterion(scores,y)
                num_loss+=loss.item()
                _,predictions=scores.max(1)
                total+=y.size(0)
                num_correct+=predictions.eq(y).sum().item()
        val_accuracy=100*num_correct/total
        val_loss=num_loss/len(val_loader)
        wandb.log({"Train_Accuracy" : train_accuracy,"Train_Loss" : train_loss,"Validation_acc" : val_accuracy,"validation_loss" : val_loss,'epoch':i})
        print(f"Train_Accuracy : {train_accuracy},Train_Loss : {train_loss}, Validation_acc : {val_accuracy},validation_loss : {val_loss},epoch:{i}")


    num_correct=0
    num_loss=0
    total=0
    cnn_model.eval()
    count=0
    i=1
    with torch.no_grad():
        for x,y in test_loader:
            x=x.to(device=device)
            y=y.to(device=device)
            # x=x.reshape(x.shape[0],-1)
            scores=cnn_model(x)
            loss=criterion(scores,y)
            num_loss+=loss.item()
            _,predictions=scores.max(1)
            if(i%2==0):
                if(count>=30):
                    break
                if(count%3==0):
                    plt.figure(figsize=(10,10))
                plt.subplot(10,3,count+1)
                plt.xticks([])
                plt.yticks([])
                plt.imshow(x[0].cpu().numpy().transpose(1, 2, 0))
                plt.xlabel(f'True: {class_labels[y[0].item()]},\npredicted: {class_labels[predictions[0].item()]}' )
                count+=1
            i+=1
            total+=y.size(0)
            num_correct+=predictions.eq(y).sum().item()
    val_accuracy=100*num_correct/total
    val_loss=num_loss/len(test_loader)
    wandb.log({"Test_acc" : val_accuracy,"Test_loss" : val_loss,'epoch':i})
    print(f" Test_acc : {val_accuracy},Test_loss : {val_loss},epoch:{i}")




def parse_arguments():
  parser = argparse.ArgumentParser(description='Training Parameters')
  parser.add_argument('-wp', '--wandb_project', type=str, default='cs23m035_DL_Assignment2',
                        help='Project name')
  
  parser.add_argument('-we', '--wandb_entity', type=str, default='Entity_DL',
                        help='Wandb Entity')
  
  parser.add_argument('-e', '--epochs', type=int, default=10,help='Number of epochs for training network')

  parser.add_argument('-sF', '--filter_size', type= int, default=3, choices = [3,5],help='Choice of kernel size')
  
  parser.add_argument('-o', '--optimizer', type=str, default='adam', choices = ["adam", "nadam"],help='Choice of optimizer')
   
  parser.add_argument('-lr', '--learning_rate', type=int, default=0.0001, help='Learning rate')

  parser.add_argument( '-eps', '--epsilon', type=int, default=0.000001, help='Epsilon used by optimizers')
  
  parser.add_argument( '-a','--activation', type=str, default="ReLU",choices=['ReLU','sigmoid','tanh','GELU','SiLU','Mish'], help='activation functions')

  parser.add_argument('-nof', '--num_filters', type=int, default=64, choices = [32, 64], help='Number of filters rate')

  parser.add_argument('-da', '--data_augmentation', type=str, default='Yes', choices = ['Yes','No'], help='Data augmentation')

  parser.add_argument('-bn', '--batch_normalization', type=str, default='No', choices = ['Yes','No'], help='Batch Normalization')

  parser.add_argument('-do', '--dropout', type=int, default=0.3, choices = [0.2,0.3], help='Dropout')

  parser.add_argument('-dn', '--dense_neurons', type=int, default=256, choices = [64,128,256,512], help='Dense Neurons')

  return parser.parse_args()


args = parse_arguments()

wandb.init(project=args.wandb_project)

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229, 0.224, 0.225])
])

train_dataset=datasets.ImageFolder(root='/content/inaturalist_12K/train',transform=transform)
test_dataset=datasets.ImageFolder(root='/content/inaturalist_12K/val',transform=transform)

train_indices, val_indices = train_test_split(list(range(len(train_dataset))), test_size=0.2, random_state=42)

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_dataset)

train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)
val_loader = DataLoader(train_dataset, batch_size=32, sampler=val_sampler)
test_loader = DataLoader(test_dataset, batch_size=32)

train_network(args.num_filters, args.activation, args.data_augmentation, args.batch_normalization, args.dense_neurons, args.dropout, args.learning_rate,args.epochs,args.optimizer,args.filter_size)
# train_network(64, 'ReLU', 'Yes', 'No', 256, 0.3, 0.00001,10)

