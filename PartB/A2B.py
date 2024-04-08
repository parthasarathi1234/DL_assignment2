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

import socket
import argparse
socket.setdefaulttimeout(30)
wandb.login()
wandb.init(project="cs23m035_DL_Assignment2")

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def ResNet50_uptil_k(k,classes):
    model = models.resnet50(pretrained=True)
    
    model_params=list(model.parameters())
    for parameter in model_params[:k]:
        parameter.requires_grad=False
    
    
    num_filters=model.fc.in_features
    model.fc=torch.nn.Linear(num_filters,classes)
    
    return model



def RESNET50(NUM_OF_CLASSES):
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, NUM_OF_CLASSES)
    
    for param in model.parameters():
        param.requires_grad = False
        
    for param in model.fc.parameters():
        param.requires_grad = True
   
    return model





def pretrain_model(epochs,learningrate,freeze,freeze_value):   
    if(freeze=='False'):
        cnn_model=RESNET50(10).to(device=device)
    else:
        cnn_model=RESNET50(freeze_value,10).to(device=device)


    criterion = nn.CrossEntropyLoss()
    optimizer=optim.Adam(cnn_model.parameters(),lr=learningrate)


    for i in range(epochs):
        print(i)
        train_loss=0.0
        train_correct=0
        train_total=0
        for image,label in train_loader:
            image=image.to(device=device)
            label=label.to(device=device)

            optimizer.zero_grad()
            scores=cnn_model(image)
            loss=criterion(scores,label)

            loss.backward()
            #gradient descent or adam step
            optimizer.step()

            train_loss+=loss.item()
            _,predicted=scores.max(1)
            train_total+=label.size(0)
            train_correct+=predicted.eq(label).sum().item()
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
                # x=x.reshape(x.shape[0],-1)
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
    with torch.no_grad():
        for x,y in test_loader:
            x=x.to(device=device)
            y=y.to(device=device)
            # x=x.reshape(x.shape[0],-1)
            scores=cnn_model(x)
            loss=criterion(scores,y)
            num_loss+=loss.item()
            _,predictions=scores.max(1)
            total+=y.size(0)
            num_correct+=predictions.eq(y).sum().item()
    val_accuracy=100*num_correct/total
    val_loss=num_loss/len(test_loader)
    wandb.log({"Test_acc" : val_accuracy,"Test_loss" : val_loss,'epoch':i})
    print(f" Test_acc : {val_accuracy},Test_loss : {val_loss},epoch:{i}")
        
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_arguments():
    parser = argparse.ArgumentParser(description='Training Parameters')

    parser.add_argument('-wp', '--wandb_project', type=str, default='cs23m035_DL_Assignment2',
                        help='Project name used to track experiments in Weights & Biases dashboard')
    
    
    parser.add_argument('-e', '--epochs', type= int, default=10, choices = [10,20,30],help='Number of epochs')

    parser.add_argument('-lE', '--learningrate', type= float, default=0.0001, choices = [0.0001,0.00001],help='Learning rates')

    parser.add_argument('-fr', '--freeze', type= str, default='False', choices = ['False','True'],help='Freeze')
    
    parser.add_argument('-fv', '--freeze_value', type= int, default=10, choices = [10,20,30],help='Choice of strategies')

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

pretrain_model(args.epochs,args.learningrate,args.freeze,args.freeze_value)