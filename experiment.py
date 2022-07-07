import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import torch
device = torch.device("cuda")
import torch.nn.functional as F
import wandb
from torchvision.models import resnet18, resnet34, resnet50
from torch.utils import data
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
from torch.optim import Adam, SGD, RMSprop
import numpy as np
from large_margin import LargeMarginLoss
import time
import argparse

def test(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad(): 
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, *_ = model(data)
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            _, idx = output.max(dim=1)
            correct += (idx == target).sum().item()

            # Clear the computed features
            model.clear_features()

    accuracy = 100. * correct / len(test_loader.dataset)
    print('Test set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset), accuracy))

    wandb.log({"accuracy": accuracy})


def test_ce(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad(): 
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            _, idx = output.max(dim=1)
            correct += (idx == target).sum().item()

    accuracy = 100. * correct / len(test_loader.dataset)
    print('Test set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset), accuracy))

    wandb.log({"accuracy": accuracy})

def _init_resnet_18(output_size, pretrained = False, features_hook = None):
    model = resnet18(pretrained=pretrained)
    model.fc = torch.nn.Linear(512, output_size)
    if features_hook is not None:
        for name, module in model.named_modules():
            if name in ['layer1', 'layer2', 'layer3', 'layer4']:
                module.register_forward_hook(features_hook)

    return model
    
def _init_resnet_34(output_size, pretrained = False, features_hook = None):
    model = resnet34(pretrained=pretrained)
    model.fc = torch.nn.Linear(1024, output_size)
    if features_hook is not None:
        for name, module in model.named_modules():
            if name in ['layer1', 'layer2', 'layer3', 'layer4']:
                module.register_forward_hook(features_hook)

    return model

def _init_resnet_50(output_size, pretrained = False, features_hook = None):
    model = resnet50(pretrained=pretrained)
    model.fc = torch.nn.Linear(2048, output_size)
    if features_hook is not None:
        for name, module in model.named_modules():
            if name in ['layer1', 'layer2', 'layer3', 'layer4']:
                module.register_forward_hook(features_hook)

    return model

def create_pretrained_model(architecture, n_classes, features_hook = None):
    pretrained = True
    if 'resnet18' in architecture:
        net = _init_resnet_18(n_classes, pretrained, features_hook)
    elif 'resnet34' in architecture:
        net = _init_resnet_34(n_classes, pretrained, features_hook)
    elif 'resnet50' in architecture:
        net = _init_resnet_50(n_classes, pretrained, features_hook)
    else:
        raise NotImplementedError()

    return net
    
def create_model(architecture, n_classes, features_hook = None):
    pretrained = False
    if 'resnet18' in architecture:
        net = _init_resnet_18(n_classes, pretrained, features_hook)
    elif 'resnet34' in architecture:
        net = _init_resnet_34(n_classes, pretrained, features_hook)
    elif 'resnet50' in architecture:
        net = _init_resnet_50(n_classes, pretrained, features_hook)
    else:
        raise NotImplementedError()

    return net

class FeatureExtractor(torch.nn.Module):
    def __init__(self, architecture, n_classes = None):
        super().__init__()
        self._features = []
        if 'pretrained' in architecture:
            self.model = create_pretrained_model(
                architecture, 
                n_classes, 
                features_hook=self.feature_hook)
        else:
            self.model = create_model(
                architecture, 
                n_classes, 
                features_hook=self.feature_hook)

    def feature_hook(self, module, input, output):
        self._features.append(output)

    def forward(self, x):
        logits = self.model(x)
        return logits, self._features

    def clear_features(self):
        self._features = []

def train_lm(model, train_loader, optimizer, epoch, lm):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        one_hot = torch.zeros(len(target), 100).scatter_(1, target.unsqueeze(1), 1.).float()
        one_hot = one_hot.cuda()
        optimizer.zero_grad()
        output, features = model(data)
        for feature in features:
            feature.retain_grad()

        loss = lm(output, one_hot, features)
        
        wandb.log({"loss": loss})
        # optional
        wandb.watch(model)
        
        loss.backward()
        optimizer.step()
        model.clear_features()
        
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def learn(args):
    pass

def main():
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--architecture", type=str,
                        help="Which architecture to use", default="resnet18")
    parser.add_argument("-p", "--pretrained", action="store_true")
    parser.add_argument("-l", "--loss", type=str, 
                        help="Which loss function to use ('CE' or 'margin_#')", default="CE")
    parser.add_argument("-d", "--detection_type", type=str, 
                        help="Which type of anomaly detection to use ('KS' or 'LS')", type=str)
    
    args = parser.parse_args()

    # if args.architecture == 'resnet50':
    #   print('Holy cow, you're using a resnet50!')

    # Setup Weights and Biases and specify hyperparameters
    wandb.init(project="Thomas-Masters-Project")

    learning_rate = 0.001
    epochs = 5
    batch_size = 256
    net_type = "pretrained_resnet18"
    num_training_classes = 100

    wandb.config = {
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "network": net_type
    }

    train_loader = data.DataLoader(
            datasets.CIFAR100('./data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229, 0.224, 0.225])
                        ])),
            batch_size=batch_size, shuffle=True, drop_last=True)

    test_loader = data.DataLoader(
            datasets.CIFAR100('./data', train=False,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229, 0.224, 0.225])
                        ])),
            batch_size=batch_size, shuffle=False, drop_last=False)

    lm = LargeMarginLoss(
        gamma=10000,
        alpha_factor=4,
        top_k=num_training_classes,
        dist_norm=np.inf
    )

    net = FeatureExtractor(net_type, num_training_classes)
    net.to(device)

    optim = Adam(net.parameters()) #SGD(net.parameters(), lr=learning_rate, momentum=0)
    for i in range(0, epochs):
        start_time = time.time()
        train_lm(net, train_loader, optim, i, lm)
        end_time = time.time()

        print('Epoch {} took {} seconds to complete'.format(i+1, end_time-start_time))

        test(net, test_loader)

if __name__ == '__main__':
    main()