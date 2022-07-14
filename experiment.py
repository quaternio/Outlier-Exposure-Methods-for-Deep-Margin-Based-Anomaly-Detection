import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import torch
device = torch.device("cuda")
import torch.nn.functional as F
import wandb
import copy
from torchvision.models import resnet18, resnet34, resnet50, efficientnet_b1
from torch.utils import data
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
from torch.optim import Adam, SGD, RMSprop
import numpy as np
from large_margin import LargeMarginLoss
import time
import argparse
from data import build_split_datasets
import pickle as pkl
from random_split_generator import FourWayClassSplit
from train_eval import train_ce_ks, train_ce_ls, train_lm
from torch.utils.data import ConcatDataset

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

def _init_efficientnet_b1(output_size, pretrained = False, features_hook = None):
    model = efficientnet_b1(pretrained=pretrained)
    model.classifier = torch.nn.Linear(1280, output_size)
    if features_hook is not None:
        for name, module in model.named_modules():
            if name in ['features.0', 'features.1', 'features.2', 'features.3', 'features.4', 'features.5', 'features.6', 'features.7', 'features.8',]:
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
    elif 'efficientnet_b1' in architecture:
        net = _init_efficientnet_b1(n_classes, pretrained, features_hook)
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
    elif 'efficientnet_b1' in architecture:
        net = _init_efficientnet_b1(n_classes, pretrained, features_hook)
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
        return logits, copy.deepcopy(self._features)

    def clear_features(self):
        self._features = []

def learn(args):
    pass

def main():
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--architecture", type=str,
                        help="Which architecture to use", default="resnet18")
    parser.add_argument("-p", "--pretrained", action="store_true")
    parser.add_argument("-l", "--loss", type=str, 
                        help="Which loss function to use ('CE' or 'margin')", default="CE")
    parser.add_argument("-d", "--detection_type", type=str, 
                        help="Which type of anomaly detection to use ('KS' or 'LS')", default="LS")
    parser.add_argument("-t", "--test", action="store_true", 
                        help="Indicates that we wish to test instead of validate model.")
    parser.add_argument("-o", "--optimizer", type=str, default="SGD")
    parser.add_argument("--learning_rate", type=float, default=0.003)
    parser.add_argument("-m", "--momentum", type=float, default=0.9)
    parser.add_argument("-b", "--batch_size", type=int, default=256)
    parser.add_argument("-s", "--split", type=int, default=0)
    parser.add_argument("-e", "--num_epochs", type=int, default=50)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--dist_norm", type=str, default="2")
    # TODO: Finish adding arguments; start instrumenting for weights and biases sweep
    args = parser.parse_args()

    # if args.architecture == 'resnet50':
    #   print('Holy cow, you're using a resnet50!')

    # Setup Weights and Biases and specify hyperparameters
    wandb.init(project="Thomas-Masters-Project")

    wandb.config = {
        "learning_rate": args.learning_rate,
        "epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "network": args.architecture
    }

    norm_dict = {"1": 1, "2": 2, "inf": np.inf}

    split = args.split
    learning_rate = args.learning_rate
    epochs = args.num_epochs
    batch_size = args.batch_size
    net_type = args.architecture
    dist_norm = norm_dict[args.dist_norm]

    # Hard-coded for now. To change, re-generate splits via random_split_generator.py
    num_training_classes = 10
    num_total_classes = 100

    # Getting split definition
    with open('id-ood-splits/split{}.pkl'.format(split), 'rb') as f:
        id_ood_split = pkl.load(f) # Type is adtools.data.FourWayClassSplit

    # Any label not in id_labels is OOD
    id_labels = id_ood_split.id_labels()
    id_label_map = {}
    for label in range(num_total_classes):
        if label in id_labels:
            id_label_map[label] = True

    # Right now, hardcoded to use CIFAR-100, with hardcoded splits
    # defined and generated in random_split_generator.py
    id_data, ood_data = build_split_datasets(split)

    id_train_data, id_val_data, id_test_data    = id_data
    ood_train_data, ood_val_data, ood_test_data = ood_data

    # Constructing Dataloaders
    id_train_loader = data.DataLoader(id_train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    id_val_loader   = data.DataLoader(id_val_data, batch_size=batch_size, shuffle=True, drop_last=True)
    id_test_loader   = data.DataLoader(id_test_data, batch_size=batch_size, shuffle=True, drop_last=True)
    ood_train_loader   = data.DataLoader(ood_train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    ood_val_loader   = data.DataLoader(ood_val_data, batch_size=batch_size, shuffle=True, drop_last=True)
    ood_test_loader   = data.DataLoader(ood_test_data, batch_size=batch_size, shuffle=True, drop_last=True)

    #################
    # LOSS FUNCTION #
    #################
    if args.loss == "margin":
        lm = LargeMarginLoss(
            gamma=10000,
            alpha_factor=4,
            top_k=args.top_k,
            dist_norm=dist_norm #np.inf
        )

    if args.detection_type == "LS":
        net = FeatureExtractor(net_type, num_training_classes)
    elif args.detection_type == "KS":
        # Include a kitchen sink class
        net = FeatureExtractor(net_type, num_training_classes+1)

    net.to(device)

    # Data and Net accounted for; Note that net will need to be modified later for 
    # kitchen sink configuration. (The number of classes will just be 
    # num_training_classes+1)

    if args.optimizer == "SGD":
        optim = SGD(net.parameters(), lr=learning_rate, momentum=args.momentum)
    elif args.optimizer == "ADAM":
        optim = Adam(net.parameters())
    else:
        raise NotImplementedError("Specified Optimizer Not Supported")

    for i in range(0, epochs):
        start_time = time.time()
        if args.loss == "margin":
            train_lm(net, id_train_loader, ood_train_loader, optim, i, lm, num_training_classes, id_label_map, device)
        elif args.loss == "CE" and args.detection_type == "LS":
            train_ce_ls(net, id_train_loader, ood_train_loader, optim, i, id_label_map, device)
        else:
            raise NotImplementedError("Specified Optimizer Not Supported")

        end_time = time.time()

        print('Epoch {} took {} seconds to complete'.format(i+1, end_time-start_time))

        test(net, id_test_loader)

if __name__ == '__main__':
    main()
