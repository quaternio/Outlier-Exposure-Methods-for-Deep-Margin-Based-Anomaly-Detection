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
from train_eval import train_ce, train_ce_ks, train_ce_ls, train_lm_ls, train_lm_ks, test_ce_ks, test_ce_ls, test_lm_ks, test_lm_ls
from torch.utils.data import ConcatDataset
import datetime

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
            if name in ['layer1', 'layer2', 'layer3', 'layer4', 'fc']:
                module.register_forward_hook(features_hook)

    return model
    
def _init_resnet_34(output_size, pretrained = False, features_hook = None):
    model = resnet34(pretrained=pretrained)
    model.fc = torch.nn.Linear(1024, output_size)
    if features_hook is not None:
        for name, module in model.named_modules():
            if name in ['layer1', 'layer2', 'layer3', 'layer4', 'fc']:
                module.register_forward_hook(features_hook)

    return model

def _init_resnet_50(output_size, pretrained = False, features_hook = None):
    model = resnet50(pretrained=pretrained)
    model.fc = torch.nn.Linear(2048, output_size)
    if features_hook is not None:
        for name, module in model.named_modules():
            if name in ['layer1', 'layer2', 'layer3', 'layer4', 'fc']:
                module.register_forward_hook(features_hook)

    return model

def _init_efficientnet_b1(output_size, pretrained = False, features_hook = None):
    model = efficientnet_b1(pretrained=pretrained)
    model.classifier = torch.nn.Linear(1280, output_size)
    if features_hook is not None:
        for name, module in model.named_modules():
            if name in ['features.0', 'features.1', 'features.2', 'features.3', 'features.4', 'features.5', 'features.6', 'features.7', 'features.8', 'classifier']:
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
        return logits, self._features

    def clear_features(self):
        self._features = []

def learn(args):
    pass

def main():
    now = datetime.datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--architecture", type=str,
                        help="Which architecture to use", default="efficientnet_b1")
    parser.add_argument("-p", "--pretrained", action="store_true")
    parser.add_argument("-b", "--baseline", action="store_true")
    parser.add_argument("-l", "--loss", type=str, 
                        help="Which loss function to use ('CE' or 'margin')", default="CE")
    parser.add_argument("-d", "--detection_type", type=str, 
                        help="Which type of outlier exposure to use ('KS' or 'LS')", default="KS")
    parser.add_argument("-t", "--test", action="store_true", 
                        help="Indicates that we wish to test instead of validate model.")
    parser.add_argument("-o", "--optimizer", type=str, default="SGD")
    parser.add_argument("--learning_rate", type=float, default=0.0542)
    parser.add_argument("-m", "--momentum", type=float, default=0.242)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("-s", "--split", type=int, default=0)
    parser.add_argument("-e", "--num_epochs", type=int, default=300)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--dist_norm", type=str, default="2")
    parser.add_argument("--gamma", type=int, default=19600)
    parser.add_argument("--alpha_factor", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--oe_test", action="store_true",
                        help="Indicates type of testing we want to do. If true, test outliers will be OE classes.")
    args = parser.parse_args()

    # Make sure that random seed 0 is used with split 0
    torch.manual_seed(args.seed)

    # Setup Weights and Biases and specify hyperparameters
    test_method = "oe_test" if args.oe_test else "new_class_test"
    if args.baseline:
        project_name = "TN_Masters_Proj_Fixed_Baseline_Val_{}".format(test_method)
    else:
        project_name = "TN_Masters_Proj_Val_{}_{}_{}".format(args.detection_type, args.loss, test_method)
    
    wandb.init(project=project_name)

    wandb.define_metric("ID_Accuracy", summary="max")
    wandb.define_metric("AUROC", summary="max")

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
    id_data, ood_data = build_split_datasets(split, args.oe_test)

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
            gamma=args.gamma, #10000,
            alpha_factor=args.alpha_factor, #4,
            top_k=args.top_k,
            dist_norm=dist_norm
        )

    if args.detection_type == "LS" or args.baseline:
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

    metric_combined_running_max = 0
    for i in range(0, epochs):
        # start_time = time.time()
        if args.baseline:
            loss = train_ce(net, id_train_loader, optim, i, id_label_map, device)
        else:
            if args.loss == "margin" and args.detection_type == "LS":
                loss = train_lm_ls(net, lm, id_train_loader, ood_train_loader, optim, i, id_label_map, device)
            elif args.loss == "margin" and args.detection_type == "KS":
                loss = train_lm_ks(net, lm, id_train_loader, ood_train_loader, optim, i, id_label_map, device)
            elif args.loss == "CE" and args.detection_type == "LS":
                loss = train_ce_ls(net, id_train_loader, ood_train_loader, optim, i, id_label_map, device)
            elif args.loss == "CE" and args.detection_type == "KS":
                loss = train_ce_ks(net, id_train_loader, ood_train_loader, optim, i, id_label_map, device)
            else:
                raise NotImplementedError("Training for the specified loss-function outlier exposure combination is not supported")

        # end_time = time.time()
        # print('Epoch {} took {} seconds to complete'.format(i+1, end_time-start_time))

        id_eval_loader  = id_val_loader
        ood_eval_loader = ood_val_loader 
        if args.test:
            id_eval_loader  = id_test_loader
            ood_eval_loader = ood_test_loader

        if args.baseline:
            acc, auc = test_ce_ls(net, id_eval_loader, ood_eval_loader, device)
        else:
            if args.loss == "margin" and args.detection_type == "LS":
                acc, auc = test_lm_ls(net, lm, id_eval_loader, ood_eval_loader, device)
            elif args.loss == "margin" and args.detection_type == "KS":
                acc, auc = test_lm_ks(net, i, id_eval_loader, ood_eval_loader, device)
            elif args.loss == "CE" and args.detection_type == "LS":
                acc, auc = test_ce_ls(net, id_eval_loader, ood_eval_loader, device)
            elif args.loss == "CE" and args.detection_type == "KS":
                acc, auc = test_ce_ks(net, i, id_eval_loader, ood_eval_loader, device)
            else:
                raise NotImplementedError("Testing for the specified loss-fucntion outlier exposure combination is not supported")

        metric_combined = 0.5 * (acc/100.) + 0.5 * auc
        wandb.log({"loss": loss, "ID_Accuracy": acc, "AUROC": auc, "metric_combined": metric_combined, "epoch": i})

        # Save Model
        # if i % 5 == 0 or metric_combined > metric_combined_running_max:
        #     if metric_combined > metric_combined_running_max:
        #         metric_combined_running_max = metric_combined

        if args.baseline:
            directory = "val_baseline_{}".format(test_method)
            torch.save({
                'epoch': i,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'auc': auc,
                'id_accuracy': acc,
                'split': args.split,
                'test_method': test_method,
            }, 'models/{}/day_{}_{}_time_{}_{}_split_{}_epoch_{}.pth'.format(directory, now.month, now.day, now.hour, now.minute, args.split, i))
        else:
            directory = "val_{}_{}_{}".format(args.loss, args.detection_type, test_method)
            torch.save({
                'epoch': i,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'auc': auc,
                'id_accuracy': acc,
                'split': args.split,
                'test_method': test_method,
                'loss': args.loss,
                'detection_type': args.detection_type,
            }, 'models/{}/day_{}_{}_time_{}_{}_split_{}_epoch_{}.pth'.format(directory, now.month, now.day, now.hour, now.minute, args.split, i))

if __name__ == '__main__':
    main()
