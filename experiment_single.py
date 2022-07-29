import torch
device = torch.device("cuda:0")
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
from train_eval import train_ce, train_ce_ks, train_ce_ls, train_lm_ks, test_ce_ks, test_ce_ls, test_lm_ks
from torch.utils.data import ConcatDataset
import datetime
from sklearn.metrics import roc_auc_score
from large_margin import _get_grad

def train_lm_ls(model, lm, beta, max_grad_norm, id_loader, ood_loader, optimizer, epoch, id_label_map, device):
    # For these lm training functions, compute nominal loss and clear features
    # before computing anomaly loss and clearing features. This will ensure 
    model.train()
    num_classes = 10
    for batch_idx, ((id_data, id_target), (ood_data, ood_target)) in enumerate(zip(id_loader, ood_loader)):
        # data = torch.vstack((id_data, ood_data))
        # data = data.to(device)
        ood_data, ood_target = ood_data.to(device), ood_target.to(device)
        id_one_hot = torch.zeros(len(id_target), num_classes).scatter_(1, id_target.unsqueeze(1), 1.).float()
        id_data, id_target = id_data.to(device), id_target.to(device)
        # ood_one_hot = (1/id_one_hot.shape[1])*torch.ones((len(ood_target), id_one_hot.shape[1])).to(device)
        id_one_hot = id_one_hot.to(device)
        #id_one_hot, ood_one_hot = id_one_hot.to(device), ood_one_hot.to(device)
        #one_hot = torch.vstack((id_one_hot, ood_one_hot))
        #one_hot = one_hot.cuda()
        optimizer.zero_grad()
        model.clear_features()
        id_output, id_features = model(id_data)
        for feature in id_features:
            feature.retain_grad()

        id_loss = lm(id_output, id_one_hot, id_features)

        model.clear_features()
        ood_output, _ = model(ood_data)
        ood_loss = torch.sum(torch.square(ood_output))

        # print("id_loss: {}".format(id_loss))
        # print("ood_loss: {}".format(ood_loss))

        loss = id_loss + beta * ood_loss

        #wandb.log({"loss": loss})
        #wandb.watch(model)

        loss.backward()

        # For gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        
        if batch_idx % 8 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, 2 * batch_idx * len(id_data), len(id_loader.dataset)+len(ood_loader.dataset),
                100. * 2 * batch_idx / (len(id_loader)+len(ood_loader)), loss.item()))

    return loss


def train_margin(model, lm, id_loader, optimizer, epoch, id_label_map, device):
    # For these lm training functions, compute nominal loss and clear features
    # before computing anomaly loss and clearing features. This will ensure 
    model.train()
    num_classes = 10
    for batch_idx, (id_data, id_target) in enumerate(id_loader):
        # data = torch.vstack((id_data, ood_data))
        # data = data.to(device)
        #ood_data, ood_target = ood_data.to(device), ood_target.to(device)
        id_one_hot = torch.zeros(len(id_target), num_classes).scatter_(1, id_target.unsqueeze(1), 1.).float()
        id_one_hot = id_one_hot.to(device)
        id_data, id_target = id_data.to(device), id_target.to(device)
        # ood_one_hot = (1/id_one_hot.shape[1])*torch.ones((len(ood_target), id_one_hot.shape[1])).to(device)
        #id_one_hot, ood_one_hot = id_one_hot.to(device), ood_one_hot.to(device)
        #one_hot = torch.vstack((id_one_hot, ood_one_hot))
        #one_hot = one_hot.cuda()
        optimizer.zero_grad()
        model.clear_features()
        id_output, id_features = model(id_data)
        for feature in id_features:
            feature.retain_grad()

        loss = lm(id_output, id_one_hot, id_features)

        model.clear_features()
        # ood_output, _ = model(ood_data)
        # ood_loss = torch.sum(torch.square(ood_output))

        # print("id_loss: {}".format(id_loss))
        # print("ood_loss: {}".format(ood_loss))

        #wandb.log({"loss": loss})
        #wandb.watch(model)

        loss.backward()

        # For gradient clipping
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        
        if batch_idx % 8 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(id_data), len(id_loader.dataset),
                100. * batch_idx / (len(id_loader)), loss.item()))

    return loss


def test_distance(logits, features, top_k, device):
    eps = 1e-8

    entries = 100
    # print("logits: {}".format(logits))
    # pd_logits = pd.DataFrame(logits.detach().cpu().numpy())
    prob = F.softmax(logits, dim=1)
    # print("prob: {}".format(prob))
    # pd_prob = pd.DataFrame(prob.detach().cpu().numpy())

    max_indices = torch.argmax(prob, dim=1)

    pseudo_correct_prob, _ = torch.max(prob, dim=1, keepdim=True)

    # print("pseudo_correct_prob: {}".format(pseudo_correct_prob))
    # print("pseudo_correct_prob shape: {}".format(pseudo_correct_prob.shape))

    pseudo_other_prob = torch.zeros(prob.shape).to(device)
    pseudo_other_prob.copy_(prob)
    pseudo_other_prob[torch.arange(prob.shape[0]),max_indices] = 0.

    # Grabs the next most likely class probabilities
    if top_k > 1:
        topk_prob, _ = pseudo_other_prob.topk(top_k, dim=1)
    else:
        topk_prob, _ = pseudo_other_prob.max(dim=1, keepdim=True)

    # print("pseudo correct prob shape: {}".format(pseudo_correct_prob.shape))
    # print("topk_prob shape: {}".format(topk_prob.shape))

    pseudo_diff_prob = pseudo_correct_prob - topk_prob

    for i, feature_map in enumerate(features):
        if i == len(features)-1:
            diff_grad = torch.stack([_get_grad(pseudo_diff_prob[:, i], feature_map) for i in range(top_k)],
                                dim=1)
            diff_gradnorm = torch.norm(diff_grad, p=2, dim=2)
            diff_gradnorm.detach_()
            distance = pseudo_diff_prob / (diff_gradnorm + eps)

    return distance


def test_lm_ls(model, topk, id_loader, ood_loader, device):
    model.eval()
    correct = 0
    anomaly_index = 10
    num_classes = 10
    anom_pred = []
    anom_labels = []
    anom_score_sequence = []
    pred_sequence = []
    target_sequence = []
    
    for batch_idx, ((id_data, id_target), (ood_data, ood_target)) in enumerate(zip(id_loader, ood_loader)):
        ood_target = anomaly_index * torch.ones_like(ood_target)

        id_one_hot = torch.zeros(len(id_target), num_classes).scatter_(1, id_target.unsqueeze(1), 1.).float()
        ood_one_hot = (1/id_one_hot.shape[1])*torch.ones((len(ood_target), id_one_hot.shape[1])).to(device)
        id_one_hot, ood_one_hot = id_one_hot.to(device), ood_one_hot.to(device)

        id_data, id_target   = id_data.to(device), id_target.to(device)
        ood_data, ood_target = ood_data.to(device), ood_target.to(device)

        model.clear_features()
        id_output, id_features  = model(id_data)
        for id_feature in id_features:
            id_feature.retain_grad()
        
        raw_id_distance = test_distance(id_output, id_features, topk, device)
        id_distance = torch.abs(raw_id_distance)
        ## print("raw id distance: {}".format(raw_id_distance))

        model.clear_features()
        ood_output, ood_features = model(ood_data)
        for ood_feature in ood_features:
            ood_feature.retain_grad()

        raw_ood_distance = test_distance(ood_output, ood_features, topk, device)
        ood_distance = torch.abs(raw_ood_distance)
        # print("id distance (should be same as above): {}".format(id_distance))
        ## print("raw ood distance: {}".format(raw_ood_distance))

        ood_pred   = ood_output.argmax(dim=1, keepdim=True)
        ood_anom_pred = [1. if ood_pred[i] == anomaly_index else 0. for i in range(len(ood_pred))]

        # Compute number of correctly classified id instances
        id_pred   = id_output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        _, id_idx = id_output.max(dim=1)
        correct += (id_idx == id_target).sum().item()
        id_anom_pred = [1. if id_pred[i] == anomaly_index else 0. for i in range(len(id_pred))]

        # Concatenate the list (order matters here)
        batch_anom_pred = ood_anom_pred + id_anom_pred
        anom_pred = anom_pred + batch_anom_pred

        pred_sequence.append(ood_pred)
        pred_sequence.append(id_pred)

        target_sequence.append(ood_target)
        target_sequence.append(id_target)

        # Compute anomaly scores
        # Use discriminant (distance) function to compute ood_scores
        ood_scores, _ = torch.max(-1 * ood_distance, dim=1)
        # print('ood scores: {}'.format(ood_scores))
        anom_score_sequence.append(ood_scores)
        for i in range(len(ood_target)):
            # 1 indicates "anomaly"
            anom_labels.append(1.)

        # Use discriminant function to compute id_scores
        id_scores, _ = torch.max(-1 * id_distance, dim=1)
        # print('id scores: {}'.format(id_scores))
        anom_score_sequence.append(id_scores)
        for i in range(len(id_target)):
            # 0 indicates "nominal"
            anom_labels.append(0.)

    # 
    anom_scores = torch.hstack(anom_score_sequence).cpu().detach().numpy()
    anom_labels = np.asarray(anom_labels)
    anom_pred = np.asarray(anom_pred)
    pred = torch.vstack(pred_sequence).cpu().detach().numpy()
    pred = np.ndarray.flatten(pred)
    targets = torch.hstack(target_sequence).cpu().detach().numpy()

    # print("anom score: {}".format(anom_scores))
    AUROC = roc_auc_score(anom_labels, anom_scores)

    accuracy = 100. * correct / len(id_loader.dataset)
    print('Test set: Accuracy: {}/{} ({:.0f}%)'.format(
        correct, len(id_loader.dataset), accuracy))
    print('Test Set: AUROC: {}\n'.format(AUROC))

    return accuracy, AUROC

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
                        help="Which loss function to use ('CE' or 'margin')", default="margin")
    parser.add_argument("-d", "--detection_type", type=str, 
                        help="Which type of outlier exposure to use ('KS' or 'LS')", default="LS")
    parser.add_argument("-t", "--test", action="store_true", 
                        help="Indicates that we wish to test instead of validate model.")
    parser.add_argument("-o", "--optimizer", type=str, default="SGD")
    parser.add_argument("--learning_rate", type=float, default=0.0542)
    parser.add_argument("-m", "--momentum", type=float, default=0.242)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("-s", "--split", type=int, default=0)
    parser.add_argument("-e", "--num_epochs", type=int, default=300)
    parser.add_argument("--top_k", type=int, default=9) #9)
    parser.add_argument("--dist_norm", type=str, default="2")
    parser.add_argument("--gamma", type=int, default=19600)
    parser.add_argument("--alpha_factor", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--oe_test", action="store_true",
                        help="Indicates type of testing we want to do. If true, test outliers will be OE classes.")
    parser.add_argument("--clip", type=float, default=100, help="Max grad norm before clipping")
    parser.add_argument("--beta", type=float, default=1)
    parser.add_argument("--margin_baseline", action="store_true")
    args = parser.parse_args()

    # Make sure that random seed 0 is used with split 0
    torch.manual_seed(args.seed)

    # Setup Weights and Biases and specify hyperparameters
    test_method = "oe_test" if args.oe_test else "new_class_test"
    if args.baseline:
        if args.margin_baseline:
            project_name = "Single_TN_Masters_Proj_margin_Baseline_Val_{}".format(test_method)
        else:
            project_name = "Single_TN_Masters_Proj_Baseline_Val_{}".format(test_method)
    else:
        project_name = "Single_TN_Masters_Proj_Val_{}_{}_{}".format(args.detection_type, args.loss, test_method)
    
    wandb.init(project=project_name)

    wandb.define_metric("ID_Accuracy", summary="max")
    wandb.define_metric("AUROC", summary="max")

    wandb.config = {
        "learning_rate": args.learning_rate,
        "epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "network": args.architecture,
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

    metric_combined_running_max = 0
    for i in range(0, epochs):
        # start_time = time.time()
        if args.baseline:
            if args.margin_baseline:
                print("margin baseline training starting")
                loss = train_margin(net, lm, id_train_loader, optim, i, id_label_map, device)
            else:
                loss = train_ce(net, id_train_loader, optim, i, id_label_map, device)
        else:
            if args.loss == "margin" and args.detection_type == "LS":
                loss = train_lm_ls(net, lm, args.beta, args.clip, id_train_loader, ood_train_loader, optim, i, id_label_map, device)
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
            if args.margin_baseline:
                print("margin baseline testing start")
                acc, auc = test_lm_ls(net, args.top_k, id_eval_loader, ood_eval_loader, device)
            else:
                acc, auc = test_ce_ls(net, id_eval_loader, ood_eval_loader, device)
        else:
            if args.loss == "margin" and args.detection_type == "LS":
                acc, auc = test_lm_ls(net, args.top_k, id_eval_loader, ood_eval_loader, device)
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

        directory = "val_baseline{}_{}".format("_margin" if args.margin_baseline else "", test_method)
        torch.save({
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'auc': auc,
            'id_accuracy': acc,
            'split': args.split,
        }, 'models/{}/epoch_{}_split_{}.pth'.format(directory,i,split))

        # if args.baseline:
        #     directory = "val_baseline_{}".format(test_method)
        #     torch.save({
        #         'epoch': i,
        #         'model_state_dict': net.state_dict(),
        #         'optimizer_state_dict': optim.state_dict(),
        #         'auc': auc,
        #         'id_accuracy': acc,
        #         'split': args.split,
        #         'test_method': test_method,
        #     }, 'models/{}/day_{}_{}_time_{}_{}_split_{}_epoch_{}.pth'.format(directory, now.month, now.day, now.hour, now.minute, args.split, i))
        # else:
        #     directory = "val_{}_{}_{}".format(args.loss, args.detection_type, test_method)
        #     torch.save({
        #         'epoch': i,
        #         'model_state_dict': net.state_dict(),
        #         'optimizer_state_dict': optim.state_dict(),
        #         'auc': auc,
        #         'id_accuracy': acc,
        #         'split': args.split,
        #         'test_method': test_method,
        #         'loss': args.loss,
        #         'detection_type': args.detection_type,
        #     }, 'models/{}/day_{}_{}_time_{}_{}_split_{}_epoch_{}.pth'.format(directory, now.month, now.day, now.hour, now.minute, args.split, i))

if __name__ == '__main__':
    main()
