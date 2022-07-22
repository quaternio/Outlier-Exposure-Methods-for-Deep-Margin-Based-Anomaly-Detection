import wandb
import torch
import copy
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, confusion_matrix

def train_lm(model, train_loader, optimizer, epoch, lm, num_classes, id_label_map, device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        one_hot = torch.zeros(len(target), num_classes).scatter_(1, target.unsqueeze(1), 1.).float()
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

############
# Baseline #
############
def train_ce(model, train_loader, optimizer, epoch, id_label_map, device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, _ = model(data)
        model.clear_features()

        loss = F.cross_entropy(output, target)

        # Logging
        # wandb.log({"loss": loss})
        # wandb.watch(model)
        
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    return loss

#############################################
# Anomaly Scoring and Detection Performance #
#############################################

def compute_auc(scores, labels):
    pass

############
# Training #
############

# TODO: Implement training and testing for all four cases
#           1) Cross entropy Loss and Kitchen Sink Training (DONE)
#           2) Cross entropy Loss and Logit Suppression (DONE)
#           3) Margin Loss and Kitchen Sink Training (DONE)
#           4) Margin Loss and Logit Suppression ()

def train_ce_ls(model, id_loader, ood_loader, optimizer, epoch, id_label_map, device):
    model.train()
    for batch_idx, ((id_data, id_target), (ood_data, ood_target)) in enumerate(zip(id_loader, ood_loader)):
        id_data, id_target  = id_data.to(device), id_target.to(device)
        ood_data, ood_target = ood_data.to(device), ood_target.to(device)
        optimizer.zero_grad()
        id_logits, _  = model(id_data)
        ood_logits, _ = model(ood_data)
        id_logits, ood_logits = id_logits.to(device), ood_logits.to(device)
        model.clear_features()

        ood_target = (1/id_logits.shape[1])*torch.ones((len(ood_target), id_logits.shape[1])).to(device)
        id_target.to(device)

        loss = F.cross_entropy(id_logits, id_target) + F.cross_entropy(ood_logits, ood_target)
        
        # Logging
        #wandb.log({"loss": loss})
        #wandb.watch(model)

        loss.backward()
        optimizer.step()
        if batch_idx % 8 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, 2 * batch_idx * len(id_data), len(id_loader.dataset)+len(ood_loader.dataset),
                100. * 2 * batch_idx / (len(id_loader)+len(ood_loader)), loss.item()))

    return loss

def train_ce_ks(model, id_loader, ood_loader, optimizer, epoch, id_label_map, device):
    model.train()
    for batch_idx, ((id_data, id_target), (ood_data, ood_target)) in enumerate(zip(id_loader, ood_loader)):
        ood_target = 10 * torch.ones_like(ood_target)
        id_data, id_target  = id_data.to(device), id_target.to(device)
        ood_data, ood_target = ood_data.to(device), ood_target.to(device)
        optimizer.zero_grad()
        id_logits, _  = model(id_data)
        ood_logits, _ = model(ood_data)
        id_logits, ood_logits = id_logits.to(device), ood_logits.to(device)
        model.clear_features()

        loss = F.cross_entropy(id_logits, id_target) + F.cross_entropy(ood_logits, ood_target)
        
        # Logging
        # wandb.log({"loss": loss})
        # wandb.watch(model)

        loss.backward()
        optimizer.step()
        if batch_idx % 8 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, 2 * batch_idx * len(id_data), len(id_loader.dataset)+len(ood_loader.dataset),
                100. * 2 * batch_idx / (len(id_loader)+len(ood_loader)), loss.item()))

    return loss

def train_lm_ks(model, lm, id_loader, ood_loader, optimizer, epoch, id_label_map, device):
    # For these lm training functions, compute nominal loss and clear features
    # before computing anomaly loss and clearing features. This will ensure 
    model.train()
    num_classes = 11
    for batch_idx, ((id_data, id_target), (ood_data, ood_target)) in enumerate(zip(id_loader, ood_loader)):
        ood_target = 10 * torch.ones_like(ood_target)
        data = torch.vstack((id_data, ood_data))
        data = data.to(device)
        id_one_hot = torch.zeros(len(id_target), num_classes).scatter_(1, id_target.unsqueeze(1), 1.).float()
        ood_one_hot = torch.zeros(len(ood_target), num_classes).scatter_(1, ood_target.unsqueeze(1), 1.).float()
        one_hot = torch.vstack((id_one_hot, ood_one_hot))
        one_hot = one_hot.cuda()
        optimizer.zero_grad()
        model.clear_features()
        output, features = model(data)
        for feature in features:
            feature.retain_grad()

        loss = lm(output, one_hot, features)

        #wandb.log({"loss": loss})
        #wandb.watch(model)

        loss.backward()
        optimizer.step()
        
        if batch_idx % 8 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, 2 * batch_idx * len(id_data), len(id_loader.dataset)+len(ood_loader.dataset),
                100. * 2 * batch_idx / (len(id_loader)+len(ood_loader)), loss.item()))

    return loss

def train_lm_ls(model, lm, id_loader, ood_loader, optimizer, epoch, id_label_map, device):
    # For these lm training functions, compute nominal loss and clear features
    # before computing anomaly loss and clearing features. This will ensure 
    model.train()
    num_classes = 10
    for batch_idx, ((id_data, id_target), (ood_data, ood_target)) in enumerate(zip(id_loader, ood_loader)):
        data = torch.vstack((id_data, ood_data))
        data = data.to(device)
        id_one_hot = torch.zeros(len(id_target), num_classes).scatter_(1, id_target.unsqueeze(1), 1.).float()
        ood_one_hot = (1/id_one_hot.shape[1])*torch.ones((len(ood_target), id_one_hot.shape[1])).to(device)
        id_one_hot, ood_one_hot = id_one_hot.to(device), ood_one_hot.to(device)
        one_hot = torch.vstack((id_one_hot, ood_one_hot))
        one_hot = one_hot.cuda()
        optimizer.zero_grad()
        model.clear_features()
        output, features = model(data)
        for feature in features:
            feature.retain_grad()

        loss = lm(output, one_hot, features)

        #wandb.log({"loss": loss})
        #wandb.watch(model)

        loss.backward()
        optimizer.step()
        
        if batch_idx % 8 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, 2 * batch_idx * len(id_data), len(id_loader.dataset)+len(ood_loader.dataset),
                100. * 2 * batch_idx / (len(id_loader)+len(ood_loader)), loss.item()))

    return loss

# def train_lm_ls_experimental(model, lm, id_loader, ood_loader, optimizer, epoch, id_label_map, device):
#     # Suppresses discriminants instead of logits
#     model.train()
#     num_classes = 10
#     for batch_idx, ((id_data, id_target), (ood_data, ood_target)) in enumerate(zip(id_loader, ood_loader)):
#         data = torch.vstack((id_data, ood_data))
#         data = data.to(device)
#         id_one_hot = torch.zeros(len(id_target), num_classes).scatter_(1, id_target.unsqueeze(1), 1.).float()
#         ood_one_hot = (1/id_one_hot.shape[1])*torch.ones((len(ood_target), id_one_hot.shape[1])).to(device)
#         id_one_hot, ood_one_hot = id_one_hot.to(device), ood_one_hot.to(device)
#         one_hot = torch.vstack((id_one_hot, ood_one_hot))
#         one_hot = one_hot.cuda()
#         optimizer.zero_grad()
#         model.clear_features()
#         output, features = model(data)
#         for feature in features:
#             feature.retain_grad()

#         loss = lm(output, one_hot, features)

#         wandb.log({"loss": loss})
#         wandb.watch(model)

#         loss.backward()
#         optimizer.step()
        
#         if batch_idx % 8 == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, 2 * batch_idx * len(id_data), len(id_loader.dataset)+len(ood_loader.dataset),
#                 100. * 2 * batch_idx / (len(id_loader)+len(ood_loader)), loss.item()))



############
# Testing  #
############

# def test_ce(model, test_loader):
#     model.eval()
#     correct = 0
#     with torch.no_grad(): 
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output, *_ = model(data)
#             pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
#             _, idx = output.max(dim=1)
#             correct += (idx == target).sum().item()

#             # Clear the computed features
#             model.clear_features()

#     accuracy = 100. * correct / len(test_loader.dataset)
#     print('Test set: Accuracy: {}/{} ({:.0f}%)'.format(
#         correct, len(test_loader.dataset), accuracy))
#     print('Test Set: AUROC: {}\n'.format(AUROC))

#     wandb.log({"accuracy": accuracy})

# This works as a test function for our baseline
def test_ce_ls(model, id_loader, ood_loader, device):
    model.eval()
    correct = 0
    anom_labels = []
    anom_score_sequence = []
    with torch.no_grad(): 
        for batch_idx, ((id_data, id_target), (ood_data, ood_target)) in enumerate(zip(id_loader, ood_loader)):
            id_data, id_target   = id_data.to(device), id_target.to(device)
            ood_data, ood_target = ood_data.to(device), ood_target.to(device)
            id_output, _  = model(id_data)
            ood_output, _ = model(ood_data)

            # Compute number of correctly classified id instances
            id_pred = id_output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            _, id_idx = id_output.max(dim=1)
            correct += (id_idx == id_target).sum().item()

            # Compute anomaly scores
            pos_ood_scores, _ = torch.max(ood_output, dim=1)
            ood_scores = -1 * pos_ood_scores
            anom_score_sequence.append(ood_scores)
            for i in range(len(ood_target)):
                # 1 indicates "anomaly"
                anom_labels.append(1.)

            pos_id_scores, _ = torch.max(id_output, dim=1)
            id_scores = -1 * pos_id_scores
            anom_score_sequence.append(id_scores)
            for i in range(len(id_target)):
                # 0 indicates "nominal"
                anom_labels.append(0.)

    anom_scores = torch.hstack(anom_score_sequence).cpu().numpy()
    anom_labels = np.asarray(anom_labels)
    
    AUROC = roc_auc_score(anom_labels, anom_scores)

    accuracy = 100. * correct / len(id_loader.dataset)
    print('Test set: Accuracy: {}/{} ({:.0f}%)'.format(
        correct, len(id_loader.dataset), accuracy))
    print('Test Set: AUROC: {}\n'.format(AUROC))

    return accuracy, AUROC


def test_ce_ks(model, epoch, id_loader, ood_loader, device):
    # For KS, do confusion matrix
    model.eval()
    correct = 0
    anomaly_index = 10
    anom_pred = []
    anom_labels = []
    anom_score_sequence = []
    pred_sequence = []
    target_sequence = []
    with torch.no_grad(): 
        for batch_idx, ((id_data, id_target), (ood_data, ood_target)) in enumerate(zip(id_loader, ood_loader)):
            ood_target = anomaly_index * torch.ones_like(ood_target)
            id_data, id_target   = id_data.to(device), id_target.to(device)
            ood_data, ood_target = ood_data.to(device), ood_target.to(device)
            id_output, _  = model(id_data)
            ood_output, _ = model(ood_data)

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
            ood_scores = ood_output[:,anomaly_index]
            anom_score_sequence.append(ood_scores)
            for i in range(len(ood_target)):
                # 1 indicates "anomaly"
                anom_labels.append(1.)

            id_scores = id_output[:,anomaly_index]
            anom_score_sequence.append(id_scores)
            for i in range(len(id_target)):
                # 0 indicates "nominal"
                anom_labels.append(0.)

    anom_scores = torch.hstack(anom_score_sequence).cpu().numpy()
    anom_labels = np.asarray(anom_labels)
    anom_pred = np.asarray(anom_pred)
    pred = torch.vstack(pred_sequence).cpu().numpy()
    pred = np.ndarray.flatten(pred)
    targets = torch.hstack(target_sequence).cpu().numpy()

    skl_conf_matrix = confusion_matrix(anom_labels, anom_pred)
    tn_count = skl_conf_matrix[0,0]
    fp_count = skl_conf_matrix[0,1]
    fn_count = skl_conf_matrix[1,0]
    tp_count = skl_conf_matrix[1,1]
    wandb.log({"Eval True Negatives per Epoch": tn_count}, step=epoch)
    wandb.log({"Eval False Positives per Epoch": fp_count}, step=epoch)
    wandb.log({"Eval False Negatives per Epoch": fn_count}, step=epoch)
    wandb.log({"Eval True Positives per Epoch": tp_count}, step=epoch)
    detection_conf_matrix = wandb.plot.confusion_matrix(y_true=anom_labels, preds=anom_pred)
    wandb.log({"Detection Confusion Matrix": detection_conf_matrix}, step=epoch)

    conf_matrix = wandb.plot.confusion_matrix(y_true=targets, preds=pred)
    wandb.log({"Confusion Matrix": conf_matrix}, step=epoch)

    AUROC = roc_auc_score(anom_labels, anom_scores)

    accuracy = 100. * correct / len(id_loader.dataset)
    print('Test set: Accuracy: {}/{} ({:.0f}%)'.format(
        correct, len(id_loader.dataset), accuracy))
    print('Test Set: AUROC: {}\n'.format(AUROC))

    return accuracy, AUROC


def test_lm_ls(model, lm, id_loader, ood_loader, device):
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
        #id_features = copy.deepcopy(id_features)
        lm(id_output, id_one_hot, id_features)
        raw_id_distance = lm.get_discriminant()
        id_distance = torch.abs(raw_id_distance)
        # print("id distance: {}".format(id_distance))

        model.clear_features()
        ood_output, ood_features = model(ood_data)
        for ood_feature in ood_features:
            ood_feature.retain_grad()
        #ood_features = copy.deepcopy(ood_features)
        lm(ood_output, ood_one_hot, ood_features)
        raw_ood_distance = lm.get_discriminant()
        ood_distance = torch.abs(raw_ood_distance)
        # print("id distance (should be same as above): {}".format(id_distance))
        # print("ood distance: {}".format(ood_distance))

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

    AUROC = roc_auc_score(anom_labels, anom_scores)

    accuracy = 100. * correct / len(id_loader.dataset)
    print('Test set: Accuracy: {}/{} ({:.0f}%)'.format(
        correct, len(id_loader.dataset), accuracy))
    print('Test Set: AUROC: {}\n'.format(AUROC))

    return accuracy, AUROC


def test_lm_ks(model, epoch, id_loader, ood_loader, device):
    model.eval()
    correct = 0
    anomaly_index = 10
    anom_pred = []
    anom_labels = []
    anom_score_sequence = []
    pred_sequence = []
    target_sequence = []
    with torch.no_grad(): 
        for batch_idx, ((id_data, id_target), (ood_data, ood_target)) in enumerate(zip(id_loader, ood_loader)):
            ood_target = anomaly_index * torch.ones_like(ood_target)
            id_data, id_target   = id_data.to(device), id_target.to(device)
            ood_data, ood_target = ood_data.to(device), ood_target.to(device)
            id_output, _  = model(id_data)
            ood_output, _ = model(ood_data)

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
            ood_scores = ood_output[:,anomaly_index]
            anom_score_sequence.append(ood_scores)
            for i in range(len(ood_target)):
                # 1 indicates "anomaly"
                anom_labels.append(1.)

            id_scores = id_output[:,anomaly_index]
            anom_score_sequence.append(id_scores)
            for i in range(len(id_target)):
                # 0 indicates "nominal"
                anom_labels.append(0.)

    anom_scores = torch.hstack(anom_score_sequence).cpu().numpy()
    anom_labels = np.asarray(anom_labels)
    anom_pred = np.asarray(anom_pred)
    pred = torch.vstack(pred_sequence).cpu().numpy()
    pred = np.ndarray.flatten(pred)
    targets = torch.hstack(target_sequence).cpu().numpy()

    names = ["nominal", "anomaly"]
    skl_conf_matrix = confusion_matrix(anom_labels, anom_pred)
    tn_count = skl_conf_matrix[0,0]
    fp_count = skl_conf_matrix[0,1]
    fn_count = skl_conf_matrix[1,0]
    tp_count = skl_conf_matrix[1,1]
    wandb.log({"Eval True Negatives per Epoch": tn_count}, step=epoch)
    wandb.log({"Eval False Positives per Epoch": fp_count}, step=epoch)
    wandb.log({"Eval False Negatives per Epoch": fn_count}, step=epoch)
    wandb.log({"Eval True Positives per Epoch": tp_count}, step=epoch)
    detection_conf_matrix = wandb.plot.confusion_matrix(y_true=anom_labels, preds=anom_pred, class_names=names)
    wandb.log({"Detection Confusion Matrix": detection_conf_matrix}, step=epoch)

    conf_matrix = wandb.plot.confusion_matrix(y_true=targets, preds=pred)
    wandb.log({"Confusion Matrix": conf_matrix}, step=epoch)

    AUROC = roc_auc_score(anom_labels, anom_scores)

    accuracy = 100. * correct / len(id_loader.dataset)
    print('Test set: Accuracy: {}/{} ({:.0f}%)'.format(
        correct, len(id_loader.dataset), accuracy))
    print('Test Set: AUROC: {}\n'.format(AUROC))

    return accuracy, AUROC