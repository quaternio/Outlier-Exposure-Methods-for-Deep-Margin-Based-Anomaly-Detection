import wandb
import torch
import copy
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, confusion_matrix
from large_margin import _get_grad

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
    correct = 0
    total = 0
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

        loss.backward()
        optimizer.step()

        # Computing Accuracy
        _, id_idx = id_logits[:,0:10].max(dim=1)
        correct += (id_idx == id_target).sum().item()
        total += id_data.shape[0]

        if batch_idx % 8 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, 2 * batch_idx * len(id_data), len(id_loader.dataset)+len(ood_loader.dataset),
                100. * 2 * batch_idx / (len(id_loader)+len(ood_loader)), loss.item()))

    return loss, correct / total

def train_lm_ks(model, lm, id_loader, ood_loader, optimizer, epoch, id_label_map, device):
    # For these lm training functions, compute nominal loss and clear features
    # before computing anomaly loss and clearing features. This will ensure 
    model.train()
    num_classes = 11
    correct = 0
    total = 0
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

        # Computing Accuracy
        _, id_idx = output[0:id_data.shape[0],0:10].max(dim=1)
        correct += (id_idx == id_target).sum().item()
        total += id_data.shape[0]
        
        if batch_idx % 8 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, 2 * batch_idx * len(id_data), len(id_loader.dataset)+len(ood_loader.dataset),
                100. * 2 * batch_idx / (len(id_loader)+len(ood_loader)), loss.item()))

    return loss, correct / total


def train_distance(logits, features, top_k, eps, device):
    prob = F.softmax(logits, dim=1)

    max_indices = torch.argmax(prob, dim=1)
    pseudo_correct_prob, _ = torch.max(prob, dim=1, keepdim=True)

    pseudo_other_prob = torch.zeros(prob.shape).to(device)
    pseudo_other_prob.copy_(prob)
    pseudo_other_prob[torch.arange(prob.shape[0]),max_indices] = 0.

    # Grabs the next most likely class probabilities
    if top_k > 1:
        topk_prob, _ = pseudo_other_prob.topk(top_k, dim=1)
    else:
        topk_prob, _ = pseudo_other_prob.max(dim=1, keepdim=True)

    pseudo_diff_prob = pseudo_correct_prob - topk_prob

    layerwise_discriminant_output = []

    for i, feature_map in enumerate(features):
        diff_grad = torch.stack([_get_grad(pseudo_diff_prob[:, i], feature_map) for i in range(top_k)],
                            dim=1)
        diff_gradnorm = torch.norm(diff_grad, p=2, dim=2)
        diff_gradnorm.detach_()
        distance = pseudo_diff_prob / (diff_gradnorm + eps)

        layerwise_discriminant_output.append(distance)

    return torch.hstack(layerwise_discriminant_output)


def min_diff_train_distance(logits, features, top_k, eps, device):
    prob = F.softmax(logits, dim=1)

    max_indices = torch.argmax(prob, dim=1)
    pseudo_correct_prob, _ = torch.max(prob, dim=1, keepdim=True)

    pseudo_other_prob = torch.zeros(prob.shape).to(device)
    pseudo_other_prob.copy_(prob)
    pseudo_other_prob[torch.arange(prob.shape[0]),max_indices] = 0.

    # Grabs the least likely class probabilities
    topk_prob, _ = pseudo_other_prob.min(dim=1, keepdim=True)

    pseudo_diff_prob = pseudo_correct_prob - topk_prob

    layerwise_discriminant_output = []

    for i, feature_map in enumerate(features):
        diff_grad = torch.stack([_get_grad(pseudo_diff_prob[:, i], feature_map) for i in range(top_k)],
                            dim=1)
        diff_gradnorm = torch.norm(diff_grad, p=2, dim=2)
        diff_gradnorm.detach_()
        distance = pseudo_diff_prob / (diff_gradnorm + eps)

        layerwise_discriminant_output.append(distance)

    return torch.hstack(layerwise_discriminant_output)


def train_lm_ls(model, lm, top_k, eps, id_loader, ood_loader, optimizer, epoch, id_label_map, device):
    # For these lm training functions, compute nominal loss and clear features
    # before computing anomaly loss and clearing features. This will ensure 

    model.train()
    num_classes = 10
    for batch_idx, ((id_data, id_target), (ood_data, ood_target)) in enumerate(zip(id_loader, ood_loader)):
        id_data, ood_data = id_data.to(device), ood_data.to(device)
        id_one_hot = torch.zeros(len(id_target), num_classes).scatter_(1, id_target.unsqueeze(1), 1.).float()
        id_one_hot= id_one_hot.to(device)
        optimizer.zero_grad()
        model.clear_features()
        id_output, id_features = model(id_data)
        for feature in id_features:
            feature.retain_grad()

        # Batch-Averaged
        id_margin_loss = lm(id_output, id_one_hot, id_features)

        # Compute OOD output and features
        model.clear_features()
        ood_output, ood_features = model(ood_data)
        for feature in ood_features:
            feature.retain_grad()

        # Note that this distances tensor will have dimension (num_ood_examples x (number of layers*top_k)).
        distances = train_distance(ood_output, ood_features, top_k, eps, device)
        assert(distances.shape[0]==ood_data.shape[0])

        squared_distances = torch.square(distances)

        # Batch-averaged square distance
        ood_ls_loss = (1./distances.shape[0]) * torch.sum(squared_distances)

        loss = id_margin_loss + ood_ls_loss

        loss.backward()
        optimizer.step()
        
        if batch_idx % 8 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, 2 * batch_idx * len(id_data), len(id_loader.dataset)+len(ood_loader.dataset),
                100. * 2 * batch_idx / (len(id_loader)+len(ood_loader)), loss.item()))

    return loss


############
# Testing  #
############

def test_distance(logits, features, top_k, device):
    eps = 1e-8
    prob = F.softmax(logits, dim=1)

    max_indices = torch.argmax(prob, dim=1)
    pseudo_correct_prob, _ = torch.max(prob, dim=1, keepdim=True)

    pseudo_other_prob = torch.zeros(prob.shape).to(device)
    pseudo_other_prob.copy_(prob)
    pseudo_other_prob[torch.arange(prob.shape[0]),max_indices] = 0.

    # Grabs the next most likely class probabilities
    if top_k > 1:
        topk_prob, _ = pseudo_other_prob.topk(top_k, dim=1)
    else:
        topk_prob, _ = pseudo_other_prob.max(dim=1, keepdim=True)

    pseudo_diff_prob = pseudo_correct_prob - topk_prob

    for i, feature_map in enumerate(features):
        if i == len(features)-1:
            diff_grad = torch.stack([_get_grad(pseudo_diff_prob[:, i], feature_map) for i in range(top_k)],
                                dim=1)
            diff_gradnorm = torch.norm(diff_grad, p=2, dim=2)
            diff_gradnorm.detach_()
            distance = pseudo_diff_prob / (diff_gradnorm + eps)

    return distance

def min_diff_test_distance(logits, features, top_k, device):
    eps = 1e-8
    prob = F.softmax(logits, dim=1)

    max_indices = torch.argmax(prob, dim=1)
    pseudo_correct_prob, _ = torch.max(prob, dim=1, keepdim=True)

    pseudo_other_prob = torch.zeros(prob.shape).to(device)
    pseudo_other_prob.copy_(prob)
    pseudo_other_prob[torch.arange(prob.shape[0]),max_indices] = 0.

    # Grabs the next least likely class probability
    topk_prob, _ = pseudo_other_prob.min(dim=1, keepdim=True)

    pseudo_diff_prob = pseudo_correct_prob - topk_prob

    for i, feature_map in enumerate(features):
        if i == len(features)-1:
            diff_grad = torch.stack([_get_grad(pseudo_diff_prob[:, i], feature_map) for i in range(top_k)],
                                dim=1)
            diff_gradnorm = torch.norm(diff_grad, p=2, dim=2)
            diff_gradnorm.detach_()
            distance = pseudo_diff_prob / (diff_gradnorm + eps)

    return distance


def test_lm_ls(model, top_k, id_loader, ood_loader, device):
    model.eval()
    correct = 0
    anomaly_index = 10
    num_classes = 10
    anom_pred = []
    anom_labels = []
    margin_anom_score_sequence = []
    max_logit_anom_score_sequence = []
    pred_sequence = []
    target_sequence = []
    
    for batch_idx, (id_data, id_target) in enumerate(id_loader):
        id_one_hot = torch.zeros(len(id_target), num_classes).scatter_(1, id_target.unsqueeze(1), 1.).float()
        id_one_hot = id_one_hot.to(device)
        id_data, id_target   = id_data.to(device), id_target.to(device)
        model.clear_features()
        id_data = id_data.to(device)
        id_output, id_features  = model(id_data)
        for id_feature in id_features:
            id_feature.retain_grad()

        ###########################
        # ID Distance Computation #
        ###########################
        raw_id_distance = test_distance(id_output, id_features, top_k, device)
        #print("id distances: {}".format(raw_id_distance))
        id_distance = torch.abs(raw_id_distance)

        # Compute number of correctly classified id instances
        id_pred   = id_output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        _, id_idx = id_output.max(dim=1)
        correct += (id_idx == id_target).sum().item()
        id_anom_pred = [1. if id_pred[i] == anomaly_index else 0. for i in range(len(id_pred))]

        # Concatenate the list (order matters here)
        id_batch_anom_pred = id_anom_pred
        anom_pred = anom_pred + id_batch_anom_pred
        pred_sequence.append(id_pred)
        target_sequence.append(id_target)

        # Compute anomaly scores
        # Use discriminant function to compute id_scores
        # id_distance = torch.abs(id_distance)
        assert(id_distance.shape[1]==top_k)
        margin_id_scores, _ = torch.max(-1 * id_distance, dim=1)
        max_logit_id_scores, _ = torch.max(id_output, dim=1)
        max_logit_id_scores = -1 * max_logit_id_scores

        # Detaching is important here because it removes these scores from the computational graph
        margin_anom_score_sequence.append(margin_id_scores.detach().cpu())
        max_logit_anom_score_sequence.append(max_logit_id_scores.detach().cpu())
        for i in range(len(id_target)):
            # 0 indicates "nominal"
            anom_labels.append(0.)

    for batch_idx, (ood_data, ood_target) in enumerate(ood_loader):
        ood_target = anomaly_index * torch.ones_like(ood_target)
        # ood_one_hot = (1/id_one_hot.shape[1])*torch.ones((len(ood_target), id_one_hot.shape[1])).to(device)
        # ood_one_hot = ood_one_hot.to(device)
        ood_data, ood_target = ood_data.to(device), ood_target.to(device)
        model.clear_features()
        ood_output, ood_features = model(ood_data)
        for ood_feature in ood_features:
            ood_feature.retain_grad()

        ############################
        # OOD Distance Computation #
        ############################
        raw_ood_distance = test_distance(ood_output, ood_features, top_k, device)
        #print("raw ood distances: {}".format(raw_ood_distance))
        ood_distance = torch.abs(raw_ood_distance)

        ood_pred   = ood_output.argmax(dim=1, keepdim=True)
        ood_anom_pred = [1. if ood_pred[i] == anomaly_index else 0. for i in range(len(ood_pred))]
        
        # Concatenate the list (order matters here)
        ood_batch_anom_pred = ood_anom_pred
        anom_pred = anom_pred + ood_batch_anom_pred
        pred_sequence.append(ood_pred)
        target_sequence.append(ood_target)

        margin_ood_scores, _ = torch.max(-1 * ood_distance, dim=1)
        max_logit_ood_scores, _ = torch.max(ood_output, dim=1)
        max_logit_ood_scores = -1 * max_logit_ood_scores

        # Detaching is important here because it removes these scores from computational graph
        margin_anom_score_sequence.append(margin_ood_scores.detach().cpu())
        max_logit_anom_score_sequence.append(max_logit_ood_scores.detach().cpu())
        for i in range(len(ood_target)):
            # 1 indicates "anomaly"
            anom_labels.append(1.)

    margin_anom_scores = torch.hstack(margin_anom_score_sequence).cpu().detach().numpy()
    max_logit_anom_scores = torch.hstack(max_logit_anom_score_sequence).cpu().detach().numpy()
    anom_labels = np.asarray(anom_labels)
    anom_pred = np.asarray(anom_pred)
    pred = torch.vstack(pred_sequence).cpu().detach().numpy()
    pred = np.ndarray.flatten(pred)
    targets = torch.hstack(target_sequence).cpu().detach().numpy()

    margin_AUROC    = roc_auc_score(anom_labels, margin_anom_scores)
    max_logit_AUROC = roc_auc_score(anom_labels, max_logit_anom_scores) 

    accuracy = 100. * correct / len(id_loader.dataset)
    print('Test set: Accuracy: {}/{} ({:.0f}%)'.format(
        correct, len(id_loader.dataset), accuracy))
    print('Test Set: margin AUROC: {}\n'.format(margin_AUROC))
    print('Test Set: max logit AUROC: {}\n'.format(max_logit_AUROC))

    return accuracy, margin_AUROC, max_logit_AUROC


# TODO: Verify Done
# This works as a test function for our baseline
def test_ce_ls(model, id_loader, ood_loader, device):
    model.eval()
    correct = 0
    anom_labels = []
    anom_score_sequence = []
    with torch.no_grad(): 
        for batch_idx, (id_data, id_target) in enumerate(id_loader):
            id_data, id_target   = id_data.to(device), id_target.to(device)
            id_output, _  = model(id_data)
            
            # Compute number of correctly classified id instances
            id_pred = id_output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            _, id_idx = id_output.max(dim=1)
            correct += (id_idx == id_target).sum().item()

            # Compute anomaly scores
            pos_id_scores, _ = torch.max(id_output, dim=1)
            id_scores = -1 * pos_id_scores
            anom_score_sequence.append(id_scores)
            for i in range(len(id_target)):
                # 0 indicates "nominal"
                anom_labels.append(0.)

        for batch_idx, (ood_data, ood_target) in enumerate(ood_loader):
            ood_data, ood_target = ood_data.to(device), ood_target.to(device)
            ood_output, _ = model(ood_data)
            
            # Compute anomaly scores
            pos_ood_scores, _ = torch.max(ood_output, dim=1)
            ood_scores = -1 * pos_ood_scores
            anom_score_sequence.append(ood_scores)
            for i in range(len(ood_target)):
                # 1 indicates "anomaly"
                anom_labels.append(1.)

    anom_scores = torch.hstack(anom_score_sequence).cpu().numpy()
    anom_labels = np.asarray(anom_labels)
    
    AUROC = roc_auc_score(anom_labels, anom_scores)

    accuracy = 100. * correct / len(id_loader.dataset)
    print('Test set: Accuracy: {}/{} ({:.0f}%)'.format(
        correct, len(id_loader.dataset), accuracy))
    print('Test Set: AUROC: {}\n'.format(AUROC))

    return accuracy, AUROC


# TODO: Verify Done
def test_ce_ks(model, id_loader, ood_loader, device):
    # For KS, do confusion matrix
    model.eval()
    correct = 0
    anomaly_index = 10
    anom_pred = []
    anom_labels = []
    ks_logit_anom_score_sequence = []
    max_id_logit_anom_score_sequence = []
    pred_sequence = []
    target_sequence = []
    with torch.no_grad(): 
        for batch_idx, (id_data, id_target) in enumerate(id_loader):
            id_data, id_target   = id_data.to(device), id_target.to(device)
            id_output, _  = model(id_data)

            # Compute number of correctly classified id instances
            id_pred   = id_output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            id_anom_pred = [1. if id_pred[i] == anomaly_index else 0. for i in range(len(id_pred))]

            # Computing Accuracy
            _, id_idx = id_output[:,0:anomaly_index].max(dim=1)
            correct += (id_idx == id_target).sum().item()
            
            # Concatenate the list (order matters here)
            id_batch_anom_pred = id_anom_pred
            anom_pred = anom_pred + id_batch_anom_pred

            pred_sequence.append(id_pred)
            target_sequence.append(id_target)

            # Compute anomaly scores
            ks_logit_id_scores = id_output[:,anomaly_index]
            ks_logit_anom_score_sequence.append(ks_logit_id_scores)

            pos_max_id_logit_scores, _ = torch.max(id_output[:,0:anomaly_index], dim=1)
            max_id_logit_scores = -1 * pos_max_id_logit_scores
            max_id_logit_anom_score_sequence.append(max_id_logit_scores)

            for i in range(len(id_target)):
                # 0 indicates "nominal"
                anom_labels.append(0.)

        for batch_idx, (ood_data, ood_target) in enumerate(ood_loader):
            ood_target = anomaly_index * torch.ones_like(ood_target)
            ood_data, ood_target = ood_data.to(device), ood_target.to(device)
            ood_output, _ = model(ood_data)
            ood_pred   = ood_output.argmax(dim=1, keepdim=True)
            ood_anom_pred = [1. if ood_pred[i] == anomaly_index else 0. for i in range(len(ood_pred))]
            ood_batch_anom_pred = ood_anom_pred
            anom_pred = anom_pred + ood_batch_anom_pred
            pred_sequence.append(ood_pred)
            target_sequence.append(ood_target)
            
            # Compute anomaly scores
            ks_logit_ood_scores = ood_output[:,anomaly_index]
            ks_logit_anom_score_sequence.append(ks_logit_ood_scores)

            pos_max_id_logit_scores, _ = torch.max(ood_output[:,0:anomaly_index], dim=1)
            max_id_logit_scores = -1 * pos_max_id_logit_scores
            max_id_logit_anom_score_sequence.append(max_id_logit_scores)

            for i in range(len(ood_target)):
                # 1 indicates "anomaly"
                anom_labels.append(1.)

    ks_logit_anom_scores = torch.hstack(ks_logit_anom_score_sequence).cpu().numpy()
    max_id_logit_anom_scores = torch.hstack(max_id_logit_anom_score_sequence).cpu().numpy()
    anom_labels = np.asarray(anom_labels)
    anom_pred = np.asarray(anom_pred)
    pred = torch.vstack(pred_sequence).cpu().numpy()
    pred = np.ndarray.flatten(pred)
    targets = torch.hstack(target_sequence).cpu().numpy()

    ks_logit_AUROC = roc_auc_score(anom_labels, ks_logit_anom_scores)
    max_id_logit_AUROC = roc_auc_score(anom_labels, max_id_logit_anom_scores)

    accuracy = 100. * correct / len(id_loader.dataset)
    print('Test set: Accuracy: {}/{} ({:.0f}%)'.format(
        correct, len(id_loader.dataset), accuracy))
    print('Test Set: ks logit AUROC: {}\n'.format(ks_logit_AUROC))
    print('Test Set: max id logit AUROC: {}\n'.format(max_id_logit_AUROC))

    return accuracy, ks_logit_AUROC, max_id_logit_AUROC


# TODO: Verify Correct
def test_lm_ks(model, id_loader, ood_loader, device):
    model.eval()
    correct = 0
    anomaly_index = 10
    anom_pred = []
    anom_labels = []
    ks_logit_anom_score_sequence = []
    max_id_logit_anom_score_sequence = []
    pred_sequence = []
    target_sequence = []
    with torch.no_grad(): 
        for batch_idx, (id_data, id_target) in enumerate(id_loader):
            id_data, id_target   = id_data.to(device), id_target.to(device)
            id_output, _  = model(id_data)
            
            # Compute number of correctly classified id instances
            id_pred   = id_output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            _, id_idx = id_output[0:anomaly_index].max(dim=1)
            correct += (id_idx == id_target).sum().item()
            id_anom_pred = [1. if id_pred[i] == anomaly_index else 0. for i in range(len(id_pred))]

            # Concatenate the list (order matters here)
            batch_anom_pred = id_anom_pred ###ood_anom_pred + id_anom_pred
            anom_pred = anom_pred + batch_anom_pred

            pred_sequence.append(id_pred)
            target_sequence.append(id_target)

            # Compute Anom. Scores
            ks_logit_id_scores = id_output[:,anomaly_index]
            ks_logit_anom_score_sequence.append(ks_logit_id_scores)

            pos_max_id_logit_scores, _ = torch.max(id_output[:,0:anomaly_index], dim=1)
            max_id_logit_scores = -1 * pos_max_id_logit_scores
            max_id_logit_anom_score_sequence.append(max_id_logit_scores)

            for i in range(len(id_target)):
                # 0 indicates "nominal"
                anom_labels.append(0.)

        for batch_idx, (ood_data, ood_target) in enumerate(ood_loader):
            ood_target = anomaly_index * torch.ones_like(ood_target)
            ood_data, ood_target = ood_data.to(device), ood_target.to(device)
            ood_output, _ = model(ood_data)

            ood_pred   = ood_output.argmax(dim=1, keepdim=True)
            ood_anom_pred = [1. if ood_pred[i] == anomaly_index else 0. for i in range(len(ood_pred))]
            pred_sequence.append(ood_pred)
            target_sequence.append(ood_target)

            # Compute anomaly scores
            ks_logit_ood_scores = ood_output[:,anomaly_index]
            ks_logit_anom_score_sequence.append(ks_logit_ood_scores)

            pos_max_id_logit_scores, _ = torch.max(ood_output[:,0:anomaly_index], dim=1)
            max_id_logit_scores = -1 * pos_max_id_logit_scores
            max_id_logit_anom_score_sequence.append(max_id_logit_scores)

            for i in range(len(ood_target)):
                # 1 indicates "anomaly"
                anom_labels.append(1.)

    ks_logit_anom_scores = torch.hstack(ks_logit_anom_score_sequence).cpu().numpy()
    max_id_logit_anom_scores = torch.hstack(max_id_logit_anom_score_sequence).cpu().numpy()
    anom_labels = np.asarray(anom_labels)
    anom_pred = np.asarray(anom_pred)
    pred = torch.vstack(pred_sequence).cpu().numpy()
    pred = np.ndarray.flatten(pred)
    targets = torch.hstack(target_sequence).cpu().numpy()

    ks_logit_AUROC = roc_auc_score(anom_labels, ks_logit_anom_scores)
    max_id_logit_AUROC = roc_auc_score(anom_labels, max_id_logit_anom_scores)

    accuracy = 100. * correct / len(id_loader.dataset)
    print('Test set: Accuracy: {}/{} ({:.0f}%)'.format(
        correct, len(id_loader.dataset), accuracy))
    print('Test Set: ks logit AUROC: {}\n'.format(ks_logit_AUROC))
    print('Test Set: max id logit AUROC: {}\n'.format(max_id_logit_AUROC))

    return accuracy, ks_logit_AUROC, max_id_logit_AUROC