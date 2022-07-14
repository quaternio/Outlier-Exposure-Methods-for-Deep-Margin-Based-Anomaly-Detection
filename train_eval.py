import wandb
import torch
import torch.nn.functional as F

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


def train_ce(model, train_loader, optimizer, epoch, id_label_map, device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, _ = model(data)
        model.clear_features()

        loss = F.cross_entropy(output, target)
        
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


# TODO: Implement training and testing for all four cases
#           1) Cross entropy Loss and Kitchen Sink Training
#           2) Cross entropy Loss and Logit Suppression
#           3) Margin Loss and Kitchen Sink Training
#           4) Margin Loss and Logit Suppression

def train_ce_ls(model, id_loader, ood_loader, optimizer, epoch, id_label_map, device):
    model.train()
    m = torch.nn.Softmax(dim=1)
    for batch_idx, ((id_data, id_target), (ood_data, ood_target)) in enumerate(zip(id_loader, ood_loader)):
        ood_target = 10 * torch.ones_like(ood_target)
        id_data, id_target   = id_data.to(device), id_target.to(device)
        ood_data, ood_target = ood_data.to(device), ood_target.to(device)
        optimizer.zero_grad()
        id_logits, _  = model(id_data)
        ood_logits, _ = model(ood_data)
        model.clear_features()

        # TODO: Pick up here to finish implementing CE + logit suppression. 
        # Use logit_supp.py for reference on how to finish this task.

        ##ood_target = (1/1)

        loss = F.cross_entropy(id_logits, id_target) + F.cross_entropy(ood_logits, ood_target)
        
        # Logging
        wandb.log({"loss": loss})
        wandb.watch(model)

        loss.backward()
        optimizer.step()
        # if batch_idx % 100 == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(loader.dataset),
        #         100. * batch_idx / len(loader), loss.item()))


def train_ce_ks(model, loader, optimizer, epoch, id_label_map, device):
    pass

def train_lm_ks(model, loader, optimizer, epoch, id_label_map, device):
    # For these lm training functions, compute nominal loss and clear features
    # before computing anomaly loss and clearing features. This will ensure 
    pass

def train_lm_ls(model, loader, optimizer, epoch, id_label_map, device):
    pass

