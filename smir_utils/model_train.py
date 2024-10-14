'''
model train
'''
from copy import deepcopy
from datetime import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import math
from tqdm import tqdm
from smir.smir_utils.model_utils import get_predictions, get_all_misclf_number

def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('accuracy on test set: %.5f %% ' % (100 * correct / total))
    return correct / total

def weighted_retrain(model, misclf_key, dataloader, seed, weight=20,
                    epoch=None, method=None, model_name=None, batchsize=None,
                     loss_type=None, set_criterion=None, optimizer=None, lr=None, scheduler=None):
    correct = 0
    total = 0
    train_loss = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if method in ['mapping', 'wce']:
        dataloader = DataLoader(dataloader.dataset, shuffle=True, batch_size=batchsize)
    if model_name == 'APTOS2019_CNN':
        dataloader = DataLoader(dataloader.dataset, shuffle=True, batch_size=128)

    if optimizer == None:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        if method == 'retrain':
            None

    for name, param in model.named_parameters():
        param.requires_grad = True

    if method == 'binary_retrain':
        para_list = ['fc']
        if model_name == 'simple_fm':
            para_list = ['fc_v2']
        params_to_update = [param for name, param in model.named_parameters() if
                            not any(x in name for x in para_list)]
        optimizer = optim.SGD(params_to_update, lr=lr, momentum=0.9)

    # Set scheduler
    if scheduler:
        if scheduler.__module__ == lr_scheduler.__name__:
            # UsingPyTorchIn-Builtscheduler
            scheduler.step()
        else:
            # Usingcustomdefinedscheduler
            for param_group in optimizer.param_groups:
                param_group['lr'] = scheduler(epoch)
            print('epoch:{}, lr:{}'.format(epoch, scheduler(epoch)))

    # Set Loss Function
    if set_criterion == None:
        error_weights = {misclf_key: weight}
        if loss_type == 'Focal_Loss':
            criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
        elif loss_type == 'Weight_Focal_Loss':
            criterion = Weighted_FocalLoss(alpha=0.25, gamma=2.0, reduction='mean', error_weights=error_weights)
        elif loss_type == 'Dynamic_Balanced_Loss':
            criterion = dynamic_balance_loss(error_weights=error_weights, k=1, t=1)
        elif loss_type == 'Dynamic_Balance_Focal_Loss':
            criterion = dynamic_balance_focal_loss(error_weights=error_weights, k=1, t=1)
        elif loss_type == 'MSE_Loss':
            criterion = nn.MSELoss()
        else:  # Weighted_Loss
            criterion = WeightedLoss(error_weights)
    else:
        criterion = set_criterion


    model.train()
    for i, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        torch.cuda.empty_cache()
        if loss_type == 'MSE_Loss':
            targets = targets.float()
            loss = criterion(outputs, targets)
        else:
            targets = targets.long()
            loss = criterion(outputs, targets)
        train_loss += loss.item() * targets.size(0)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, dim=1)
        # _, predicted = outputs.max(1)
        total += targets.size(0)
        if loss_type != 'MSE_Loss':
            correct += predicted.eq(targets).sum().item()

    return train_loss, correct

def binary_retrain(model, model_name, misclf_key, dataloader, seed, weight=1,
                   epoch=None, method=None, batchsize=None, loss_type=None,
                   set_criterion=None, lr=None, scheduler=None):
    correct = 0
    total = 0
    train_loss = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    if scheduler == None:
        scheduler = None

    if scheduler:
        if scheduler.__module__ == lr_scheduler.__name__:
            # UsingPyTorchIn-Builtscheduler
            scheduler.step()
        else:
            # Usingcustomdefinedscheduler
            for param_group in optimizer.param_groups:
                param_group['lr'] = scheduler(epoch)
        current_lr = optimizer.param_groups[0]['lr']
        print('epoch:{}, lr:{}'.format(epoch, current_lr))

    # Set Loss Function
    if set_criterion == None:
        error_weights = {(1, 0): weight}
        if loss_type == 'Focal_Loss':
            criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
        elif loss_type == 'Weight_Focal_Loss':
            criterion = Weighted_FocalLoss(alpha=0.25, gamma=2.0, reduction='mean', error_weights=error_weights)
        elif loss_type == 'Dynamic_Balanced_Loss':
            criterion = DynamicBalanceLoss_v3(error_weights=error_weights, k=1, t=1)
        elif loss_type == 'Dynamic_Balance_Focal_Loss':
            criterion = dynamic_balance_focal_loss(error_weights=error_weights, k=1, t=1)
        elif loss_type == 'MSE_Loss':
            criterion = nn.MSELoss()
        else:  # Weighted_Loss
            criterion = WeightedLoss(error_weights)
    else:
        criterion = set_criterion

    model.train()
    for i, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        if method in ['feature_rec', 'feature_rec-alpha', 'feature_rec-beta']:
            outputs = model(inputs)
        else:
            outputs = model.forward_v2(inputs)

        targets = targets.long()
        loss = criterion(outputs, targets)
        train_loss += loss.item() * targets.size(0)  #
        # train_loss += loss.item()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, dim=1)
        # _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return train_loss, correct

class WeightedLoss(torch.nn.Module):
    def __init__(self, error_weights):
        super(WeightedLoss, self).__init__()
        self.error_weights = error_weights

    def forward(self, output, target):
        batch_size = output.size(0)
        loss = F.cross_entropy(output, target, reduction='none')
        for i in range(batch_size):
            true_class = target[i].item()
            predicted_class = output[i].argmax().item()
            if (true_class, predicted_class) in self.error_weights:
                loss[i] *= self.error_weights[(true_class, predicted_class)]
        return loss.mean()

    def __repr__(self):
        return f"WeightedLoss(error_weights={self.error_weights})"

class DynamicBalanceLoss_v3(nn.Module):
    def __init__(self, error_weights={(1, 0): 1}, k=0.5, t=3):

        super(DynamicBalanceLoss_v3, self).__init__()
        self.error_weights = error_weights
        self.k = k
        self.t = t

    def forward(self, output, target):

        batch_size = output.size(0)
        losses = F.cross_entropy(output, target, reduction='none')

        loss1 = 0.0
        loss2 = 0.0
        count1 = 0

        for i in range(batch_size):
            true_class = target[i].item()
            predicted_class = output[i].argmax().item()
            if (true_class, predicted_class) in self.error_weights:
                loss1 += losses[i] * self.error_weights[(true_class, predicted_class)]
                count1 += 1
            else:
                loss2 += losses[i]

        count2 = batch_size - count1
        if count1 > 0:
            loss1 = loss1 / batch_size
        if count2 > 0:
            loss2 = loss2 / batch_size

        if loss1 > 0:
            weight_for_loss2 = 1 / (1 + torch.exp(-self.k * (loss1.detach() - self.t)))
            total_loss = loss1 + weight_for_loss2 * loss2
        else:
            total_loss = loss2 / batch_size

        return total_loss

    def __repr__(self):
        return (f"DynamicBalanceLoss(error_weights={self.error_weights}, "
                f"k={self.k}, t={self.t})")

class dynamic_balance_focal_loss(nn.Module):
    def __init__(self, error_weights={(1, 0): 1}, k=0.5, t=3, alpha=1, gamma=2, reduction='mean'):
        super(dynamic_balance_focal_loss, self).__init__()
        self.error_weights = error_weights
        self.k = k
        self.t = t

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(alpha, float):
            self.alpha = torch.tensor([alpha]).to(self.device)
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.error_weights = error_weights

    def forward(self, output, target):
        batch_size = output.size(0)
        BCE_loss = F.cross_entropy(output, target, reduction='none')
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        loss1 = 0.0
        loss2 = 0.0
        for i in range(batch_size):
            true_class = target[i].item()
            predicted_class = output[i].argmax().item()
            if (true_class, predicted_class) in self.error_weights:
                loss1 += F_loss[i] * self.error_weights[(true_class, predicted_class)]
            else:
                loss2 += F_loss[i]

        if loss1 > 0:
            loss1 = loss1 / batch_size
            loss2 = loss2 / batch_size
            weight_for_loss2 = 1 / (1 + torch.exp(-self.k * (loss1.detach() - self.t)))
            total_loss = loss1 + weight_for_loss2 * loss2
        else:
            total_loss = loss2 / batch_size

        return total_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(alpha, float):
            self.alpha = torch.tensor([alpha]).to(self.device)
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward_binary(self, inputs, targets):
        log_pt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(log_pt)
        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0, targets.data.view(-1))
            log_pt = log_pt * at

        focal_loss = -1 * (1 - pt) ** self.gamma * log_pt
        if self.ignore_index >= 0:
            valid_mask = (targets != self.ignore_index)
            focal_loss = focal_loss[valid_mask]
            targets = targets[valid_mask]
        loss = F.nll_loss(focal_loss, targets, reduction=self.reduction)

        return loss

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

class Weighted_FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean', ignore_index=-100, error_weights=None):
        super(Weighted_FocalLoss, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.alpha = 1
        if isinstance(alpha, float):
            self.alpha = torch.tensor([alpha]).to(self.device)
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.error_weights = error_weights

    def forward(self, outputs, targets):
        BCE_loss = F.cross_entropy(outputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        batch_size = outputs.size(0)
        for i in range(batch_size):
            true_class = targets[i].item()
            predicted_class = outputs[i].argmax().item()
            if (true_class, predicted_class) in self.error_weights:
                F_loss[i] *= self.error_weights[(true_class, predicted_class)]

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
    def __repr__(self):
        return (f"Weighted_FocalLoss(alpha={self.alpha}, gamma={self.gamma}, "
                f"reduction='{self.reduction}', error_weights={self.error_weights})")

class CosineScheduler:
    def __init__(self, max_update, base_lr=0.01, final_lr=0,
               warmup_steps=5, warmup_begin_lr=0):
        self.warmup_begin_lr = warmup_begin_lr
        self.base_lr_orig = base_lr
        self.final_lr = final_lr
        self.max_update = max_update
        self.warmup_steps = warmup_steps
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) \
                       * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (
                self.base_lr_orig - self.final_lr) * (1 + math.cos(
                math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr

class StepScheduler:
    def __init__(self, step_size, gamma=0.1, base_lr=0.01):
        self.step_size = step_size
        self.gamma = gamma
        self.base_lr_orig = base_lr
        self.base_lr = base_lr

    def __call__(self, epoch):
        num_steps = epoch // self.step_size
        self.base_lr = self.base_lr_orig * (self.gamma ** num_steps)
        return self.base_lr

def get_lr(trainer, scheduler):
    lr = scheduler.get_last_lr()[0]
    trainer.step()
    scheduler.step()
    return lr

def get_loss(loss_type, model, misclf_key, dataloader, weight=20, set_criterion=None, forward=None):
    loss_total = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    if set_criterion == None:
        error_weights = {misclf_key: weight}
        if loss_type == 'Focal_Loss':
            criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
        elif loss_type == 'Weight_Focal_Loss':
            criterion = Weighted_FocalLoss(alpha=0.25, gamma=2.0, reduction='mean', error_weights=error_weights)
        elif loss_type == 'Dynamic_Balanced_Loss':
            criterion = dynamic_balance_loss(error_weights=error_weights, k=1, t=1)
        elif loss_type == 'Dynamic_Balance_Focal_Loss':
            criterion = dynamic_balance_focal_loss(error_weights=error_weights, k=1, t=1)
        elif loss_type == 'MSE_Loss':
            criterion = nn.MSELoss()
        else:  # Weighted_Loss
            criterion = WeightedLoss(error_weights)
    else:
        criterion = set_criterion

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            if forward != 'v2':
                outputs = model(inputs)
            else:
                outputs = model.forward_v2(inputs)
            loss = criterion(outputs, labels)
            loss_total += loss.item() * labels.size(0)

    return loss_total

val_losses = []
eval_losses = []
epochs = []
def plot_for_loss(epoch, val_loss=None, eval_loss=None):
    print('plot_for_loss')
    import matplotlib.pyplot as plt

    if epoch == 0:
        plt.ion()
        plt.figure(figsize=(10, 5))

    val_losses.append(val_loss)
    eval_losses.append(eval_loss)
    epochs.append(epoch)
    plt.clf()

    if val_loss:
        plt.plot(epochs, val_losses, label='Training Loss')
    if eval_loss:
        plt.plot(epochs, eval_losses, label='Testing Loss')

    plt.legend()
    plt.title('Training and Evaluation Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.pause(0.1)
    plt.show()

    return

def print_memory_usage():
    allocated_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved_memory = torch.cuda.memory_reserved(device) / (1024 ** 2)
    print(f'Allocated memory: {allocated_memory:.2f} MB')
    print(f'Reserved memory: {reserved_memory:.2f} MB')


