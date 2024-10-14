
import numpy as np
from copy import deepcopy
import torch.nn as nn
import torch
from tqdm import tqdm
import pickle
import os
from smir.smir_utils.other_utils import save_to_pickle, read_from_pickle
from torch.utils.data import TensorDataset, DataLoader
import random
from smir.smir_utils.model_utils import get_predictions, get_target_misclf

def localise_by_deeplift(model_for_dl, dataloader, model_name, model_file, misclf_key, seed,
                          select_method, select_num, target_method=None):
    print('\nlocalise_by_deeplift')
    import torch
    import re
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path_mdl_before_retrain = model_file

    fixed_seed = 0
    fixed_epochs = 200
    fixed_weight = 1
    retrain_weight = 'fc-freezed'

    if model_name == 'simple_fm':
        fixed_name = 'fm_simple_withforward2_patched_by_binary_retrain'
    elif model_name == 'simple_cm':
        fixed_name = 'cm_simple_withforward2_patched_by_retrain_seed_deeplift_patched_by_binary_retrain'
    elif model_name == 'C10_CNN1':
        fixed_name = 'C10_CNN1_withforward2_patched_by_retrain_seed_deeplift_patched_by_binary_retrain'
    elif model_name == 'simple_svhn':
        fixed_name = 'simple_svhn_withforward2_acc(0.8715)_patched_by_binary_retrain'
    elif model_name == 'SVHN_CNN1':
        fixed_name = 'svhn_cnn1_withforward2_acc(0.9317)_patched_by_binary_retrain'
    elif model_name == 'NEU-CLS-64_CNN':
        fixed_name = 'NEU-CLS-64_CNN_withoutDropout_acc(0.9217)_patched_by_binary_retrain'
    elif model_name == 'APTOS2019_ResNet18':
        fixed_name = 'APTOS2019_resnet18_new_withDropout_acc(0.6576)_patched_by_binary_retrain'
    else:
        raise ValueError("path_mdl_after_retrain in not set")

    path_mdl_after_retrain = f'../lhr_results/{model_name}/patched_model/{fixed_name}_{misclf_key}' \
                             f'_seed{fixed_seed}_epochs{fixed_epochs}_weight{fixed_weight}_fc-freezed.pth'
    print('path_mdl_after_retrain:{}'.format(path_mdl_after_retrain))

    model_bf_re = torch.load(path_mdl_before_retrain, map_location=torch.device(device))
    model_af_re = torch.load(path_mdl_after_retrain, map_location=torch.device(device))

    batch_size = 1024
    FE_bf_re = FeatureExtractor(model_bf_re, device)
    FE_bf_re.extract_features(dataloader, model_name)
    feature_bf_re_loader = FE_bf_re.create_feature_loader(batch_size)

    FE_af_re = FeatureExtractor(model_af_re, device)
    FE_af_re.extract_features(dataloader, model_name)
    feature_af_re_loader = FE_af_re.create_feature_loader(batch_size)

    target_method = 'True'
    save_path = '../lhr_data/models/{}/loc/index_difference_list_{}_{}_target-{}.pkl'.\
        format(model_name, misclf_key, retrain_weight, target_method)
    print(save_path)

    if os.path.exists(save_path):
        index_difference_list, attributions_bf_re, attributions_af_re = read_from_pickle(save_path)
        print('load index_difference_list from path:{}'.format(save_path))
    else:
        attributions_bf_re = return_mean_attrbutions(model_for_dl, model_name, feature_bf_re_loader, misclf_key, target_method)
        attributions_af_re = return_mean_attrbutions(model_for_dl, model_name, feature_af_re_loader, misclf_key, target_method)

        differences = attributions_af_re - attributions_bf_re

        sorted_indices = torch.argsort(differences, descending=True)

        index_difference_list = [(idx.item(), differences[idx].item()) for idx in sorted_indices]
        save_file = (index_difference_list, attributions_bf_re, attributions_af_re)
        save_to_pickle(save_path, save_file)

    indices_list = []

    print('localise strategy:{}, localised dimension number:{}'.format(select_method, select_num))
    if select_method == 'random':
        indices_list = random_select_from_list(list(range(len(index_difference_list))), select_num, seed)
    elif select_method == 'bf_most':
        _, top_indices = torch.sort(attributions_bf_re, descending=True)
        indices_list = top_indices[:select_num]

    elif select_method == 'af_most':
        _, top_indices = torch.sort(attributions_af_re, descending=True)
        indices_list = top_indices[:select_num]

    elif select_method == 'change_most':
        for i in range(select_num):
            index = index_difference_list[i][0]
            indices_list.append(index)
    return indices_list, feature_bf_re_loader, feature_af_re_loader, model_af_re


def get_feature_modified(indices_list, feature_bf_re_loader, feature_af_re_loader):
    modified_features = []
    modified_labels = []

    for (features_bf, labels_bf), (features_af, _) in zip(feature_bf_re_loader, feature_af_re_loader):
        for index in indices_list:
            features_bf[:, index] = features_af[:, index]

        modified_features.append(features_bf)
        modified_labels.append(labels_bf)

    modified_features = torch.cat(modified_features, dim=0)
    modified_labels = torch.cat(modified_labels, dim=0)

    modified_dataset = TensorDataset(modified_features, modified_labels)
    modified_dataloader = DataLoader(modified_dataset,  batch_size=100, shuffle=False)

    return modified_dataloader

def return_mean_attrbutions(model_for_dl, model_name, dataloader, misclf_key, target_method=None):
    from captum.attr import DeepLift
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_for_dl.eval()

    deep_lift = DeepLift(model=model_for_dl, multiply_by_inputs=True, eps=1e-10)
    attributions = []
    for images, labels in tqdm(dataloader):
        images, labels = images.to(device), labels.to(device)
        input_tensors = images
        input_tensors.requires_grad = True

        for i, label in enumerate(labels):
            if target_method == 'True':
                attribution = deep_lift.attribute(input_tensors[i].unsqueeze(0), target=label.item())
            elif target_method == 'Orig':
                attribution = deep_lift.attribute(input_tensors[i].unsqueeze(0), target=int(misclf_key[0]))
            elif target_method == 'Pred':
                attribution = deep_lift.attribute(input_tensors[i].unsqueeze(0), target=int(misclf_key[1]))
            attributions.append(attribution)

    mean_attributions = torch.cat(attributions, dim=0).mean(dim=0)
    print(f"Mean attributions: {mean_attributions}")
    return mean_attributions

class FeatureExtractor:
    def __init__(self, original_model, device):
        self.model = original_model
        self.device = device
        self.features_list = []
        self.labels_list = []

    def forward_hook(self, module, input, output):
        self.features_list.append(output.view(output.size(0), -1))

    def extract_features(self, dataloader, model_name):
        if model_name == 'simple_fm':
            hook = self.model.relu.register_forward_hook(self.forward_hook)
        elif model_name in ['simple_cm', 'simple_svhn']:
            hook = self.model.maxpool1.register_forward_hook(self.forward_hook)
        elif model_name in ['C10_CNN1', 'SVHN_CNN1']:
            hook = self.model.maxpool_v2.register_forward_hook(self.forward_hook)
        elif model_name == 'NEU-CLS-64_CNN':
            hook = self.model.maxpool_v2.register_forward_hook(self.forward_hook)
        elif model_name == 'APTOS2019_ResNet18':
            hook = self.model.avgpool.register_forward_hook(self.forward_hook)
        else:
            raise ValueError("hook is not set")

        self.model.eval()
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                self.model(inputs)
                self.labels_list.append(labels)
        hook.remove()

    def create_feature_loader(self, batch_size):
        features = torch.cat(self.features_list, dim=0)
        labels = torch.cat(self.labels_list, dim=0)

        dataset = TensorDataset(features, labels)
        feature_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.features_list = []
        self.labels_list = []

        return feature_loader

    def perturb_features(self, feature_loader, perturb_dim, seed, p_amount):
        perturbed_features_list = []
        labels_list = []
        for features, labels in feature_loader:
            features, labels = features.to(self.device), labels.to((self.device))
            perturbed_features = features.clone()
            torch.manual_seed(seed)
            noise = torch.randn(features.size(0), device=self.device) * p_amount
            perturbed_features[:, perturb_dim] += noise
            perturbed_features_list.append(perturbed_features)
            labels_list.append(labels)

        perturbed_features = torch.cat(perturbed_features_list, dim=0)
        labels = torch.cat(labels_list, dim=0)

        perturbed_dataset = TensorDataset(perturbed_features, labels)
        perturbed_loader = DataLoader(perturbed_dataset, batch_size=feature_loader.batch_size, shuffle=False)

        return perturbed_loader

def random_select_from_list(input_list, num_of_return, seed):
    num_of_return = min(num_of_return, len(input_list))
    random.seed(seed)
    selected_elements = random.sample(input_list, num_of_return)

    return selected_elements

def get_loc_result(args, model_for_dl, model_after_retrain, indices_list,
                   eval_feat_bf_re_loader, eval_feat_af_re_loader,
                   eval_retrain_binary_loader, misclf_key):
    eval_feat_modified_loader = get_feature_modified(indices_list, eval_feat_bf_re_loader, eval_feat_af_re_loader)
    (_, eval_binary_labels, eval_binary_corrects_orig) = \
        get_predictions(model_for_dl, eval_feat_bf_re_loader, return_bool=True, return_tensor=False,
                        is_print=False, misclf_key=None)
    eval_sus_y = [label for _, label in eval_feat_bf_re_loader.dataset]
    eval_target_now_indices = get_target_misclf(misclf_key, eval_binary_labels, eval_sus_y, is_print=False)

    eval_feat_modified_loader = DataLoader(eval_feat_modified_loader.dataset, batch_size=1024, shuffle=False)
    (_, eval_binary_labels_partial_replaced, eval_binary_corrects_partial_replaced) = \
        get_predictions(model_for_dl, eval_feat_modified_loader, return_bool=True, return_tensor=False,
                        is_print=False, misclf_key=None)

    (_, eval_binary_labels_overall_replaced, eval_binary_corrects_overall_replaced) = \
        get_predictions(model_for_dl, eval_feat_af_re_loader, return_bool=True, return_tensor=False,
                        is_print=False, misclf_key=None)

    (_, eval_binary_labels, eval_binary_corrects) = \
        get_predictions(model_after_retrain, eval_retrain_binary_loader, return_bool=True, return_tensor=False,
                        is_print=False, misclf_key=None)
    num_of_data = len(eval_feat_bf_re_loader.dataset)

    result_dict_loc = {'misclf_key': misclf_key,
                       'Loc stategy': args.loc_select_method,
                       'selected dimensions': args.loc_dimensions,
                       'num of correct(binary eval orig)': sum(eval_binary_corrects_orig),
                       'accuracy(binary eval orig)': sum(eval_binary_corrects_orig),
                       'num of correct(binary eval partial replaced)': sum(eval_binary_corrects_partial_replaced),
                       'num of correct(binary eval overall replaced)': sum(eval_binary_corrects_overall_replaced),
                       }
    return result_dict_loc

def loc_for_acc(model_name, device, dataloader, model_bf_re,
                model_for_dl, misclf_key, args, t_decresed,
                select_method, select_num, p_amount):

    batch_size = 1024
    FE_bf_re = FeatureExtractor(model_bf_re, device)
    FE_bf_re.extract_features(dataloader, model_name)
    feature_bf_re_loader = FE_bf_re.create_feature_loader(batch_size)

    (_, eval_binary_labels_original, eval_binary_corrects_original) = \
        get_predictions(model_for_dl, feature_bf_re_loader, return_bool=True, return_tensor=False,
                        is_print=False, misclf_key=None)
    print('The number of correct original predictions for categories A and B in eval is: {}'.format(
        sum(eval_binary_corrects_original)))

    for data, labels in feature_bf_re_loader:
        feature_size = data.size(1)
        break
    print(f'feature size is {feature_size}')

    random.seed(args.seed)
    feature_range = list(range(feature_size))
    random.shuffle(feature_range)

    for perturb_dim in feature_range:
        feature_af_p_loader = FE_bf_re.perturb_features\
            (feature_bf_re_loader, perturb_dim, args.seed, p_amount=p_amount)

        (_, eval_binary_labels_perturbed, eval_binary_corrects_perturbed) = \
            get_predictions(model_for_dl, feature_af_p_loader , return_bool=True, return_tensor=False,
                            is_print=False, misclf_key=None)
        print('The number of correct predictions for categories A and B in eval after perturbation is: {}'.format(
            sum(eval_binary_corrects_perturbed)))

        num_decresed = sum(eval_binary_corrects_original)-sum(eval_binary_corrects_perturbed)
        if num_decresed >= t_decresed:
            print(f'Requirements met, the perturbed feature dimensions are: {perturb_dim}')
            break
    target_method = 'True'
    attributions_bf_re = return_mean_attrbutions(model_for_dl, model_name, feature_af_p_loader, misclf_key, target_method)
    attributions_af_re = return_mean_attrbutions(model_for_dl, model_name, feature_bf_re_loader, misclf_key, target_method)
    differences = attributions_af_re - attributions_bf_re
    sorted_indices = torch.argsort(differences, descending=True)

    index_difference_list = [(idx.item(), differences[idx].item()) for idx in sorted_indices]
    indices_list = []

    print('Feature selection strategy: {}, number of selected feature dimensions: {}'.format(select_method, select_num))
    if select_method == 'random':
        indices_list = random_select_from_list(feature_range, select_num, args.seed)
        print(indices_list)
    elif select_method == 'bf_most':
        _, top_indices = torch.sort(attributions_bf_re, descending=True)
        indices_list = top_indices[:select_num]

    elif select_method == 'af_most':
        _, top_indices = torch.sort(attributions_af_re, descending=True)
        indices_list = top_indices[:select_num]

    elif select_method == 'change_most':
        for i in range(select_num):
            index = index_difference_list[i][0]
            indices_list.append(index)

    for item in index_difference_list:
        if item[0] == perturb_dim:
            print(f"Perturbed dimension: {item[0]}, Change: {item[1]}")

    if perturb_dim in indices_list:
        is_loc_accurate = 1
        loc_positions = [index for index, value in enumerate(indices_list) if value == perturb_dim]

        if len(loc_positions) == 1:
            loc_position = loc_positions[0]+1
        else:
            raise ValueError("The positions list contains more than one element or is empty")
        print(f'Perturbed dimension located: {perturb_dim}, located at top {loc_position}')
    else:
        is_loc_accurate = 0
        loc_position = -1
        print(f'Perturbed dimension not located')

    result_dict_loc = {'misclf_key': misclf_key,
                       'Loc stategy': args.loc_select_method,
                       'selected dimensions': args.loc_dimensions,
                       'is_loc_correct': is_loc_accurate,
                       'loc_position': loc_position,
                           }
    return result_dict_loc

def get_overall_loc_results(model_name, result_dict_overall, result_list, loc_dimensions):
    from collections import Counter

    result_counts = {-1: 0}
    for i in range(1, loc_dimensions + 1):
        result_counts[i] = 0

    loc_positions = [result_dict['loc_position'] for result_dict in result_list]
    loc_position_counts = Counter(loc_positions)

    for position, count in loc_position_counts.items():
        result_counts[position] = count
    result_dict_overall['loc_position'] = result_counts

    if model_name == 'simple_fm':
        loc_dimensions = [1, 3, 5, 10, 20, 50]
    elif model_name in ['simple_cm', 'C10_CNN1', 'simple_svhn', 'SVHN_CNN1']:
        loc_dimensions = [5, 10, 20, 50, 100, 500]

    loc_dict = result_dict_overall['loc_position']
    total_count = sum(loc_dict.values())
    if total_count == 0:
        raise ValueError("Total count of all loc positions is zero, cannot calculate frequencies.")

    topk_rates = {}
    for k in loc_dimensions:
        topk_keys = range(1, k + 1)
        topk_count = sum(loc_dict.get(key, 0) for key in topk_keys)
        topk_rate = topk_count / total_count * 100
        topk_rates[f"top{k}"] = str(topk_rate) + '%'
    result_dict_overall['accuracy'] = topk_rates

    return result_dict_overall

