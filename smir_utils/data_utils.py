import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset
from smir.smir_utils import tsne
from smir.smir_utils.model_utils import get_predictions

def save_fig(dataset, image, image_id, save_dir, tight_layout=True, extension='jpg', dpi=1):
    plt_save = show_fig(dataset, image, show=False)
    if tight_layout:
        plt_save.tight_layout()
    save_path = os.path.join(save_dir, image_id + '.' + extension)
    plt_save.savefig(save_path, format=extension, dpi=dpi)

def show_fig(dataset, image, show = True):
    if dataset in ['mnist', 'fashion_mnist']:
        image_2d = image.reshape(28, 28)
        plt.figure(figsize=(28, 28), dpi=1)
        plt.imshow(image_2d, cmap=mpl.cm.binary)
        plt.axis('off')
    elif dataset in ['cifar_10', 'cifar_100']:
        print('cifar_10')
    if show:
        plt.show()
    else:
        return plt

def save_precditions(save_path, prediction_labels, true_labels):
    import csv
    with open(save_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['index', 'true', 'pred'])
        for i in range(len(prediction_labels)):
            writer.writerow([i, true_labels[i], prediction_labels[i]])
    return save_path

def load_precditions(save_path):
    print('\nload_precditions_from_excel')
    import pandas as pd
    df = pd.read_csv(save_path, index_col='index')
    misclf_df = df.loc[df.true != df.pred]
    print('load finished')
    return

def split_into_wrong_and_correct(prediction_bool):
    indices_dict = {'wrong':[], 'correct':[]}
    indices_to_wrong, = np.where(prediction_bool == False)
    indices_to_correct, = np.where(prediction_bool == True)
    indices_dict['correct'] = list(indices_to_correct)
    indices_dict['wrong'] = list(indices_to_wrong)
    return indices_dict

def get_balanced_dataset(pred_file, top_n, idx=0):
    import pandas as pd
    from collections.abc import Iterable
    idx = idx if idx == 0 else 1
    target_idx = idx
    eval_idx = np.abs(1 - target_idx)
    df = pd.read_csv(pred_file, index_col='index')
    misclf_df = df.loc[df.true != df.pred]
    misclfds_idx_target = get_misclf_indices_balanced(misclf_df, idx=target_idx)
    misclfds_idx_eval = get_misclf_indices_balanced(misclf_df, idx=eval_idx)
    indices_to_corr = df.loc[df.true == df.pred].sort_values(by=['true']).index.values
    indices_to_corr_target = [_idx for i, _idx in enumerate(indices_to_corr) if i % 2 == target_idx]
    indices_to_corr_eval = [_idx for i, _idx in enumerate(indices_to_corr) if i % 2 == eval_idx]

    np.random.seed(0)
    sorted_keys = sort_keys_by_cnt(misclfds_idx_target)
    if top_n >= len(sorted_keys):
        msg = "{} is provided when there is only {} number of misclfs".format(
            top_n, len(sorted_keys))
        assert False, msg
    else:
        misclf_key = sorted_keys[top_n]
        misclf_indices = misclfds_idx_target[misclf_key]
        misclf_indices_eval = misclfds_idx_eval[misclf_key]

        new_data_indices = []; new_test_indices = []
        for sorted_k in sorted_keys:
            
            new_data_indices.extend(misclfds_idx_target[sorted_k])
            new_test_indices.extend(misclfds_idx_eval[sorted_k])

        new_data_indices += indices_to_corr_target
        new_test_indices += indices_to_corr_eval
        np.random.shuffle(new_data_indices)
        np.random.shuffle(new_test_indices)
        return (misclf_key, new_data_indices, misclf_indices, new_test_indices, misclf_indices_eval)

def get_target_dataset(pred_file, top_n, idx_to_class=None, misclf_key=None):
    np.random.seed(0)
    import pandas as pd
    df = pd.read_csv(pred_file, index_col='index')
    num_of_data = df.shape[0]
    misclf_df = df.loc[df.true != df.pred]
    misclfds_idx = get_misclf_indices(misclf_df)
    corr_indices = df.loc[df.true == df.pred].sort_values(by=['true']).index.values

    sorted_keys = sort_keys_by_cnt(misclfds_idx)
    if top_n >= len(sorted_keys):
        msg = "{} is provided when there i s only {} number of misclfs".format(
            top_n, len(sorted_keys))
        assert False, msg
    else:
        if misclf_key is None:
            misclf_key = sorted_keys[top_n]
        if misclf_key in misclfds_idx:
            misclf_indices = misclfds_idx[misclf_key]
        else:
            misclf_indices = []

    length = min(len(sorted_keys), 10)
    for n in range(length):
        k = sorted_keys[n]
        misclf_label = (idx_to_class[k[0]], idx_to_class[k[1]])
        print(f'top{n}error is {k}, namely {misclf_label}, num is {misclfds_idx[k].size}')

    data_indices = range(num_of_data)
    return (misclf_key, data_indices, misclf_indices, corr_indices)


def get_target_dataset_arachne(pred_file, top_n, idx_to_class=None, misclf_key=None):
    print('\nget_target_dataset')
    np.random.seed(0)
    import pandas as pd
    df = pd.read_csv(pred_file, index_col='index')
    num_of_data = df.shape[0]  
    misclf_df = df.loc[df.true != df.pred]
    misclfds_idx = get_misclf_indices(misclf_df)

    corr_indices = df.loc[df.true == df.pred].sort_values(by=['true']).index.values

    sorted_keys = sort_keys_by_cnt(misclfds_idx) 
    if top_n >= len(sorted_keys):  
        msg = "{} is provided when there i s only {} number of misclfs".format(
            top_n, len(sorted_keys))
        assert False, msg
    else:
        if misclf_key is None:
            misclf_key = sorted_keys[top_n] 
        if misclf_key in misclfds_idx:
            misclf_indices = misclfds_idx[misclf_key] 
            print(f"Found indices for key {misclf_key}: {misclf_indices}")
        else:
            print(f"Key {misclf_key} not found in misclfds_idx, randomly selected one index")
            misclf_indices = []

    data_indices = range(num_of_data)  
    return (misclf_key, data_indices, misclf_indices, corr_indices)

def get_misclf_indices(df):
    misclf_types = list(set(
        [tuple(pair) for pair in df[["true", "pred"]].values]))
    ret_misclfds = {}  
    for misclf_type in misclf_types:  
        misclf_type = tuple(misclf_type)
        true_label, pred_label = misclf_type
        indices_to_misclf = df.loc[
            (df.true == true_label) & (df.pred == pred_label)].index.values
        ret_misclfds[misclf_type] = indices_to_misclf
    return ret_misclfds

def get_misclf_indices_balanced(df, idx=0):
    misclf_types = list(set(
        [tuple(pair) for pair in df[["true", "pred"]].values]))
    ret_misclfds = {}  
    for misclf_type in misclf_types:
        misclf_type = tuple(misclf_type)
        true_label, pred_label = misclf_type
        
        indices_to_misclf = df.loc[
            (df.true == true_label) & (df.pred == pred_label)].index.values
        if len(indices_to_misclf) >= 2:  
            
            indices_1, indices_2 = np.array_split(indices_to_misclf, 2)
            ret_misclfds[misclf_type] = indices_1 if idx == 0 else indices_2  
        else:  
            ret_misclfds[misclf_type] = indices_to_misclf

    return ret_misclfds


def sort_keys_by_cnt(misclfds):
    cnts = []  
    for misclf_key in misclfds:
        current_misclf = [misclf_key, len(misclfds[misclf_key])]  
        cnts.append(current_misclf)
    
    sorted_keys = [v[0] for v in sorted(cnts, key = lambda v:v[1], reverse = True)]
    return sorted_keys

def get_results_for_arachne_rq3(RR_val, RR_eval):
    ratio = RR_eval / RR_val
    mean = ratio.mean()
    std = ratio.std()
    print(mean)
    print(std)
    return

def convert_ndarray_to_tensor(dataset, data):
    if torch.is_tensor(data[0][0]):  
        return data
    data_X, data_y = data
    data_X = data_X.tolist()
    data_y = data_y.tolist()
    X_Tensor = []
    y_Tensor = []
    for index, data_list in enumerate(data_X):
        data_tensor = torch.Tensor(data_list)
        if dataset == 'fashion_mnist':
            data_tensor = data_tensor.reshape(1, 1, 28, 28)
        elif dataset == 'cifar_10':
            data_tensor = data_tensor.reshape(1, 3, 32, 32)
        X_Tensor.append(data_tensor)

    for index, data_list in enumerate(data_y):
        data_list = [data_list]
        data_tensor = torch.Tensor(data_list)
        y_Tensor.append(data_tensor)

    data_Tensor = (X_Tensor, y_Tensor)
    return data_Tensor


from torch.utils.data import Dataset
class custom_dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        image = self.data[0][index].squeeze(0)
        label = self.data[1][index][0]
        return (image, label)


    def __len__(self):
        data_size = len(self.data[0])
        return data_size

class binary_dataset(Dataset):
    def __init__(self, original_dataset, misclf_key):
        self.dataset = original_dataset
        self.misclf_key = misclf_key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        binary_label = 1 if label == self.misclf_key[0] else 0
        return image, binary_label

class target_binary_dataset(Dataset):
    def __init__(self, original_dataset, misclf_key, orig_prediction_labels):
        self.dataset = original_dataset
        self.misclf_key = misclf_key
        self.pred_labels = orig_prediction_labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        pred_label = self.pred_labels[idx]
        
        binary_label = 1 if ((label, pred_label) == self.misclf_key) else 0
        return image, binary_label

class SelectedDimensionsDataset(Dataset):
    def __init__(self, bf_dataset, af_dataset, indices_list):
        self.bf_dataset = bf_dataset
        self.af_dataset = af_dataset
        self.indices_list = indices_list

    def __len__(self):
        return min(len(self.bf_dataset), len(self.af_dataset))

    def __getitem__(self, idx):
        bf_data, _ = self.bf_dataset[idx]
        af_data, _ = self.af_dataset[idx]
        selected_bf_data = bf_data[self.indices_list]
        selected_af_data = af_data[self.indices_list]
        return selected_bf_data, selected_af_data



def create_selected_dimensions_dataloader(bf_loader, af_loader, indices_list, batch_size=1024):
    bf_dataset = bf_loader.dataset
    af_dataset = af_loader.dataset

    selected_dataset = SelectedDimensionsDataset(bf_dataset, af_dataset, indices_list)
    selected_dataloader = DataLoader(selected_dataset, batch_size=batch_size, shuffle=False)

    return selected_dataloader

def replace_features_dataloader(model, dataloader, indices_list):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  

    updated_features = []  
    labels = []

    with torch.no_grad():  
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            model_inputs = inputs[:, indices_list].to(device)
            model_outputs = model(model_inputs)
            inputs[:, indices_list] = model_outputs
            updated_features.append(inputs)
            labels.append(targets)

    updated_features_tensor = torch.cat(updated_features, dim=0)
    labels_tensor = torch.cat(labels, dim=0)

    new_dataset = TensorDataset(updated_features_tensor, labels_tensor)
    new_dataloader = DataLoader(new_dataset, batch_size=dataloader.batch_size, shuffle=False)
    return new_dataloader

def format_label(labels, num_label):
    """
    format label which has a integer label to flag type
    e.g., [3, 5] -> [[0,0,1,0,0],[0,0,0,0,1]]
    """
    num_data = len(labels)
    from collections.abc import Iterable
    new_labels = np.zeros([num_data, num_label])
    for i, v in enumerate(labels):
        if isinstance(v, Iterable):
            new_labels[i, v[0]] = 1
        else:
            new_labels[i, v] = 1

    return new_labels

def get_label_times(list):
    count_dict = {}
    for num in list:
        if num in count_dict:
            count_dict[num] += 1
        else:
            count_dict[num] = 1
    return count_dict


def merge_dataloaders(dataset, indices_a, loader_a, indices_b, loader_b, batch_size=128):
    all_indices = []
    all_data = []
    all_targets = []

    for batch_idx, (data, target) in enumerate(loader_a):
        indices = indices_a[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        all_indices.extend(indices)
        all_data.extend(data)
        all_targets.extend(target)

    for batch_idx, (data, target) in enumerate(loader_b):
        indices = indices_b[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        all_indices.extend(indices)
        all_data.extend(data)
        all_targets.extend(target)

    sorted_data = [None] * len(all_indices)
    sorted_targets = [None] * len(all_indices)

    for index, data, target in zip(all_indices, all_data, all_targets):
        sorted_data[index] = data
        sorted_targets[index] = target

    sorted_data = torch.stack(sorted_data)
    sorted_targets = torch.tensor(sorted_targets)

    class CustomDataset(Dataset):
        def __init__(self, data, targets):
            self.data = data
            self.targets = targets

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            return self.data[index], self.targets[index]

    new_dataset = CustomDataset(sorted_data, sorted_targets)
    new_dataloader = DataLoader(new_dataset, batch_size=batch_size, shuffle=False)

    return new_dataloader

def get_top_n_confidence_indices(misclf_key, data_y_orig, predictions, n):
    true_class = misclf_key[0]
    predicted_class = misclf_key[1]
    true_indices = np.where(data_y_orig == true_class)[0]
    predicted_probs = predictions[true_indices, predicted_class]
    top_n_indices = np.argsort(predicted_probs)[-n:][::-1]
    final_indices = true_indices[top_n_indices]
    return final_indices
