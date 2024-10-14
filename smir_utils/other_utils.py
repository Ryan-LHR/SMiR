"""
other_utils
"""
import pickle
import os

def save_point_print(text='None'):
    length = 50 
    divider_long = length * '-' 
    
    import re
    text_en = re.sub("[\u4e00-\u9fa5\0-9\,\。]", "", text) 
    text_ch = re.sub("[A-Za-z0-9\,\。]", "", text) 
    text_num = re.sub("[A-Za-z\u4e00-\u9fa5\,\。]", "", text) 
    text_length = len(text_en) + len(text_num) + 1.5*len(text_ch) 

    half_divider_length = int((length - text_length)/2)
    divider_short = half_divider_length * '-'
    divider_text = divider_short + text +divider_short

    print('\n' + divider_long)
    print(divider_text)
    print(divider_long)

def get_result_overall(result_list):
    num_exp = len(result_list)  
    result_dict_overall = {'OverAll Result': 'As follows'}  
    for key in result_list[0].keys():  
        if type(result_list[0].get(key)) == (int or float):  
            result_dict_overall[key] = 0
            for i in range(num_exp):
                if result_list[i].get(key) is None:
                    result_dict_overall[key] += 0  
                else:
                    result_dict_overall[key] += result_list[i].get(key)  
            result_dict_overall[key] = round(result_dict_overall[key]/num_exp, ndigits=5)   
        else:  
            result_dict_overall[key] = None

    if 'num of repair on(eval)' and 'num of introducing misclf(eval)' in result_list[0].keys():
        result_tuples = [
            (d['num of repair on(eval)'], d['num of introducing misclf(eval)'])
            for d in result_list
        ]
        result_dict_overall['OverAll Result List'] = result_tuples
    else:
        result_dict_overall['OverAll Result List'] = 'key does not exist'
    print('result of overall exam: \n' + str(result_dict_overall))
    return result_dict_overall


def save_result_to_excel(save_path, result_dict_overall, ex_setup):
    import csv
    folder_path, filename = os.path.split(save_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with open(save_path, 'a+', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([ex_setup])  
        for key in result_dict_overall.keys():  
            writer.writerow([key, result_dict_overall[key]])
        writer.writerow(['--------------']) 

def setup_seed(seed):
    import torch
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def save_to_pickle(path, data):
    folder_path, filename = os.path.split(path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with open(path, 'wb') as file:
        pickle.dump(data, file)
        print("Save data to %s" % path)


def read_from_pickle(path):
    with open(path, 'rb') as file:  
        data_loaded = pickle.load(file)
        print("Read data from %s" % path)
    return data_loaded




