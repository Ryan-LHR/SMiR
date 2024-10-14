'''
SMiR
'''
# region (Pre)import pkgs
import argparse
import os
import time
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import itertools

from smir.smir_utils import load_data, model_utils, data_utils, other_utils
from smir.smir_utils.load_data import load_data
from smir.smir_utils.data_utils import *
from smir.smir_utils.model_stru import *
from smir.smir_utils.other_utils import save_point_print, save_to_pickle, read_from_pickle
from smir.smir_utils.model_utils import *
from smir.smir_utils.model_train import *
from smir.smir_utils.localise import *
from smir.smir_utils.model_train import *
from smir.smir_utils.data_utils import binary_dataset

# endregion import (Pre)pkgs

def lhr_main(args):
    print('smir_main')
    save_point_print('experiment:{}'.format(args.rq))
    save_point_print('method:{}'.format(args.method))
    """------------------------------(0)Loading------------------------------"""
    # region (0)Loading
    save_point_print('(0)Loading')
    wandb.login()
    wandb.config = vars(args)
    run = wandb.init( project=f'lhr_{args.rq}')
    seed = args.seed
    top_n = args.top_n
    on_train = args.on_train

    random.seed(seed)  # set random seed
    np.random.seed(seed)

    # load dataset
    train_set, test_set = load_data(dataset=args.dataset, path_to_data=args.dataset_dir,
                                          batch_size=1024, return_dataset=True)
    # load dataset in dataloader format
    train_loader, test_loader = load_data(dataset=args.dataset, path_to_data=args.dataset_dir,
                                          batch_size=1024, return_dataloader=True)

    (val_loader, eval_loader) = (train_loader, test_loader) if on_train else None
    (val_set, eval_set) = (train_set, test_set) if on_train else None

    val_y = [label for _, label in val_set]
    eval_y = [label for _, label in eval_set]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pytorch_model_path = args.model_file
    pytorch_model = torch.load(pytorch_model_path, map_location=torch.device(device))
    # endregion (0)Loading
    """------------------------------(0)Loading------------------------------"""

    """------------------------------(1)Predictions Processing------------------------------"""
    # region (1)Predictions Processing
    save_point_print('(1)Predictions Processing')
    (_, val_prediction_labels, _, val_number) = get_predictions(pytorch_model, val_loader,
                                                return_bool=True, return_tensor=False, return_number='Both')
    val_wrong_orig = val_number[2]
    (_, eval_prediction_labels, _, eval_number) = get_predictions(pytorch_model, eval_loader,
                                                return_bool=True, return_tensor=False, return_number='Both')
    eval_corrects_orig = eval_number[1]; eval_wrong_orig = eval_number[2]
    wrong_orig = (val_wrong_orig, eval_wrong_orig)

    if not (os.path.exists(args.indices_file) and os.path.exists(args.indices_file_onTrain)) :
        smir_data_utils.save_precditions(args.indices_file_onTrain, val_prediction_labels, val_y)
        smir_data_utils.save_precditions(args.indices_file, eval_prediction_labels, eval_y)

    if args.rq not in ['xx']:
        if on_train:
            indices_test = args.indices_file
            indices_train = args.indices_file_onTrain

            idx_to_class = {v: k for k, v in train_set.class_to_idx.items()}
            (misclf_key, eval_indices, eval_indices_to_wrong, eval_indices_to_corr) = \
                get_target_dataset(indices_test, top_n, idx_to_class)
            (_, val_indices, val_indices_to_wrong, val_indices_to_corr) = \
                get_target_dataset(indices_train, top_n, idx_to_class, misclf_key)

            no_topn_val_indices = list(set(val_indices).difference(set(val_indices_to_wrong)))
            no_topn_eval_indices = list(set(eval_indices).difference(set(eval_indices_to_wrong)))
            args.misclf_key = misclf_key

        misclf_label = (idx_to_class[misclf_key[0]], idx_to_class[misclf_key[1]])
        print('\nthe number of val data is %d, with correct misclf: %d, target misclf: %d' %
              (len(val_indices), len(val_indices_to_corr), len(val_indices_to_wrong)))
        print('the number of eval data is%d, with correct misclf: %d, target misclf: %d' %
              (len(eval_indices), len(eval_indices_to_corr), len(eval_indices_to_wrong)))
        print('target misclf : %d → %d, namely:%s → %s' % (misclf_key[0], misclf_key[1], misclf_label[0], misclf_label[1]))

        eval_target_wrong_orig = get_target_misclf(misclf_key, eval_prediction_labels, eval_y,
                                                   is_print=False)

        if args.rq == 'rq4_safety_critical':
            is_scm(misclf_key, train_set, is_print=True)
            misclf_dict_orig = get_all_misclf_number(eval_prediction_labels, eval_y)
            print(f'the type and number of misclassification on eval dataset: {misclf_dict_orig}')

        # get indices of data
        indices_dict = {}
        indices_dict['val_indices'] = val_indices
        indices_dict['val_indices_to_wrong'] = val_indices_to_wrong
        indices_dict['eval_indices'] = eval_indices
        indices_dict['eval_indices_to_wrong'] = eval_indices_to_wrong
        indices_dict['no_topn_val_indices'] = no_topn_val_indices
        indices_dict['no_topn_eval_indices'] = no_topn_eval_indices

    result_dict_orig = summary_results_v3(pytorch_model, val_set, eval_set, indices_dict, misclf_key, wrong_orig)
    print('\nThe original result of model:{}'.format(result_dict_orig))

    # endregion (1)Predictions Processing
    """------------------------------(1)Predictions Processing------------------------------"""

    """------------------------------(2)Localise------------------------------"""
    # region (2)Localise
    save_point_print('(2)Localise')
    print(f"The amount of parameters: {get_model_parameters(pytorch_model)}")

    if not args.only_loc:
        result_dict_v1 = summary_results_v3(pytorch_model, val_set, eval_set, indices_dict, misclf_key, wrong_orig)
        print('original result:{}'.format(result_dict_v1))

    (_, eval_labels, _) = get_predictions(pytorch_model, eval_loader, return_bool=True, return_tensor=False, is_print=False)
    get_target_misclf(misclf_key, eval_labels, eval_y)

    t1_loc = time.time()
    if args.is_loc:
        if args.loc_method == 'deeplift':
            # The A & B samples in val and eval are converted into binary data sets, and the corresponding dataloader is obtained
            val_binary_indices = [i for i, (_, label) in enumerate(val_set) if label in misclf_key]
            eval_binary_indices = [i for i, (_, label) in enumerate(eval_set) if label in misclf_key]

            val_retrain_binary_set = Subset(val_set, val_binary_indices)
            val_retrain_binary_loader = DataLoader(val_retrain_binary_set, shuffle=False, batch_size=128)
            eval_retrain_binary_loader = DataLoader(Subset(eval_set, eval_binary_indices), shuffle=False, batch_size=128)

            # load the original classifier
            model_for_dl = torch.load(args.model_feature_to_n_file, map_location=torch.device(device))

            if args.rq == 'rq3_feature_loc_acc':
                # Fault injection location experiment
                result_dict_loc = loc_for_acc(model_name, device, eval_retrain_binary_loader, pytorch_model,
                                              model_for_dl, misclf_key, args, t_decresed=args.t_decresed,
                                              select_method=args.loc_select_method, select_num=args.loc_dimensions,
                                              p_amount=args.p_amount)
                return result_dict_loc

            indices_list, val_feat_bf_re_loader, val_feat_af_re_loader, model_after_retrain = \
                localise_by_deeplift(model_for_dl, val_retrain_binary_loader, args.model, args.model_file, misclf_key, seed,
                                     select_method=args.loc_select_method, select_num=args.loc_dimensions)

            _, eval_feat_bf_re_loader, eval_feat_af_re_loader, _ = \
                localise_by_deeplift(model_for_dl, eval_retrain_binary_loader, args.model, args.model_file, misclf_key, seed,
                                     select_method=args.loc_select_method, select_num=args.loc_dimensions)
            # Get location synthesis results
            result_dict_loc = get_loc_result(args, model_for_dl, model_after_retrain,  indices_list,
                                             eval_feat_bf_re_loader, eval_feat_af_re_loader,
                                             eval_retrain_binary_loader, misclf_key)
            if args.only_loc:
                return result_dict_loc
    t2_loc = time.time()
    print("\nTime for Localise: %f" % (t2_loc - t1_loc))
    # endregion (2)Localise
    """------------------------------(2)Localise------------------------------"""

    """------------------------------(3)Preparing for Repair------------------------------"""
    # region (3)Preparing for Repair
    save_point_print('(3)Preparing for Repair')

    # Set random number seed
    from smir.smir_utils.other_utils import setup_seed
    setup_seed(0)
    batch_size = 128
    best_result = 100000

    if args.method in ['feature_rec', 'feature_rec-alpha', 'feature_rec-beta', 'feature_rec-gamma']:
        val_binary_set = binary_dataset(val_set, misclf_key)
        val_binary_loader = DataLoader(val_binary_set, shuffle=False, batch_size=128)

        # Initialize Validator(feature m-2)
        model_feature_to_2 = build_model_feature_to_2(model_name, device)

        if args.method in ['feature_rec', 'feature_rec-alpha']:
            # Convert the suspect set on eval to a binary class dataloader
            eval_sus_indices = [index for index, value in enumerate(eval_prediction_labels) if value == misclf_key[1]]
            eval_sus_set = Subset(eval_set, eval_sus_indices)
            eval_sus_binary_set = binary_dataset(eval_sus_set, misclf_key)
            eval_sus_binary_loader = DataLoader(eval_sus_binary_set, shuffle=False, batch_size=128)
            eval_sus_loader = DataLoader(Subset(eval_set, eval_sus_indices), shuffle=False, batch_size=128)

            FE_bf_re_val = FeatureExtractor(pytorch_model, device)
            FE_bf_re_val.extract_features(val_binary_loader, model_name)
            val_binary_feat_loader = FE_bf_re_val.create_feature_loader(batch_size)

            FE_bf_re_eval_sus = FeatureExtractor(pytorch_model, device)
            FE_bf_re_eval_sus.extract_features(eval_sus_binary_loader, model_name)
            eval_sus_binary_feat_loader = FE_bf_re_eval_sus.create_feature_loader(batch_size)

        elif args.method == 'feature_rec-beta':
            val_target_binary_set = target_binary_dataset(val_set, misclf_key, val_prediction_labels)
            val_target_binary_loader = DataLoader(val_target_binary_set, shuffle=False, batch_size=128)

            eval_target_binary_set = target_binary_dataset(eval_set, misclf_key, eval_prediction_labels)
            eval_target_binary_loader = DataLoader(eval_target_binary_set, shuffle=False, batch_size=128)

            FE_bf_re_val_tb = FeatureExtractor(pytorch_model, device)
            FE_bf_re_val_tb.extract_features(val_target_binary_loader, model_name)
            val_target_binary_feat_loader = FE_bf_re_val_tb.create_feature_loader(batch_size)

            FE_bf_re_eval_tb = FeatureExtractor(pytorch_model, device)
            FE_bf_re_eval_tb.extract_features(eval_target_binary_loader, model_name)
            eval_target_binary_feat_loader = FE_bf_re_eval_tb.create_feature_loader(batch_size)

        elif args.method == 'feature_rec-gamma':
            val_sus_indices = [index for index, value in enumerate(val_prediction_labels) if value == misclf_key[1]]
            val_sus_loader = DataLoader(Subset(val_set, val_sus_indices), shuffle=False, batch_size=128)
            val_all_indices = list(range(len(val_set)))
            val_remain_indices = list(set(val_all_indices) - set(val_sus_indices))
            val_remain_loader = DataLoader(Subset(val_set, val_remain_indices), shuffle=False, batch_size=128)

            eval_sus_indices = [index for index, value in enumerate(eval_prediction_labels) if value == misclf_key[1]]
            eval_sus_loader = DataLoader(Subset(eval_set, eval_sus_indices), shuffle=False, batch_size=128)
            eval_all_indices = list(range(len(eval_set)))
            eval_remain_indices = list(set(eval_all_indices) - set(eval_sus_indices))
            eval_remain_loader = DataLoader(Subset(eval_set, eval_remain_indices), shuffle=False, batch_size=128)

            FE_bf_re_val_sus = FeatureExtractor(pytorch_model, device)
            FE_bf_re_val_sus.extract_features(val_sus_loader, model_name)
            val_sus_feat_loader = FE_bf_re_val_sus.create_feature_loader(batch_size)

            FE_bf_re_val_remain = FeatureExtractor(pytorch_model, device)
            FE_bf_re_val_remain.extract_features(val_remain_loader, model_name)
            val_remain_feat_loader = FE_bf_re_val_remain.create_feature_loader(batch_size)

            FE_bf_re_eval_sus = FeatureExtractor(pytorch_model, device)
            FE_bf_re_eval_sus.extract_features(eval_sus_loader, model_name)
            eval_sus_feat_loader = FE_bf_re_eval_sus.create_feature_loader(batch_size)

            FE_bf_re_eval_remain = FeatureExtractor(pytorch_model, device)
            FE_bf_re_eval_remain.extract_features(eval_remain_loader, model_name)
            eval_remain_feat_loader = FE_bf_re_eval_remain.create_feature_loader(batch_size)

        'feature mapping operation'
        if args.method != 'feature_rec-alpha':
            val_feat_mapping_loader = create_selected_dimensions_dataloader(val_feat_bf_re_loader, val_feat_af_re_loader,
                                                                   indices_list)
            # Initialize model for feature mapping
            model_feat_mapping = MLP_Model(len(indices_list)).to(device)

            # Train model for feature mapping
            optimizer_map = optim.SGD(model_feat_mapping.parameters(), lr=args.lr_for_mapping, momentum=0.9)
            for epoch_for_mapping in range(args.epochs_for_mapping):
                val_loss, correct = weighted_retrain(model_feat_mapping, misclf_key, val_feat_mapping_loader, seed,
                                                     epoch=epoch_for_mapping, method='mapping',
                                                     model_name=args.model, loss_type='MSE_Loss', optimizer=optimizer_map,
                                                     batchsize=args.batch_size_for_mapping)

                eval_feat_af_map_loader = replace_features_dataloader(model_feat_mapping, eval_feat_bf_re_loader,
                                                                      indices_list)
                (_, eval_binary_labels, eval_binary_corrects) = \
                    get_predictions(model_for_dl, eval_feat_af_map_loader, return_bool=True, return_tensor=False,
                                    is_print=False, misclf_key=None)
                eval_binary_y = [label for _, label in eval_feat_bf_re_loader.dataset]
                eval_target_now_indices = get_target_misclf(misclf_key, eval_binary_labels, eval_binary_y, is_print=False)
                print('The eval part of the A and B classes learns the mapping and predicts the correct number to be :{} and the wrong number to be :{}'.
                      format(sum(eval_binary_corrects), len(eval_target_now_indices)))

            if args.method == 'feature_rec':
                val_binary_feat_map_loader = replace_features_dataloader(model_feat_mapping,
                                                                         val_binary_feat_loader, indices_list)
                eval_sus_binary_feat_map_loader = replace_features_dataloader(model_feat_mapping,
                                                                              eval_sus_binary_feat_loader, indices_list)
            elif args.method == 'feature_rec-beta':
                val_target_binary_feat_map_loader = replace_features_dataloader(model_feat_mapping,
                                                                                val_target_binary_feat_loader, indices_list)
                eval_target_binary_feat_map_loader = replace_features_dataloader(model_feat_mapping,
                                                                                 eval_target_binary_feat_loader, indices_list)
                eval_target_wrong_orig = get_target_misclf(misclf_key, eval_prediction_labels, eval_y,
                                                                 is_print=False)
                print('eval original prediction correct number :{}, specific misclassification number :{}'.
                      format(eval_corrects_orig, len(eval_target_wrong_orig)))
            elif args.method == 'feature_rec-gamma':
                val_sus_feat_map_loader = replace_features_dataloader(model_feat_mapping,
                                                                                val_sus_feat_loader, indices_list)
                eval_sus_feat_map_loader = replace_features_dataloader(model_feat_mapping,
                                                                                eval_sus_feat_loader, indices_list)
                val_merge_feat_loader = merge_dataloaders(val_set, val_sus_indices, val_sus_feat_map_loader,
                                                   val_remain_indices, val_remain_feat_loader, batch_size=128)
                eval_merge_feat_loader = merge_dataloaders(eval_set, eval_sus_indices, eval_sus_feat_map_loader,
                                                   eval_remain_indices, eval_remain_feat_loader, batch_size=128)
                eval_target_wrong_orig = eval_indices_to_wrong

        if args.method in ['feature_rec', 'feature_rec-alpha']:
            (_, eval_sus_labels, eval_binary_corrects_orig) = \
                get_predictions(pytorch_model, eval_sus_loader, return_bool=True, return_tensor=False,
                                is_print=False, misclf_key=None)
            eval_sus_y = [label for _, label in eval_sus_loader.dataset]
            eval_target_wrong_orig = get_target_misclf(misclf_key, eval_sus_labels, eval_sus_y, is_print=False)
            # eval_target_wrong_orig = eval_target_now_indices_orig
            print('The eval portion of the doubt set is the original predicted correct number :{}, and the specific misclassification number is :{}'.
                  format(sum(eval_binary_corrects_orig), len(eval_target_wrong_orig)))

    elif args.method == 'retrain':
        best_result = result_dict_v1['eval中引入误分类'] # the introduction misclassifications of eval set

    elif args.method == 'binary_retrain':
        from smir.smir_utils.data_utils import binary_dataset
        val_binary_indices = [i for i, (_, label) in enumerate(val_set) if label in misclf_key]
        eval_binary_indices = [i for i, (_, label) in enumerate(eval_set) if label in misclf_key]

        val_retrain_binary_set = Subset(val_set, val_binary_indices)
        val_retrain_binary_loader = DataLoader(val_retrain_binary_set, shuffle=True, batch_size=128)
        eval_retrain_binary_loader = DataLoader(Subset(eval_set, eval_binary_indices), shuffle=False, batch_size=128)

        (_, eval_prediction_labels, eval_correct, number_3) = get_predictions(pytorch_model, eval_retrain_binary_loader,
                                                                   return_bool=True, return_tensor=False, return_number='Both')
        eval_retrain_binary_corr_orig = number_3[1]
        print('The correct number of eval predictions for class A and B is:{}'.format(eval_retrain_binary_corr_orig))
        best_result = 0

    """------------------------------(3)Preparing for Repair------------------------------"""
    # endregion (3)Preparing for Repair

    """------------------------------(4)Repair------------------------------"""
    # print(torch.cuda.memory_summary(device=None, abbreviated=False))
    # region (4)Repair
    save_point_print('(4)Repair')
    t1_train = time.time()

    best_model = deepcopy(pytorch_model)

    val_results, eval_results, loss_list = [], [], []
    for epoch in tqdm(range(args.epochs_for_repair)):
        para_misclf_key = None

        if args.method == 'wce':
            args.criterion = Weighted_FocalLoss(alpha=0.25, gamma=args.gamma, reduction='mean', error_weights={misclf_key: args.weight})
            if epoch == 0:
                save_point_print(str(args.criterion))
            val_loss, correct = weighted_retrain(pytorch_model, misclf_key, val_loader, seed, weight=None,
                                                 epoch=epoch, method=args.method, loss_type=None, set_criterion=args.criterion,
                                                 batchsize=args.batch_size, lr=args.lr, scheduler=args.scheduler)

        # retrain the underfitting model
        elif args.method == 'retrain':
            seed = 0
            args.weight = 1
            val_loss, correct = weighted_retrain(pytorch_model, misclf_key, val_loader, seed,
                             weight=args.weight, epoch=epoch, method=args.method,
                             model_name=args.model, loss_type=None, set_criterion=args.criterion,
                             batchsize=args.batch_size, lr=args.lr, scheduler=None)

        # train the DFE
        elif args.method == 'binary_retrain':
            args.batch_size = 1024
            args.lr = 0.0005
            val_loss, correct = weighted_retrain(pytorch_model, misclf_key, val_retrain_binary_loader, seed,
                                                 weight=args.weight, epoch=epoch, method=args.method,
                                                 model_name=args.model, loss_type=None, set_criterion=args.criterion,
                                                 batchsize=args.batch_size, lr=args.lr, scheduler=None)

            (_, eval_binary_labels, eval_binary_corrects) = \
                get_predictions(pytorch_model, eval_retrain_binary_loader, return_bool=True, return_tensor=False, is_print=False,
                                misclf_key=para_misclf_key)
            args.tips = 'fc-freezed'
            print('The correct number of eval predictions for categories A and B is:{}'.format(sum(eval_binary_corrects)))

        # SMiR
        elif args.method in ['feature_rec', 'feature_rec-alpha', 'feature_rec-beta']:
            if args.method == 'feature_rec':
                val_feat_loader = val_binary_feat_map_loader
                eval_sus_feat_loader = eval_sus_binary_feat_map_loader

            elif args.method == 'feature_rec-alpha':
                val_feat_loader = val_binary_feat_loader
                eval_sus_feat_loader = eval_sus_binary_feat_loader

            elif args.method == 'feature_rec-beta':
                criterion = WeightedLoss(error_weights={(1, 0): 30})
                val_feat_loader = val_target_binary_feat_map_loader
                eval_feat_loader = eval_target_binary_feat_map_loader

            val_loss, correct = binary_retrain(model_feature_to_2, args.model, misclf_key, val_feat_loader, seed=seed,
                                               weight=None, epoch=epoch, method=args.method, batchsize=args.batch_size,
                                               set_criterion=args.criterion, lr=args.lr, scheduler=args.scheduler)

            '--evaluation--'
            if args.method in ['feature_rec', 'feature_rec-alpha']:
                # get the overall predictions of Original Model & Validator
                eval_final_labels = \
                    trained_results_for_feature_rec(model_feature_to_2, misclf_key, eval_sus_feat_loader,
                                                    eval_sus_y, eval_sus_indices, eval_prediction_labels,
                                                    eval_y, eval_binary_corrects_orig, eval_target_wrong_orig)
            elif args.method == 'feature_rec-beta':
                (_, eval_binary_labels, _) = \
                    get_predictions(model_feature_to_2, eval_feat_loader, return_bool=True, return_tensor=False,
                                    is_print=False, misclf_key=None)
                eval_final_labels = [misclf_key[0] if binary_label == 1
                                     else eval_prediction_labels[i] for i, binary_label in enumerate(eval_binary_labels)]

        elif args.method == 'feature_rec-gamma':
            args.criterion = DynamicBalanceLoss_v3(error_weights={misclf_key: 1}, k=args.k, t=args.t)
            criterion = args.criterion
            val_loss, correct = weighted_retrain(model_for_dl, misclf_key, val_merge_feat_loader,
                                                 seed, weight=None, epoch=epoch, method=args.method, batchsize=args.batch_size,
                                                 set_criterion=criterion, lr=args.lr, scheduler=args.scheduler)

            (_, eval_final_labels, _) = \
                get_predictions(model_for_dl, eval_merge_feat_loader, return_bool=True, return_tensor=False,
                                is_print=False, misclf_key=None)

        elif args.method == 'idrb':
            break

        if args.method in ['arachne', 'wce', 'retrain', 'binary_retrain', 'idrb']:
            (_, eval_final_labels, _) = \
                get_predictions(pytorch_model, eval_loader, return_bool=True, return_tensor=False, is_print=False,
                                misclf_key=None)

        'evluation process'
        eval_final_corrects = [True for a, b in zip(eval_final_labels, eval_y) if a == b]
        num_of_eval_corr = sum(eval_final_corrects); num_of_eval_wrong = len(eval_final_corrects) - sum(eval_final_corrects)
        num_of_eval_wrong_increased = num_of_eval_wrong - eval_wrong_orig
        num_of_eval_target_wrong = len(get_target_misclf(misclf_key, eval_final_labels, eval_y, is_print=False))

        num_of_repair = len(eval_target_wrong_orig) - num_of_eval_target_wrong
        num_of_introduce = eval_corrects_orig - num_of_eval_corr
        ratio = num_of_repair / (num_of_introduce + 1e-10)

        print(f'IM of eval set{num_of_introduce}, '
              f'repair of eval set{num_of_repair} ')
        wandb.log({"IM of eval set": num_of_introduce,
                   "repair of eval set": num_of_repair})


        if args.rq == 'rq4_safety_critical':
            misclf_dict_curr = get_all_misclf_number(eval_final_labels, eval_y, is_print=True)
            result_dict_safety_critical = compare_misclf(misclf_dict_orig, misclf_dict_curr, train_set)

        'Select the best slice'
        if args.method == 'retrain':
            if best_result > num_of_eval_wrong_increased:
                print('performance improved')
                best_result = num_of_eval_wrong_increased
                best_model = deepcopy(pytorch_model)
        elif args.method == 'binary_retrain':
            if epoch+1 in args.epochs_list:
                args.epochs = epoch + 1
                save_model(best_model, args, path=None)
            if best_result <= sum(eval_binary_corrects):
                print('performance improved')
                best_result = sum(eval_binary_corrects)
                best_model = deepcopy(pytorch_model)

    t2_train = time.time()
    print("\nTraining Time: %f" % (t2_train - t1_train))
    result_dict_v2 = summary_results_v3(best_model, val_set, eval_set, indices_dict, misclf_key, wrong_orig)
    print('overall result of first time:{}'.format(result_dict_v1))
    print('overall result of second time:{}'.format(result_dict_v2))


    # endregion (3)Repair
    """------------------------------(4)Repair------------------------------"""

    """------------------------------(5)Saving patched model------------------------------"""
    # region (5)Saving patched model
    if args.method == 'retrain':
        save_model(best_model, args, path=None)
        result_dict_retrained = summary_results_v3(best_model, val_set, eval_set, indices_dict, misclf_key, wrong_orig)
        print('retrain result:{}'.format(result_dict_retrained))
    elif args.method == 'binary_retrain':
        save_model(best_model, args, path=None)
    elif args.method == 'wce':
        save_model(pytorch_model, args, path=None)

    # endregion (5)Saving patched model

    result_dict_v2 = {'Targeted misclf': f'{misclf_key[0]} → {misclf_key[1]}, namely {misclf_label[0]} → {misclf_label[1]}',
                      'num of all targeted misclf(eval)': len(eval_target_wrong_orig),
                    'num of repair on(eval)': num_of_repair,
                    'num of remain targeted misclf(eval)' : num_of_eval_target_wrong,
                    'num of introducing misclf(eval)': num_of_introduce,
    }
    if args.rq == 'rq4_safety_critical':
        result_dict_v2.update(result_dict_safety_critical)

    return result_dict_v2

if __name__ == '__main__':

    from arguments import parser, get_detail_args
    args = parser.parse_args()

    # get files path
    args.t = 0
    args = get_detail_args(args)

    ex_time = time.strftime("%Y%m%d_%H%M")
    dataset = args.dataset
    model_name = args.model

    args.method = 'feature_rec'

    args.is_loc = False
    args.only_loc = False
    args.loc_method = 'deeplift'

    args.epochs_for_mapping = 10
    args.lr_for_mapping = 0.005
    args.batch_size_for_mapping = 1024

    args.batch_size = 1024
    args.epochs_for_repair = 100
    args.lr = 0.0005

    args.scheduler = CosineScheduler(max_update=100, base_lr=0.0015, final_lr=0.0001,
                                     warmup_steps=5, warmup_begin_lr=0)

    if args.rq in ['rq1', 'rq2', 'rq4_safety_critical', 'rq5_ablation', 'baselines']:
            if args.rq == 'rq4_safety_critical':
                if args.model == 'NEU-CLS-64_CNN':
                    top_n_list = [0, 4, 6]  # for NEU-CLS-64 dataset
                elif args.model == 'APTOS2019_ResNet18':
                    top_n_list = [1, 2, 4]  # for APTOS dataset
            else:
                top_n_list = list(range(3))

            if args.method in ['wce', 'feature_rec-alpha']:
                args.is_loc = False
                args.loc_dimensions = None
            else:
                args.is_loc = True
                args.loc_select_method = 'change_most'

            for args.top_n in top_n_list:
                    args = get_detail_args(args)
                    save_point_print(str(args.criterion))
                    result_list = []
                    seed_list = list(range(5))
                    for args.seed in seed_list:
                        save_point_print('top%d错误的seed=%d试验开始' % (args.top_n, args.seed))
                        result_dict = lhr_main(args)
                        print('current result: ' + str(result_dict))
                        result_list.append(result_dict)
                        save_point_print('top%d错误的seed=%d试验结束' % (args.top_n, args.seed))
                        result_dict['ex_time'] = time.strftime("%Y%m%d_%H%M")
                        result_dict['seed'] = args.seed
                        save_path_csv = os.path.join(args.save_dir_csv, f"{str(ex_time)}_{args.method}_top{args.top_n}_"
                                                                   f"num{args.loc_dimensions}_t{args.t}_k{args.k}seed{seed_list}.csv")
                        if args.method in ['wce', 'idrb']:
                            save_path_csv = os.path.join(args.save_dir_csv, f"{str(ex_time)}_{args.method}_top{args.top_n}_"
                                                                       f"weight{args.weight}_seed{seed_list}.csv")
                        other_utils.save_result_to_excel(save_path_csv, result_dict, args.rq)

                    other_utils.save_point_print('Average result')
                    result_dict_overall = other_utils.get_result_overall(result_list)
                    result_dict_overall['args'] = args
                    result_dict_overall['time'] = ex_time
                    other_utils.save_result_to_excel(save_path_csv, result_dict_overall, args.rq)

    if args.rq == 'rq1_baselines':
        'baseline: idrb and wce'
        # args.method = 'idrb'
        args.method = 'wce'
        # for args.top_n in [0]:
        for args.top_n in range(3):
            # for args.weight in [5, 10, 15, 20, 30, 40, 50, 100, 200, 500, 1000]:
            # for args.weight in [1]:
            result_list = []  # 定义全部实验result_dict组成的list
            # seed_list = list(range(5))
            # seed_list = [0]
            for args.seed in seed_list:
                # seed = top_n  # arachne原始设置: 根据top_n调整seed
                save_point_print('top%d错误的seed=%d试验开始' % (args.top_n, args.seed))
                result_dict = lhr_main(args)  # 运行主函数!!!
                print('current result/当前结果: ' + str(result_dict))
                save_point_print('top%d错误的seed=%d试验结束' % (args.top_n, args.seed))
                # 单次实验结果写入csv文件
                result_dict['ex_time'] = time.strftime("%Y%m%d_%H%M")
                result_dict['seed'] = args.seed
                other_utils.save_result_to_excel(save_path_csv, result_dict, 'rq1_binary_retrain')

    elif args.rq == 'rq1_binary_retrain':
        '用于rq1-2训练专注于AB两类样本的特征提取器'
        args.method = 'binary_retrain'
        args.criterion = nn.CrossEntropyLoss()
        top_n_list = [1, 2, 4]
        # top_n_list = range(3)
        for args.top_n in top_n_list:
            args.weight = 1
            # args.epochs_list = [100, 200]
            args.epochs_list = [200]
            args.epochs_for_repair = max(args.epochs_list)  # 提取epochs序列中的最大值, 一次性训练
            seed_list = [0]
            for args.seed in seed_list:
                # seed = top_n  # arachne原始设置: 根据top_n调整seed
                save_point_print('top%d错误的seed=%d试验开始' % (args.top_n, args.seed))
                result_dict = lhr_main(args)  # 运行主函数!!!
                print('current result/当前结果: ' + str(result_dict))
                save_point_print('top%d错误的seed=%d试验结束' % (args.top_n, args.seed))
                # 单次实验结果写入csv文件
                result_dict['ex_time'] = time.strftime("%Y%m%d_%H%M")
                result_dict['seed'] = args.seed
                save_path_csv = os.path.join(args.save_dir_csv, f"{str(ex_time)}_{args.method}_top{args.top_n}_"
                                                                f"epoch{args.epochs_for_repair}_weight{args.weight}_seed{seed_list}.csv")
                other_utils.save_result_to_excel(save_path_csv, result_dict, 'rq1_binary_retrain')

    elif args.rq == 'rq1_retrain':
        '用于重训练欠拟合的模型'
        args.method = 'retrain'
        args.criterion = nn.CrossEntropyLoss()

        args.weight = 1
        args.epochs_list = [200]
        args.epochs_for_repair = max(args.epochs_list)  # 提取epochs序列中的最大值, 一次性训练
        seed_list = [0]
        for args.seed in seed_list:
            # seed = top_n  # arachne原始设置: 根据top_n调整seed
            save_point_print('top%d错误的seed=%d试验开始' % (args.top_n, args.seed))
            result_dict = lhr_main(args)  # 运行主函数!!!
            print('current result/当前结果: ' + str(result_dict))
            save_point_print('top%d错误的seed=%d试验结束' % (args.top_n, args.seed))
            # 单次实验结果写入csv文件
            result_dict['ex_time'] = time.strftime("%Y%m%d_%H%M")
            result_dict['seed'] = args.seed
            save_path_csv = os.path.join(args.save_dir_csv, f"{str(ex_time)}_{args.method}_top{args.top_n}_"
                                                            f"epoch{args.epochs_for_repair}_weight{args.weight}_seed{seed_list}.csv")
            other_utils.save_result_to_excel(save_path_csv, result_dict, args.rq)

    elif args.rq == 'rq3_feature_recon':
        '仅进行特征重建(定位特征替换)'
        args.is_loc = True
        args.only_loc = True
        args.method = 'feature_rec'
        for  args.top_n in range(3):
        #     dimensions = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90]  # for simple_fm
            # dimensions = [80, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]  # for simple_cm & simple_svhn
            dimensions = [100, 200, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]  # for C10_CNN1 & SVHN_CNN
            for args.loc_dimensions in tqdm(dimensions):
                for args.loc_select_method in ['change_most', 'random', 'bf_most', 'af_most']:
                    result_list = []  # reset results list
                    if args.loc_select_method == 'random':  # 仅对random策略取多次平均值
                        seed_list = list(range(5))
                    else:
                        seed_list = [0]
                    for args.seed in seed_list:
                        save_point_print('top%d错误的seed=%d试验开始' % (args.top_n, args.seed))
                        result_dict = lhr_main(args)  # 运行主函数!!!
                        print('current result/当前结果: ' + str(result_dict))
                        result_list.append(result_dict)  # 实验结果加入list
                        save_point_print('top%d错误的seed=%d试验结束' % (args.top_n, args.seed))
                        '''单次实验结果写入csv文件'''
                        result_dict['ex_time'] = time.strftime("%Y%m%d_%H%M")
                        result_dict['seed'] = args.seed
                        save_path_csv = \
                            os.path.join(args.save_dir_csv, f"{args.method}_{str(ex_time)}_top{args.top_n}_"
                           f"num{args.loc_dimensions}_{args.loc_select_method}_seed{str(seed_list)}.csv")
                        other_utils.save_result_to_excel(save_path_csv, result_dict, 'rq3_feature_recon')

                    '''多次随机实验结果写入csv文件'''
                    other_utils.save_point_print('全部实验平均结果')
                    result_dict_overall = other_utils.get_result_overall(result_list)  # 获得全部实验平均结果
                    result_dict_overall['参数'] = args  # 记录参数
                    result_dict_overall['time'] = ex_time  # 记录实验时间
                    other_utils.save_result_to_excel(save_path_csv, result_dict_overall, args.rq)

    elif args.rq == 'rq3_feature_loc_acc':
        '验证特征定位准确度'
        args.is_loc = True
        args.only_loc = True
        args.method = 'feature_rec'
        for args.top_n in [0]:
            # for args.loc_dimensions in [100]:  # for simple_fm
            for args.loc_dimensions in [1000]:  # for simple_cm & CNN
                for args.loc_select_method in ['change_most', 'random', 'bf_most', 'af_most']:
                # for args.loc_select_method in ['bf_most', 'af_most']:
                #     for args.t_decresed in [3]:
                    for args.t_decresed in [5]:
                        args.p_amount = 1.0  # for most situations
                        args.p_amount = 5.0  # for svhn_CNN to better get five errors
                        result_list = []  # reset results list
                        seed_list = list(range(200))
                        # seed_list = [0, 1]
                        # 获取 seed_list 的范围
                        start = seed_list[0]
                        end = seed_list[-1]
                        # 将范围格式化为字符串
                        seed_range_str = f"{start}-{end}"
                        for args.seed in tqdm(seed_list):
                            save_point_print('top%d错误的seed=%d试验开始' % (args.top_n, args.seed))
                            result_dict = lhr_main(args)  # 运行主函数!!!
                            print('current result/当前结果: ' + str(result_dict))
                            result_list.append(result_dict)  # 实验结果加入list
                            save_point_print('top%d错误的seed=%d试验结束' % (args.top_n, args.seed))

                            '''单次实验结果写入csv文件'''
                            result_dict['ex_time'] = time.strftime("%Y%m%d_%H%M")
                            result_dict['seed'] = args.seed
                            save_path_csv = os.path.join(args.save_dir_csv, f"{args.method}_{str(ex_time)}_top{args.top_n}_"
                                                           f"num{args.loc_dimensions}_t-{args.t_decresed}_"
                                                           f"amount-{args.p_amount}_{args.loc_select_method}_"
                                                           f"seed{str(seed_range_str)}.csv")
                            other_utils.save_result_to_excel(save_path_csv, result_dict, 'rq3_feature_recon')
                            result_dict_overall = other_utils.get_result_overall(result_list)  # 获得当前实验平均结果
                            print(f'current results: {get_overall_loc_results(args.model, result_dict_overall, result_list, args.loc_dimensions)}')

                        '''多次随机实验结果写入csv文件'''
                        other_utils.save_point_print('全部实验平均结果')
                        result_dict_overall = other_utils.get_result_overall(result_list)  # 获得全部实验平均结果
                        result_dict_overall['参数'] = args  # 记录参数
                        result_dict_overall['time'] = ex_time  # 记录实验时间
                        result_dict_overall = get_overall_loc_results(args.model, result_dict_overall,
                                                                      result_list, args.loc_dimensions)
                        other_utils.save_result_to_excel(save_path_csv, result_dict_overall, args.rq)





