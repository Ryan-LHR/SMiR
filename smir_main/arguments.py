# -*-coding:utf-8-*-
import os
import sys
import argparse

from smir.smir_utils.smir_model_train import DynamicBalanceLoss_v3, CosineScheduler


def get_detail_args(args):
    args.save_dir_csv = os.path.join(args.save_dir, "{}/{}/results".format(args.rq, args.model))
    os.makedirs(args.save_dir_csv, exist_ok=True)

    if args.loc_file == None:
        args.save_dir_loc = os.path.join(args.save_dir, "{}/{}/loc".format(args.rq, args.model))
        os.makedirs(args.save_dir_loc, exist_ok=True)

    if args.patched_model_file == None:
        args.patched_model_file = os.path.join(args.save_dir, "{}/patched_model".format(args.model))
        os.makedirs(args.patched_model_file, exist_ok=True)

    index = 1
    stack_data = 1
    if args.indices_file == None:
        data_dir = args.data_dir  # "../lhr_data"
        data_dir_indices = os.path.join(args.data_dir, "models/{}".format(args.model))
        os.makedirs(data_dir_indices, exist_ok=True)  # e.g.'../lhr_data\\models/simple_fm'

        args.indices_file = os.path.join(data_dir_indices, "{}_indices.csv".format(args.model))
        # if args.on_train:
        args.indices_file_onTrain = os.path.join(data_dir_indices, "{}_indices_onTrain.csv".format(args.model))


        retrained = True
        if retrained:
            args.indices_file = os.path.join(data_dir_indices, "{}_retrained_indices.csv".format(args.model))
            args.indices_file_onTrain = os.path.join(data_dir_indices, "{}_retrained_indices_onTrain.csv".format(args.model))
            print(args.indices_file)

    if args.model == 'simple_fm':
        args.model_file = '../smir_data/models/simple_fm/fm_simple_withforward2.pth'
        args.model_feature_to_n_file ='../lhr_data/models/simple_fm/fm_simple_withforward2_feature_to_n.pth'
        args.loc_dimensions = 3
        args.k = 4.5
        if args.rq != 'rq2':
            if args.top_n is not None:
                if args.top_n == 0:
                    args.t = 0.55
                elif args.top_n == 1:
                    args.t = 0.3
                elif args.top_n == 2:
                    args.t = 0.3
        args.criterion = DynamicBalanceLoss_v3(error_weights={(1, 0): 1}, k=args.k, t=args.t)
        args.scheduler = CosineScheduler(max_update=100, base_lr=0.015, final_lr=0.0001,
                                         warmup_steps=5, warmup_begin_lr=0)
    elif args.model == 'simple_cm':
        args.model_file = '../lhr_data/models/simple_cm/cm_simple_withforward2_patched_by_retrain_seed_deeplift.pth'
        args.model_feature_to_n_file = '../lhr_data/models/simple_cm/cm_simple_withforward2_patched_by_retrain_seed_deeplift_test1.pth'
        args.loc_dimensions = 80
        args.scheduler = CosineScheduler(max_update=100, base_lr=0.0015, final_lr=0.0001,
                                         warmup_steps=5, warmup_begin_lr=0)  # simple_cm最佳？？
        args.k = 3.5
        if args.rq != 'rq2':
            args.t = 0.5
        args.criterion = DynamicBalanceLoss_v3(error_weights={(1, 0): 1}, k=args.k, t=args.t)

    elif args.model == 'C10_CNN1':
        args.model_file = '../lhr_data/models/C10_CNN1/C10_CNN1_withforward2_patched_by_retrain_seed_deeplift.pth'
        args.model_feature_to_n_file = '../lhr_data/models/C10_CNN1/C10_CNN1_withforward2_patched_by_retrain_seed_deeplift_feature_to_n.pth'

        args.loc_dimensions = 200
        args.scheduler = CosineScheduler(max_update=100, base_lr=0.0005, final_lr=0.0001,
                                         warmup_steps=0, warmup_begin_lr=0)  # c10-CNN1尝试
        args.k = 4
        if args.rq != 'rq2':
            if args.top_n is not None:
                if args.top_n == 0:
                    args.t = 0.4
                elif args.top_n == 1:
                    args.t = 0.3
                elif args.top_n == 2:
                    args.t = 0.05
        args.criterion = DynamicBalanceLoss_v3(error_weights={(1, 0): 1}, k=args.k, t=args.t)

    elif args.model == 'simple_svhn':
        args.model_file = '../lhr_data/models/simple_svhn/simple_svhn_withforward2_acc(0.8715).pth'
        args.model_feature_to_n_file = '../lhr_data/models/simple_svhn/simple_svhn_withforward2_acc(0.8715)_feature_to_n.pth'
        args.loc_dimensions = 80
        args.scheduler = CosineScheduler(max_update=100, base_lr=0.001, final_lr=0.0001,
                                         warmup_steps=5, warmup_begin_lr=0)
        args.k = 1.5
        if args.rq != 'rq2':
            if args.top_n is not None:
                if args.top_n == 0:
                    args.t = 0.1
                elif args.top_n == 1:
                    args.t = 0.05
                elif args.top_n == 2:
                    args.t = 0.05
        args.criterion = DynamicBalanceLoss_v3(error_weights={(1, 0): 1}, k=args.k, t=args.t)

    elif args.model == 'SVHN_CNN1':
        args.model_file = '../lhr_data/models/SVHN_CNN1/svhn_cnn1_acc(0.9317).pth'
        args.model_feature_to_n_file = '../lhr_data/models/SVHN_CNN1/svhn_cnn1_acc(0.9317)_feature_to_n.pth'
        args.loc_dimensions = 200
        args.scheduler = CosineScheduler(max_update=100, base_lr=0.005, final_lr=0.0001,
                                         warmup_steps=5, warmup_begin_lr=0)  # simple_cm最佳？？
        args.k = 2
        if args.rq != 'rq2':
            if args.top_n is not None:
                if args.top_n == 0:
                    args.t = 0.15
                elif args.top_n == 1:
                    args.t = 0.05
                elif args.top_n == 2:
                    args.t = 0.05
        args.criterion = DynamicBalanceLoss_v3(error_weights={(1, 0): 1}, k=args.k, t=args.t)

    elif args.model == "NEU-CLS-64_CNN":
        args.model_file = '../lhr_data/models/NEU-CLS-64_CNN/NEU-CLS-64_CNN_withoutDropout_acc(0.9217).pth'
        args.model_feature_to_n_file = '../lhr_data/models/NEU-CLS-64_CNN/NEU-CLS-64_CNN_withoutDropout_acc(0.9217)_feature_to_n.pth'
        if args.rq != 'rq2':
            if args.top_n is not None:
                if args.top_n == 0:
                    args.scheduler = CosineScheduler(max_update=100, base_lr=0.001, final_lr=0.0001,
                                                     warmup_steps=5, warmup_begin_lr=0)
                    args.criterion = DynamicBalanceLoss_v3(error_weights={(1, 0): 1}, k=3, t=0.5)
                elif args.top_n == 4:
                    args.scheduler = CosineScheduler(max_update=100, base_lr=0.005, final_lr=0.0001,
                                                     warmup_steps=5, warmup_begin_lr=0)
                    args.criterion = DynamicBalanceLoss_v3(error_weights={(1, 0): 1}, k=3, t=0.65)
                elif args.top_n == 6:
                    args.scheduler = CosineScheduler(max_update=100, base_lr=0.005, final_lr=0.0001,
                                                     warmup_steps=5, warmup_begin_lr=0)
                    args.criterion = DynamicBalanceLoss_v3(error_weights={(1, 0): 1}, k=3, t=1.2)

    elif args.model == 'APTOS2019_ResNet18':
        args.model_file = '../lhr_data/models/APTOS2019_ResNet18/APTOS2019_resnet18_new_withDropout_acc(0.6576).pth'
        args.model_feature_to_n_file = '../lhr_data/models/APTOS2019_ResNet18/APTOS2019_resnet18_new_withDropout_acc(0.6576)_feature_to_n.pth'
        args.batch_size = 128
        args.scheduler = CosineScheduler(max_update=100, base_lr=0.0005, final_lr=0.0001,
                                         warmup_steps=5, warmup_begin_lr=0)
        args.k = 2
        if args.top_n is not None:
            if args.top_n == 1:
                args.t = 1.2
            elif args.top_n == 2:
                args.t = 2
            elif args.top_n == 4:
                args.t = 4
        args.criterion = DynamicBalanceLoss_v3(error_weights={(1, 0): 1}, k=args.k, t=args.t)
        args.loc_dimensions = 40
    else:
        raise ValueError("model_for_dl is not set")

    if args.method == 'wce':
        args.epochs_for_repair = 50
        args.gamma = 5.0

        if args.model == 'simple_svhn':
            args.epochs_for_repair = 5  # for better performance

        if args.rq != 'rq2':
            if args.model == 'simple_fm':
                args.weight = 15
            elif args.model == 'simple_cm':
                args.weight = 50
            elif args.model == 'C10_CNN1':
                args.weight = 10
            elif args.model == 'simple_svhn':
                if args.top_n is not None:
                    if args.top_n == 0:
                        args.weight = 100  # for better performance
                    elif args.top_n == 1:
                        args.weight = 10
                    elif args.top_n == 2:
                        args.weight = 20
            elif args.model == 'SVHN_CNN1':
                args.weight = 50
            elif args.model == 'NEU-CLS-64_CNN':
                args.epochs_for_repair = 20
                args.weight = 50
            elif args.model == 'APTOS2019_ResNet18':
                args.weight = 50

    return args


parser = argparse.ArgumentParser()
parser.add_argument("-seed", action="store", default=0, type=int)  # random seed
parser.add_argument("-top_n", action="store", default=0, type=int)  # top n 误分类
parser.add_argument("-rq", action="store", default='arachne_rq3',
                    type=str, help='arachne_rq2, arachne_rq3')  # 执行哪个rq实验
parser.add_argument("-loc_method", action="store", default='random',
                    help='random, mrc_likewise')  # 定位方法
parser.add_argument("-data_dir", action="store", type=str)  # 加载dataset, model, indices的根目录
parser.add_argument("-dataset_dir", action="store", type=str)  # 加载dataset的目录
parser.add_argument("-save_dir", action="store", type=str)  # 存储results, patched_model的根目录
parser.add_argument("-dataset", action="store", default='fashion_mnist',
                    type=str, help='fashion_mnist, cifar_10, GTSRB')  # which dataset
parser.add_argument("-model", action="store", default=None, type=str)  # which model
parser.add_argument("-indices_file", action="store", default=None, type=str)  # predictions indices path
parser.add_argument("-loc_file", action="store", default=None, type=str)  # loc file path
parser.add_argument("-save_dir_loc", action="store", default=None, type=str)  # save_dir_loc
parser.add_argument("-patched_model_file", action="store", default=None, type=str)  # patched model file path
parser.add_argument("-batch_size", type=int, default=None)
parser.add_argument("-scheduler", type=int, default=None)
parser.add_argument("-weight", type=int, default=10)  # train weight for target misclf
parser.add_argument("-target_layer_name", action="store",
                    default='All', type=str, help="an index to the layer to localiser nws")

parser.add_argument("-on_train", action="store", default=True, type=str)  # 是否使用训练集数据进行训练
methods = ['lhr', 'arachne', 'apricot', 'wce', 'focal_loss',
           'retrain', 'lhr_retrain', 'feature_rec', 'idrb']
parser.add_argument("-method", action="store", default='lhr', type=str, choices=methods)

parser.add_argument("-setmethod", action="store", default='lhr', type=str)  # 临时参数