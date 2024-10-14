"""
convert models and model utils
"""
import numpy as np
import torch
import time

from smir.smir_utils.other_utils import save_point_print, save_to_pickle, read_from_pickle
from torch.utils.data import DataLoader, Subset
from smir.smir_utils.model_stru import *

def get_pytorch_weights(model, target_paras='All'):
    print('\nget_pytorch_weights')
    weights_dict = {}
    for name, para in model.named_parameters():
        if target_paras == 'All' or name in target_paras:
            weights_dict[name] = para
            print(name, '', para.shape)
    print('get weights finished')
    return weights_dict


def get_para_from_dict(weights_dict, para_name):
    target_para = weights_dict[para_name].data
    return target_para

def get_para_from_model(model, para_name):
    temp = model
    attr_list = para_name.split('.')
    for attr in attr_list:
        assert hasattr(temp, attr), "Not found target parameter: {}".format(attr)
        temp = getattr(temp, attr)
    target_para = temp.data
    return target_para

def set_pytorch_para(model, para_name, para):
    temp = model
    attr_list = para_name.split('.')
    attr_list.append('data')
    obj_dict = {}
    parent_dict = {}
    parent_attr_name = 'model'
    for attr in attr_list:
        assert hasattr(temp, attr), "Not found target parameter: {}".format(attr)
        temp = getattr(temp, attr)

        obj_dict[attr] = temp
        parent_dict[attr] = parent_attr_name
        parent_attr_name = attr

    attr_list.reverse()
    temp_obj = para
    for attr in attr_list:
        parent_attr_name = parent_dict[attr]
        assert hasattr(obj_dict[parent_attr_name], attr), "Not found target parameter: {}".format(attr)
        setattr(obj_dict[parent_attr_name], attr, temp_obj)
        temp_obj = obj_dict[attr]

    print('set weights finished')
    return model


def get_keras_weights(model, target_layers='All'):
    print('\nmodel summary: '); print(model.summary())
    weights_dict = {}  # key = layer index, value = [layer name, [weight, bias]]
    for i, layer in enumerate(model.layers):
        class_name = type(layer).__name__
        print('layer type o`f index %d is: %s' % (i, class_name))
        if target_layers== 'All' or i in target_layers:
            try:
                weight = layer.get_weights()
                weights_dict[i] = [class_name, weight]
                if weight != []:  # has weight
                    print('get weight\n')
                else:
                    print('get empty weight\n')
            except:
                print('can not get weight\n')

    print('get weights finished')
    return weights_dict

def save_patched_keras_model(indices_to_places_to_fix, best_weight, patched_weight):
    for index, item in enumerate(indices_to_places_to_fix):
        indice_to_weight = item[1]
        (i, j) = indice_to_weight
        patched_weight[i][j] = best_weight[index]
    return patched_weight

def get_preditions_vectors (model_type, test_data, model_path=None, model=None,  from_pytorch=False, is_print=True):
    data_X, data_y = test_data
    if model_type == 'keras':
        from tensorflow.keras.models import load_model
        # from keras.models import load_model
        if model == None:
            model = load_model(model_path, compile=False)
        model._make_predict_function()
        predictions = model.predict(data_X)
        print('Keras prediction for first input: ' + str(predictions[0]))

    elif model_type == 'onnx':
        import onnx
        import onnxruntime as ort

        providers = ['CPUExecutionProvider']
        session = ort.InferenceSession(model_path, providers=providers)

        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        if from_pytorch == False:
            predictions = session.run([output_name], {input_name: data_X})

        else:
            import numpy as np
            predictions = []

            for images in data_X:
                prediction = session.run([output_name], {input_name: images})
                predictions.append(prediction)
            predictions = np.asarray(predictions)
        print('Onnx prediction for first input: ' + str(predictions[0]))

    elif model_type == 'pytorch':
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model == None:
            model = torch.load(model_path, map_location=torch.device(device))
        # model = torch.load(model_path, map_location=torch.device('cpu'))

        data_X_tensor = data_X
        predictions = []
        for images in data_X_tensor:
            # if torch.cuda.is_available():
            images = images.to(device)
            prediction = model(images).detach()
            prediction = prediction.cpu().numpy()
            # prediction = prediction.numpy()
            prediction = prediction[0]
            predictions.append(prediction)

        import numpy as np
        predictions = np.asarray(predictions)

        if is_print:
            print('Pytorch prediction for first input: ' + str(predictions[0]))
    return predictions

def get_preditions_labels (prediction_vectors, test_data, is_print=True):
    if is_print:
        print('\nget_preditions_label')
        print('prediction_vectors.shape: ' + str(prediction_vectors.shape))
    import numpy as np
    data_X, data_y = test_data
    predictions = prediction_vectors

    if len(prediction_vectors.shape) >= 3:
        predictions = predictions.reshape((predictions.shape[0], predictions.shape[-1]))
    correct_labels = data_y

    if len(predictions.shape) >= 2 and predictions.shape[-1] > 1:
        prediction_labels = np.argmax(predictions, axis=1)
    else:
        prediction_labels = np.round(predictions).flatten()

    if torch.is_tensor(correct_labels[0]):
        correct_labels = [i.item() for i in correct_labels]

    prediction_bool = prediction_labels == correct_labels

    if is_print:
        print('prediction label for first input: ' + str(prediction_labels[0]))
        print('prediction correctness for first input: ' + str(prediction_bool[0]))
        print('num of correctness predictions: ' + str(sum(prediction_bool)))
    return (prediction_labels, prediction_bool)

def get_correctness_number (model_type, model, test_data):
    # print('\nget_correctness_number')
    if model_type == 'pytorch':
        predictions_vectors = get_preditions_vectors('pytorch', test_data, model=model, is_print=False)
        prediction_labels, prediction_bool = get_preditions_labels(predictions_vectors, test_data, is_print=False)

    all_number = len(prediction_bool)
    correct_number = sum(prediction_bool)
    wrong_number = all_number - correct_number
    return (all_number, correct_number, wrong_number)


def summary_results_v3(model, val_set, eval_set, indices_dict, misclf_key, wrong_orig):
    val_wrong_orig, eval_wrong_orig = wrong_orig

    val_loader = DataLoader(val_set, batch_size=64)
    (num_of_val_all, _, num_of_val_wrong) = \
        get_predictions(model, val_loader, return_bool=True, return_number=True, is_print=False)
    num_of_val_wrong_increased = num_of_val_wrong - val_wrong_orig

    eval_loader = DataLoader(eval_set, batch_size=64)
    (num_of_eval_all, _, num_of_eval_wrong) = \
        get_predictions(model, eval_loader, return_bool=True, return_number=True, is_print=False)
    num_of_eval_wrong_increased = num_of_eval_wrong - eval_wrong_orig

    val_target_loader = DataLoader(Subset(val_set, indices_dict['val_indices_to_wrong']), batch_size=64)
    (val_target, val_target_patched, val_target_wrong) = \
        get_predictions(model, val_target_loader, return_bool=True, return_number=True, is_print=False)

    eval_target_set = Subset(eval_set, indices_dict['eval_indices_to_wrong'])
    eval_target_loader = DataLoader(eval_target_set, batch_size=64)
    (_, eval_target_labels, eval_target_correct) = \
        get_predictions(model, eval_target_loader, return_bool=True, return_tensor=False, is_print=False)

    eval_target_se = get_target_misclf(misclf_key, eval_target_labels,
                                       [label for _, label in eval_target_set], is_print=False)

    eval_y = [label for _, label in eval_set]
    (_, eval_labels, _) = get_predictions(model, eval_loader, return_bool=True, return_tensor=False, is_print=False)
    eval_se = get_target_misclf(misclf_key, eval_labels, eval_y, is_print=False)

    result_dict = {'Initial specific misclassifications in val': val_target, \
                   'Specific misclassifications fixed in val': val_target_patched, \
                   'Wrong classifications in val': num_of_val_wrong, \
                   'Introduced misclassifications in val': num_of_val_wrong_increased, \
                   'Initial specific misclassifications in eval (original indices)': len(eval_target_set), \
                   'Misclassifications fixed in eval (original indices)': sum(eval_target_correct),
                   'Introduced misclassifications in eval': num_of_eval_wrong_increased,
                   'Remaining misclassifications in eval (original indices)': len(eval_target_set) - sum(
                       eval_target_correct), \
                   'Remaining specific misclassifications in eval (original indices)': len(eval_target_se), \
                   'Total specific misclassifications in eval': len(eval_se),
                   'Specific misclassifications outside original indices in eval': len(eval_se) - len(eval_target_se),
                   'Introduced misclassifications in eval': num_of_eval_wrong_increased
                   }
    return result_dict

def trained_results_for_feature_rec(model_feature_to_2, misclf_key, eval_feat_loader,
                                    eval_sus_y, eval_sus_indices, eval_labels, eval_y,
                                    eval_binary_corrects_orig, eval_target_now_indices_orig):
    (eval_binary_vectors, eval_binary_labels, _) = \
        get_predictions(model_feature_to_2, eval_feat_loader, return_bool=True, return_tensor=False,
                        is_print=False, misclf_key=None)
    eval_binary_labels = convert_list(eval_binary_labels, misclf_key)
    eval_binary_corrects = [True for a, b in zip(eval_binary_labels, eval_sus_y) if a == b]

    replace_dict = dict(zip(eval_sus_indices, eval_binary_labels))

    eval_final_labels = [replace_dict[i] if i in replace_dict else label for i, label in enumerate(eval_labels)]

    eval_target_now_indices = get_target_misclf(misclf_key, eval_final_labels, eval_y, is_print=False)
    print(f'The eval set introduces misclassification as{sum(eval_binary_corrects_orig) - sum(eval_binary_corrects)}, '
          f'The number of specific misclassification fixes in the eval set is{len(eval_target_now_indices_orig) - len(eval_target_now_indices)}, '
          f'The number of remaining specific misclassification:{len(eval_target_now_indices)}')
    return eval_final_labels

def save_indices_by_keras(model_path, save_path, data):
    # from tensorflow import keras
    # from keras.models import load_model
    from tensorflow.keras.models import load_model
    from lhr.lhr_utils.lhr_data_utils import  save_precditions

    predictions_keras = get_preditions_vectors('keras', data, model_path=model_path) #验证keras模型的结果
    prediction_labels, prediction_bool = get_preditions_labels(predictions_keras, data)

    data_y = data[1]
    save_precditions(save_path, prediction_labels, data_y)
    return

def get_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def get_predictions(model, dataloader, return_bool=True, return_tensor=True,
                    return_number=False, is_print=True, misclf_key=None, return_list=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    prediction_vectors = []
    prediction_labels = []
    prediction_correct = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            prediction_vectors.extend(outputs)
            prediction_labels.extend(predicted)

            if return_bool:
                correct = (predicted == labels)
                prediction_correct.extend(correct)

        if misclf_key != None:
            misclf_orig, misclf_pred = misclf_key

            prediction_labels_tensor = torch.tensor(prediction_labels, device=device)
            indices = (prediction_labels_tensor == misclf_pred).nonzero(as_tuple=True)[0]
            indices_list = indices.tolist()

            dataset = dataloader.dataset
            for idx in indices_list:
                input_curr, label_curr = dataset.__getitem__(idx)
                label_curr = torch.tensor(label_curr)
                input_curr, label_curr = input_curr.to(device), label_curr.to(device)
                input_curr = input_curr.unsqueeze(0)
                output = model.forward_v2(input_curr)
                _, predicted = torch.max(output, 1)
                if predicted == 1:
                    predicted = misclf_orig
                    prediction_vectors[idx] = output
                    prediction_labels[idx] = predicted
                    prediction_correct[idx] = predicted == label_curr
    if return_list:
        return indices_list

    if not return_tensor:
        prediction_labels = [t.item() for t in prediction_labels]
        prediction_correct = [t.item() for t in prediction_correct]

    if is_print:
        print('\nPrediction vector for first input: ' + str(prediction_vectors[0]))
        print('prediction label for first input: ' + str(prediction_labels[0]))
        print('prediction correctness for first input: ' + str(prediction_correct[0]))
        print('num of correctness predictions: ' + str(sum(prediction_correct)))

    all_number = len(prediction_correct)
    correct_number = int(sum(prediction_correct))
    wrong_number = all_number - correct_number
    number = (all_number, correct_number, wrong_number)
    if not return_number:
        return (prediction_vectors, prediction_labels, prediction_correct)
    elif return_number == 'Both':
        return (prediction_vectors, prediction_labels, prediction_correct, number)
    else:
        return (all_number, correct_number, wrong_number)

def get_predictions_binary(model, dataloader, return_bool=True, return_tensor=True,
                    return_number=False, is_print=True, misclf_key=None, return_list=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    prediction_vectors = []
    prediction_labels = []
    prediction_correct = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model.forward_v2(inputs)
            _, predicted = torch.max(outputs, 1)

            prediction_vectors.extend(outputs)
            prediction_labels.extend(predicted)

            if return_bool:
                correct = (predicted == labels)
                prediction_correct.extend(correct)

    if not return_tensor:
        prediction_labels = [t.item() for t in prediction_labels]
        prediction_correct = [t.item() for t in prediction_correct]

    if is_print:
        print('\nPrediction vector for first input: ' + str(prediction_vectors[0]))
        print('prediction label for first input: ' + str(prediction_labels[0]))
        print('prediction correctness for first input: ' + str(prediction_correct[0]))
        print('num of correctness predictions: ' + str(sum(prediction_correct)))

    all_number = len(prediction_correct)
    correct_number = int(sum(prediction_correct))
    wrong_number = all_number - correct_number
    number = (all_number, correct_number, wrong_number)
    if not return_number:
        return (prediction_vectors, prediction_labels, prediction_correct)
    elif return_number == 'Both':
        return (prediction_vectors, prediction_labels, prediction_correct, number)
    else:
        return (all_number, correct_number, wrong_number)

def convert_list(input_list, misclf_key):
    return [misclf_key[0] if num == 1 else misclf_key[1] for num in input_list]

def get_target_misclf(misclf_key, prediction_labels, true_labels, is_print=True):
    indices = []
    for i, label_tuple in enumerate(zip(true_labels, prediction_labels)):
        if label_tuple == misclf_key:
            indices.append(i)
    if is_print:
        print('num of target misclf:{}'.format(len(indices)))
    return indices

def get_all_misclf_number(prediction_labels, true_labels, is_print=True):
    misclf_counts = {}
    for pred_label, true_label in zip(prediction_labels, true_labels):
        if pred_label != true_label:
            misclf_key = (true_label, pred_label)
            if misclf_key in misclf_counts:
                misclf_counts[misclf_key] += 1
            else:
                misclf_counts[misclf_key] = 1
    misclf_counts = {k: misclf_counts[k] for k in sorted(misclf_counts)}
    return misclf_counts

def compare_misclf(misclf_bf_patch, misclf_af_patch, dataset):
    misclf_changed = {}

    for key in misclf_af_patch:
        if key in misclf_bf_patch:
            change = misclf_af_patch[key] - misclf_bf_patch[key]
            if change != 0:
                misclf_changed[key] = change
        else:
            misclf_changed[key] = misclf_af_patch[key]

    for key in misclf_bf_patch:
        if key not in misclf_af_patch:
            misclf_changed[key] = -misclf_bf_patch[key]
    misclf_changed = {key: misclf_changed[key] for key in sorted(misclf_changed)}

    misclf_scm_changed, misclf_nscm_changed = {}, {}
    total_scm, total_nscm = 0, 0
    for key in misclf_changed:
        if is_scm(key, dataset):
            misclf_scm_changed[key] = misclf_changed[key]
            total_scm += misclf_changed[key]
        else:
            misclf_nscm_changed[key] = misclf_changed[key]
            total_nscm += misclf_changed[key]
    print(f'{total_scm} new safety-critical misclassifications introduced, details: {misclf_scm_changed}')
    print(f'{total_nscm} new non-safety-critical misclassifications introduced, details: {misclf_nscm_changed}')
    result_dict = {'New safety-critical misclassifications': total_scm,
                   'Details of safety-critical misclassification changes': misclf_scm_changed,
                   'New non-safety-critical misclassifications': total_nscm,
                   'Details of non-safety-critical misclassification changes': misclf_nscm_changed, }

    return result_dict

def save_model(model, args, path=None):
    import os
    _, filename = os.path.split(args.model_file)
    basename, extension = os.path.splitext(filename)
    if args.method == 'retrain':
        filename = basename + '_patched_by_{}_seed{}'.format(args.method, args.seed) + extension

    elif args.method == 'binary_retrain':
        filename = basename + f'_patched_by_{args.method}_{args.misclf_key}_seed{args.seed}_' \
                              f'epochs{args.epochs}_weight{args.weight}_{args.tips}' + extension
    else:
        filename = basename + '_patched_by_{}_{}_seed{}'.format(args.method, args.misclf_key, args.seed) + extension

    path = os.path.join(args.patched_model_file, filename)

    torch.save(model, path)
    print('\nmodel saved in: {}'.format(path))

def is_scm(misclf_key, dataset, is_print=False):
    from collections import Counter
    class_to_idx = dataset.class_to_idx
    class_with_decre_severity = dataset.class_with_decre_severity

    idx_to_class = {v: k for k, v in class_to_idx.items()}

    class_orig = [k for k, v in class_to_idx.items() if v == misclf_key[0]][0]
    class_pred = [k for k, v in class_to_idx.items() if v == misclf_key[1]][0]

    severity_index_orig = class_with_decre_severity.index(class_orig)
    severity_index_pred = class_with_decre_severity.index(class_pred)

    is_scm = severity_index_orig < severity_index_pred
    if is_print:
        if is_scm:
            print(f'misclf:{misclf_key[0]}({class_orig}) → {misclf_key[1]}({class_pred})is SCM')
        else:
            print(f'misclf{misclf_key[0]}({class_orig}) → {misclf_key[1]}({class_pred})is not SCM')
        dataloader = DataLoader(dataset.dataset, batch_size=1, shuffle=False)

        label_counts = Counter()
        for data in dataloader:
            _, labels = data
            label_counts.update(labels.tolist())

        for idx, count in label_counts.items():
            class_name = idx_to_class[idx]
            print(f"Class '{class_name}' (Index {idx}): {count} samples")
    return is_scm

def build_model_feature_to_2(model_name, device):
    if model_name == 'simple_fm':
        model_feature_to_2 = fm_simple_Net_feature_to_2_v2().to(device)
    elif model_name in ['simple_cm', 'simple_svhn']:
        model_feature_to_2 = c10_simple_Net_feature_to_2().to(device)
    elif model_name in ['C10_CNN1', 'SVHN_CNN1']:
        model_feature_to_2 = C10_CNN1_Net_feature_to_2().to(device)
    elif model_name == 'NEU-CLS-64_CNN':
        model_feature_to_2 = neucls64_Net_feature_to_2().to(device)
    elif model_name == 'APTOS2019_ResNet18':
        model_feature_to_2 = aptos_resnet18_Net_new_feature_to_2().to(device)
    else:
        raise ValueError("path_mdl_after_retrain is not set")
    return model_feature_to_2



