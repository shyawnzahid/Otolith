import torch
import torchvision
import math
import pickle
import os

from datasets import OtolithAgesDataset
from PIL import Image
from torcheval.metrics import R2Score
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot as plt

# Data structure for storing metrics for an epoch.
class BestEpoch():
    def __init__(self):
        self.data = {
            'default': {
                'targets': None,
                'preds': None,
                'acc': -100,
                'loss': 1000,
                'within_1_acc': 0,
                'within_3_acc': 0,
                'within_5_acc': 0,
                'reread_trigger_acc': 0
            },
            'rounded': {
                'targets': None,
                'preds': None,
                'acc': -100,
                'loss': 1000,
                'within_1_acc': 0,
                'within_3_acc': 0,
                'within_5_acc': 0,
                'reread_trigger_acc': 0
            }
        }

# Data structure for storing metrics for both default and rounded outputs.
class Metric():
    def __init__(self, default=0, rounded=0, epoch=0):
        self.default = default
        self.default_targets = None
        self.default_preds = None

        self.rounded = rounded
        self.rounded_targets = None
        self.rounded_preds = None

        self.epoch = epoch

# Data

def denormalize(normalized, age_normalization, model_type):
    # Denormalize ages from zero mean, unit variance ages (regression) or class labels (classification) to actual age values in years
    if model_type == 'regression':
        std, mean = age_normalization

        return normalized * std + mean
    else:
        min_age = age_normalization

        return normalized + min_age

def read_img(fp, mode='RGB'):
    img = Image.open(fp)
    img = img.convert(mode)

    return img

def load_data(ages, testing, data_path, load_masks):
    # Read in images and ages from data_path
    test_suffix = 'test/' if testing else ''
    img_data = []
    age_data = []
    bg_mask_data = [] if load_masks else None
    to_tensor = torchvision.transforms.ToTensor()

    for file_name in sorted(os.listdir(f'{data_path}/{test_suffix}')):
        if file_name == 'test':
            continue

        index = int(file_name.split('.')[0])

        if index not in ages:
            continue

        age = ages[index]
        if math.isnan(age):
            continue
        age = torch.as_tensor(age, dtype=torch.float)

        f = f'{data_path}/{test_suffix}{file_name}'
        img = read_img(f)
        img = to_tensor(img)

        img_data.append(img)
        age_data.append(age)

        if bg_mask_data is not None:
            f_bg_mask = f'{data_path}_bg_mask/{test_suffix}{file_name}'
            bg_mask = read_img(f_bg_mask)
            bg_mask = torch.mean(to_tensor(bg_mask), dim=0).bool()
            bg_mask_data.append(bg_mask)
            
    img_data = torch.stack(img_data, dim=0)
    age_data = torch.stack(age_data, dim=0)

    if bg_mask_data is not None:
        bg_mask_data = torch.stack(bg_mask_data, dim=0)

    return img_data, age_data, bg_mask_data

def oversampled(train_img_data, train_age_data, train_bg_mask_data, threshold=10):
    # Editing data tensors to account for oversampling.
    img_data = list(torch.unbind_copy(train_img_data, dim=0))
    age_data = list(torch.unbind_copy(train_age_data, dim=0))
    bg_mask_data = None if train_age_data is None else list(torch.unbind_copy(train_bg_mask_data, dim=0))

    ages, counts = torch.unique(train_age_data, sorted=True, return_counts=True)
    oversample_ages = ages[counts < threshold]
    oversample_counts = counts[counts < threshold]
    oversample_dict = dict(zip(oversample_ages.tolist(), oversample_counts.tolist()))

    oversampled_imgs = []
    oversampled_ages = []
    oversampled_bg_masks = None if bg_mask_data is None else []

    for idx, age_tensor in enumerate(train_age_data):
        age = age_tensor.item()
        if age in oversample_dict:
            count = math.ceil(threshold/oversample_dict[age]) - 1
            oversampled_imgs += [img_data[idx]]*count
            oversampled_ages += [age_data[idx]]*count

            if oversampled_bg_masks is not None:
                oversampled_bg_masks += [bg_mask_data[idx]]*count
    
    img_data += oversampled_imgs
    img_data = torch.stack(img_data, dim=0)

    age_data += oversampled_ages
    age_data = torch.stack(age_data, dim=0)

    if bg_mask_data is not None:
        bg_mask_data += oversampled_bg_masks
        bg_mask_data = torch.stack(bg_mask_data, dim=0)

    return img_data, age_data, bg_mask_data

def make_test_dataset(base_directory, bg_transform, model_type, age_normalization, cuda, max_label):
    # Create test dataset for testing trained models.
    img_data = None
    age_data = None
    bg_mask_data = None

    label_dir = base_directory + '/saved_labels' + '/OA_labels.pkl'

    with open(label_dir, 'rb') as f:
        ages = pickle.load(f)
    
    data_path = base_directory + '/data/OA'

    load_masks = not (bg_transform == 'orig')

    img_data, age_data, bg_mask_data = load_data(
        ages=ages,
        testing=True,
        data_path=data_path,
        load_masks=load_masks
    )

    if model_type == 'regression':
        # Normalize ages to zero mean and unit variance for regression model.
        std, mean = age_normalization
        age_data = (age_data - mean) / std
    elif model_type == 'classification':
        # Convert ages into class labels by subtracting the minimum age for classification model.
        min_age = age_normalization
        age_data = (age_data - min_age).byte()
        age_data = torch.where(age_data > max_label, max_label, age_data)
    
    if cuda:
        img_data = img_data.to('cuda')
        age_data = age_data.to('cuda')

        if bg_mask_data is not None:
            bg_mask_data = bg_mask_data.to('cuda')

    dataset = OtolithAgesDataset(
        img_data=img_data,
        age_data=age_data,
        bg_mask_data=bg_mask_data,
        augmentations=(False, False, False, False), # No data augmentation for test dataset.
        bg_transform=bg_transform
    )

    return dataset


def make_datasets(base_directory, bg_transform, augmentations=(False, False, False, False), cuda=False, oversample=False, model_type='regression', fold=0, testing=False, oversample_threshold=10):
    # Load in training and validation dataset.
    img_data = None
    age_data = None

    label_dir = base_directory + '/saved_labels'
    OA_label_dir = label_dir + '/OA_labels.pkl'

    with open(OA_label_dir, 'rb') as f:
        OA_ages = pickle.load(f)

    img_data = None
    age_data = None
    bg_mask_data = None
    
    load_masks = not (bg_transform == 'orig')

    img_data, age_data, bg_mask_data = load_data(
        ages=OA_ages,
        testing=False,
        data_path=f'{base_directory}/data/OA',
        load_masks=load_masks
    )

    # 5 - fold validation for training. Select one fold for validation data, remaining 4 fold form training data.
    kfold = StratifiedKFold(
        n_splits=5,
        shuffle=False
    )

    indices = torch.arange(img_data.size(0))
    folds = kfold.split(
        X=indices,
        y=age_data
    )
    folds = list(folds)

    train_indices, val_indices = folds[fold]
    
    out_features = None
    max_label = None
    if model_type == 'regression':
        # Normalize ages to zero mean and unit variance for regression model.
        std, mean = torch.std_mean(age_data)
        age_data = (age_data - mean) / std

        age_normalization = (std, mean)
        out_features = 1
    elif model_type == 'classification':
        # Convert ages into class labels by subtracting the minimum age for classification model.
        min_age = torch.min(age_data)
        age_data = (age_data - min_age).byte()

        age_normalization = min_age
        out_features = int(torch.max(age_data) + 1)
        max_label = torch.max(age_data)
    assert out_features is not None

    train_img_data = img_data[train_indices]
    train_age_data = age_data[train_indices]

    val_img_data = img_data[val_indices]
    val_age_data = age_data[val_indices]

    train_bg_mask_data = None
    val_bg_mask_data = None

    if bg_mask_data is not None:
        train_bg_mask_data = bg_mask_data[train_indices]
        val_bg_mask_data = bg_mask_data[val_indices]

    # For testing, training data = (train + val).
    if testing:
        train_img_data = img_data
        train_age_data = age_data
        train_bg_mask_data = bg_mask_data

        train_indices = None
        val_indices = None

    if oversample:
        train_img_data, train_age_data, train_bg_mask_data = oversampled(train_img_data, train_age_data, train_bg_mask_data, threshold=oversample_threshold)

    if cuda:
        train_img_data = train_img_data.to('cuda')
        train_age_data = train_age_data.to('cuda')

        if train_bg_mask_data is not None:
            train_bg_mask_data = train_bg_mask_data.to('cuda')


    train_dataset = OtolithAgesDataset(
        img_data=train_img_data,
        age_data=train_age_data,
        bg_mask_data=train_bg_mask_data,
        augmentations=augmentations,
        bg_transform=bg_transform
    )

    # For testing, validation data = test.
    if testing:
        val_dataset = make_test_dataset(
            base_directory=base_directory,
            bg_transform=bg_transform,
            model_type=model_type,
            age_normalization=age_normalization,
            cuda=cuda,
            max_label=max_label
        )
    else:
        if cuda:
            val_img_data = val_img_data.to('cuda')
            val_age_data = val_age_data.to('cuda')

            if val_bg_mask_data is not None:
                val_bg_mask_data.to('cuda')

        val_dataset = OtolithAgesDataset(
            img_data=val_img_data,
            age_data=val_age_data,
            bg_mask_data=val_bg_mask_data,
            augmentations=(False, False, False, False),
            bg_transform=bg_transform
        )

    return train_dataset, val_dataset, train_indices, val_indices, age_normalization, out_features

# Save plots, models

def write_best_metrics_plot(tag, data, reread_trigger, writer):
    # Write plots to tensorboard for epochs with best metrics.
    targets = data['targets']
    preds = data['preds']

    acc = data['acc']
    loss = data['loss']

    within_1_acc = data['within_1_acc']
    within_3_acc = data['within_3_acc']
    within_5_acc = data['within_5_acc']

    reread_trigger_acc = data['reread_trigger_acc']

    max_age = int(math.ceil(max(targets)))
    min_age = int(math.ceil(min(targets)))

    ages = list(range(min_age, max_age + 1))

    R_squared = 'R\u00b2'
    plus_minus = '\u00b1'

    dataset_name = 'OtolithAges'
    title = f'{dataset_name} Test Dataset (n = {len(preds)}): {R_squared} = {acc:.4f}, Loss = {loss:.4f}'
    caption = f'{100*within_1_acc:.2f}% {plus_minus}1 accuracy, {100*within_3_acc:.2f}% {plus_minus}3 accuracy, {100*within_5_acc:.2f}% {plus_minus}5 accuracy, {100*reread_trigger_acc:.2f}% reread trigger accuracy'

    plt.figure(figsize=(8,6))

    if reread_trigger:
        fill_ages_ranges = [
            list(range(0, 6)),
            list(range(5, 11)),
            list(range(10, 16)),
            list(range(15, 21)),
            list(range(20, 26)),
            list(range(25, max(31, max_age + 5)))
        ]

        color_list = [
            'g',
            'b',
            'y',
            'r',
            'c',
            'm'
        ]

        for age_range, fill_ages in enumerate(fill_ages_ranges):
            fill_min = [max(age - age_range, 0) for age in fill_ages]
            fill_max = [age + age_range for age in fill_ages]

            plt.fill_between(fill_ages, fill_min, fill_max, color=color_list[age_range], alpha=.15)

    else:
        fill_ages = list(range(0, max_age + 5))

        fill_min_1 = [max(age - 1, 0) for age in fill_ages]
        fill_max_1 = [age + 1 for age in fill_ages]

        fill_min_3 = [max(age - 3, 0) for age in fill_ages]
        fill_max_3 = [age + 3 for age in fill_ages]

        fill_min_5 = [max(age - 5, 0) for age in fill_ages]
        fill_max_5 = [age + 5 for age in fill_ages]

        plt.fill_between(fill_ages, fill_min_1, fill_max_1, color='b', alpha=.15)
        plt.fill_between(fill_ages, fill_min_3, fill_max_3, color='r', alpha=.1)
        plt.fill_between(fill_ages, fill_min_5, fill_max_5, color='g', alpha=.05)
    
    gt_label = dataset_name + ' Ages'
    plt.plot(ages, ages, label=gt_label, color='black')

    plt.scatter(targets, preds, alpha=0.6, marker='x', color='green', label='Predicted Ages')
    
    plt.ylim(0, max_age + 5)
    plt.xlim(0, max_age + 2)
    
    plt.title(title)

    plt.figtext(
        x=0.5,
        y=0.01,
        s=caption,
        wrap=True, 
        horizontalalignment='center',
        fontsize=10
    )
    
    plt.xlabel(gt_label)
    plt.ylabel('Predicted Ages')
    plt.legend(loc='upper left')
    
    writer.add_figure(tag, plt.gcf())
    writer.flush()
    
    plt.close()

def save_models(best_acc, best_acc_model, best_loss, best_loss_model, model_path):
    # Save models for epochs with best accuracy and best loss.
    best_acc_model_path = f'{model_path}/r2_{best_acc:.4f}.pt'
    torch.save(
        obj=best_acc_model.state_dict(),
        f=best_acc_model_path
    )

    best_loss_model_path = f'{model_path}/loss_{best_loss:.4f}.pt'
    torch.save(
        obj=best_loss_model.state_dict(),
        f=best_loss_model_path
    )

# Calculate metrics

def update_running_metrics(running_loss, running_within_1, running_within_3, running_within_5, running_reread_trigger, loss, input_size, labels, batch_preds):
    # Aggregate running metrics across batches during training.
    running_loss += loss.item() * input_size

    reread_trigger_thresholds = {
        (0, 5): 0,
        (6, 10): 1,
        (11, 15): 2,
        (16, 20): 3,
        (21, 25): 4,
        (26, math.inf): 5
    }

    for i in range(len(batch_preds)):
        curr_pred = batch_preds[i].item()
        curr_label = labels.data[i]

        if curr_pred <= curr_label + 1 and curr_pred >= curr_label - 1:
            running_within_1 += 1

        if curr_pred <= curr_label + 3 and curr_pred >= curr_label - 3:
            running_within_3 += 1

        if curr_pred <= curr_label + 5 and curr_pred >= curr_label - 5:
            running_within_5 += 1

        for age_range in reread_trigger_thresholds:
            range_min, range_max = age_range

            threshold = reread_trigger_thresholds[age_range]

            if curr_label >= range_min and curr_label <= range_max and curr_pred >= curr_label - threshold and curr_pred <= curr_label + threshold:
                running_reread_trigger += 1
                break
    
    return running_loss, running_within_1, running_within_3, running_within_5, running_reread_trigger

def calculate_epoch_metrics(running_loss, running_within_1, running_within_3, running_within_5, running_reread_trigger, dataset_sizes, phase, targets, preds):
    # Use running metrics to calculate final metrics for epoch.
    epoch_loss = running_loss / dataset_sizes[phase]

    epoch_within_1 = running_within_1 / dataset_sizes[phase]
    epoch_within_3 = running_within_3 / dataset_sizes[phase]
    epoch_within_5 = running_within_5 / dataset_sizes[phase]

    epoch_reread_trigger = running_reread_trigger / dataset_sizes[phase]

    metric = R2Score()
    metric.update(input=torch.Tensor(preds), target=torch.Tensor(targets))
    epoch_acc = metric.compute()
    
    return epoch_loss, epoch_acc, epoch_within_1, epoch_within_3, epoch_within_5, epoch_reread_trigger

def update_best_epoch_metrics(model, targets, preds, preds_r, best_r2_data, best_loss_data, best_within_1_data, best_within_3_data, best_within_5_data, best_reread_trigger_data, best_model_r2_path=None, best_model_loss_path=None):
    acc, acc_r, best_acc = best_r2_data
    loss, loss_r, best_loss = best_loss_data

    within_1, within_1_r, best_within_1 = best_within_1_data
    within_3, within_3_r, best_within_3 = best_within_3_data
    within_5, within_5_r, best_within_5 = best_within_5_data

    reread_trigger, reread_trigger_r, best_reread_trigger = best_reread_trigger_data

    curr_epoch = {
        'targets': targets,
        'preds': preds,
        'acc': acc,
        'loss': loss,
        'within_1_acc': within_1,
        'within_3_acc': within_3,
        'within_5_acc': within_5,
        'reread_trigger_acc': reread_trigger
    }

    curr_epoch_r = {
        'targets': targets,
        'preds': preds_r,
        'acc': acc_r,
        'loss': loss_r,
        'within_1_acc': within_1_r,
        'within_3_acc': within_3_r,
        'within_5_acc': within_5_r,
        'reread_trigger_acc': reread_trigger_r
    }

    # default
    if acc > best_acc['default']['acc']:
        best_acc['default'] = curr_epoch

        if best_model_r2_path is not None:
            torch.save(
                obj=model.state_dict(),
                f=best_model_r2_path
            )
    
    if loss < best_loss['default']['loss']:
        best_loss['default'] = curr_epoch

        if best_model_loss_path is not None:
            torch.save(
                obj=model.state_dict(),
                f=best_model_loss_path
            )

    if within_1 > best_within_1['default']['within_1_acc']:
        best_within_1['default'] = curr_epoch
    
    if within_3 > best_within_3['default']['within_3_acc']:
        best_within_3['default'] = curr_epoch

    if within_5 > best_within_5['default']['within_5_acc']:
        best_within_5['default'] = curr_epoch

    if reread_trigger > best_reread_trigger['default']['reread_trigger_acc']:
        best_reread_trigger['default'] = curr_epoch

    # rounded
    if acc_r > best_acc['rounded']['acc']:
        best_acc['rounded'] = curr_epoch_r
    
    if loss_r < best_loss['rounded']['loss']:
        best_loss['rounded'] = curr_epoch_r

    if within_1_r > best_within_1['rounded']['within_1_acc']:
        best_within_1['rounded'] = curr_epoch_r
    
    if within_3_r > best_within_3['rounded']['within_3_acc']:
        best_within_3['rounded'] = curr_epoch_r

    if within_5_r > best_within_5['rounded']['within_5_acc']:
        best_within_5['rounded'] = curr_epoch_r

    if reread_trigger_r > best_reread_trigger['rounded']['reread_trigger_acc']:
        best_reread_trigger['rounded'] = curr_epoch_r

    return best_acc, best_loss, best_within_1, best_within_3, best_within_5, best_reread_trigger