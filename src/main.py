import os
import torch
import numpy
import random
import datetime

torch.manual_seed(42)
numpy.random.seed(42)
random.seed(42)

from absl import app, flags
from src.train import train_model
from torch.utils.data import DataLoader
from dotenv import load_dotenv, find_dotenv
from src.models.OtolithViT import OtolithViT
from torch.utils.tensorboard import SummaryWriter
from src.models.OtolithResNet import OtolithResNet
from src.utils import write_best_metrics_plot, save_models, make_datasets

# Miscellaneous.
flags.DEFINE_enum('device', 'cuda', ['cuda', 'cpu'], 'Which device to use') 
flags.DEFINE_string('logdir', 'log', 'Log directory for TensorBoard')

# Data.
flags.DEFINE_enum('bg_transform', 'rem', ['orig', 'rem'], 'What form of bg processing to do on the image backgrounds')
flags.DEFINE_boolean('oversample', True, 'Oversample ages with less than 10 samples in training dataset')
flags.DEFINE_integer('batch_size', 64, 'Batch size')
flags.DEFINE_integer('fold', 1, 'Which of the 5 folds to use for validation')
flags.DEFINE_boolean('color_augs', True, 'Random color jittering augmentations')
flags.DEFINE_boolean('flip_augs', True, 'Random horizontal/vertical flip')
flags.DEFINE_boolean('crop_augs', True, 'Random vertical cropping by 10 pixels')
flags.DEFINE_boolean('rotate_augs', True, 'Random rotation by +/- 10 degrees')

# Model.
flags.DEFINE_enum('model', 'res18', [
                                        'res18', 'res34', 'res50', 'res101', 'res152',
                                        'vit_16_224', 'vit_16_384', 
                                        'vit_32_224', 'vit_32_384',
                                        'vit_16_224_clip', 'vit_16_384_clip',
                                        'vit_32_224_clip', 'vit_32_256_clip', 'vit_32_384_clip', 'vit_32_448_clip',
                                    ], 'Model name.')

# Training.
flags.DEFINE_integer('early_stopping', 50, 'Number of epochs of decreasing accuracy before early stopping (0 means no early stopping).')
flags.DEFINE_list('optimizer_params', ['AdamW', '5e-2'], 'Optimizer type and parameters: Supports Adam, AdamW, & SGD')
flags.DEFINE_enum('loss_function', 'L1', ['MSE', 'L1', 'CE'], 'Which loss function to optimize during training.')
flags.DEFINE_enum('model_type', 'regression', ['regression', 'classification'], 'Which model type to train')
flags.DEFINE_list('scheduler_params', ['StepLR', '30', '1'], 'Scheduler type and parameters: Supports StepLR, CosineAnnealingLR, & CosineAnnealingWarmRestarts')
flags.DEFINE_integer('fe_layer_cutoff_index', 1, 'Layer number cutoff for feature extraction')
flags.DEFINE_float('fe_lr', 1e-4, 'Learning rate for feature extraction stage with ImageNet')
flags.DEFINE_float('fe_colder_lr', 1e-8, 'Learning rate for fine tuning stage with ImageNet')
flags.DEFINE_float('ft_lr', 1e-5, 'Learning rate for pretrained layers in fine tuning with ImageNet')
flags.DEFINE_float('ft_lr_decay', 0.85, 'Reverse layer wise pretrained learning rate decay with ImageNet')
flags.DEFINE_integer('fe_epochs', 1, 'Number of feature extraction epochs with ImageNet')
flags.DEFINE_integer('ft_epochs', 1, 'Number of fine tuning epochs with ImageNet')
flags.DEFINE_boolean('testing', False, 'Use full training data, report on test dataset')
flags.DEFINE_integer('oversample_threshold', 10, 'Maximum frequency for oversampling')

FLAGS = flags.FLAGS

load_dotenv(find_dotenv())

def main(argv):
    base_directory = os.environ['BASE_DIRECTORY']
    device = torch.device(FLAGS.device)

    # Logging training run and hyperparameters with timestamp for tensorboard.
    timestamp = f'{datetime.datetime.now()}'.split('.')[0][2:]
    writer = SummaryWriter(f'{base_directory}/{FLAGS.logdir}/{timestamp}')

    hparam_keys = [
        'MISC.device',
        'MISC.logdir',
        'DATA.bg_transform', 
        'DATA.oversample',
        'DATA.batch_size', 
        'DATA.fold',
        'AUG.color_augs', 
        'AUG.flip_augs', 
        'AUG.crop_augs',
        'AUG.rotate_augs',
        'MODEL.model', 
        'TRAINING.early_stopping',
        'TRAINING.optimizer_params', 
        'TRAINING.loss_function', 
        'TRAINING.model_type',
        'TRAINING.scheduler_params', 
        'TRAINING.fe_layer_cutoff_index',
        'TRAINING.fe_lr', 
        'TRAINING.fe_colder_lr', 
        'TRAINING.ft_lr', 
        'TRAINING.ft_lr_decay', 
        'TRAINING.fe_epochs', 
        'TRAINING.ft_epochs',
        'TRAINING.testing',
        'TRAINING.oversample_threshold'
    ]
    
    hparams = {hparam_key: str(FLAGS[hparam_key.split('.')[1]].value) for hparam_key in hparam_keys}

    # Setting the path where the trained model will be saved after training is finished.
    model_path = f'{base_directory}/saved_models/{timestamp}'
    while True:
        try:
            os.makedirs(model_path)
            break
        except:
            model_path += '-dup'

    # Reading in data with desired augmentations.
    augmentations = (FLAGS.flip_augs, FLAGS.color_augs, FLAGS.crop_augs, FLAGS.rotate_augs)

    train_dataset, val_dataset, train_indices, val_indices, age_normalization, out_features = make_datasets(
        base_directory=base_directory,
        bg_transform=FLAGS.bg_transform,
        augmentations=augmentations,
        cuda=(FLAGS.device == 'cuda' and not FLAGS.model.startswith('vit')),
        oversample=FLAGS.oversample,
        model_type=FLAGS.model_type,
        fold=(FLAGS.fold - 1),
        testing=FLAGS.testing,
        oversample_threshold=FLAGS.oversample_threshold
    )

    # ViT uses different parameters for color augmentations.
    if FLAGS.model.startswith('vit'):
        train_dataset.vit_augs()

    # Assembling and initializing data loaders for training and validation data.
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)
    
    generator = torch.Generator()
    generator.manual_seed(42)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=False
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=FLAGS.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=False
    )

    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }

    dataset_sizes = {
        'train': train_dataset.__len__(),
        'val': val_dataset.__len__()
    }

    # Initializing aging model architectures.
    if FLAGS.model.startswith('res'):
        model = OtolithResNet(
            model_name=FLAGS.model,
            out_features=out_features,
            train_indices=train_indices,
            val_indices=val_indices
        )

        layers = [
            'conv1', 
            'bn1', 
            'layer1', 
            'layer2', 
            'layer3', 
            'layer4', 
            'fc'
        ]
    else:
        model = OtolithViT(
            model_name=FLAGS.model,
            out_features=out_features,
            train_indices=train_indices,
            val_indices=val_indices
        )

        layers = [
            'cls_token',
            'pos_embed',
            'patch_embed',
        ]

        if FLAGS.model.endswith('clip'):
            layers += ['norm_pre']

        layers += [
            'blocks.0',
            'blocks.1',
            'blocks.2',
            'blocks.3',
            'blocks.4',
            'blocks.5',
            'blocks.6',
            'blocks.7',
            'blocks.8',
            'blocks.9',
            'blocks.10',
            'blocks.11',
            'norm',
            'head'
        ]

    # Freezing layers for feature extraction training phase.
    num_layers = len(layers)
    fe_layer_cutoff_index = FLAGS.fe_layer_cutoff_index
    assert fe_layer_cutoff_index < num_layers
    
    fe_layers = layers[-fe_layer_cutoff_index::]
    fe_params = model.unfreeze(
        layers=fe_layers, 
        learning_rates=[FLAGS.fe_lr]*len(fe_layers)
    )

    ft_layers = layers[:(num_layers - fe_layer_cutoff_index)]
    ft_learning_rates = list(reversed([FLAGS.ft_lr*(FLAGS.ft_lr_decay**i) for i in range(len(ft_layers))]))
    model.freeze(
        layers=ft_layers
    )

    model.to(device)

    # Initializing loss function, optimizer/parameters, scheduler/parameters
    if FLAGS.model_type == 'regression':
        if FLAGS.loss_function == 'L1':
            criterion = torch.nn.L1Loss()
        else:
            criterion = torch.nn.MSELoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    optimizer_params = FLAGS.optimizer_params
    optimizer_name = optimizer_params[0]
    weight_decay = float(optimizer_params[1])

    optimizer = None
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(
            params=fe_params,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'AdamW':
        optimizer = torch.optim.Adam(
            params=fe_params,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'SGD':
        momentum = float(optimizer_params[2])
        optimizer = torch.optim.SGD(
            params=fe_params,
            momentum=momentum,
            weight_decay=weight_decay
        )
    assert optimizer is not None

    scheduler_params = FLAGS.scheduler_params
    scheduler_name = scheduler_params[0]

    scheduler = None
    if scheduler_name == 'StepLR':
        step_size = int(scheduler_params[1])
        gamma = float(scheduler_params[2])
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=step_size,
            gamma=gamma
        )
    elif scheduler_name == 'CosineAnnealingLR':
        T_max = int(scheduler_params[1])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=T_max
        )
    elif scheduler_name == 'CosineAnnealingWarmRestarts':
        T_0 = int(scheduler_params[1])
        T_mult = int(scheduler_params[2])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=T_0,
            T_mult=T_mult
        )
    assert scheduler is not None

    # Model training.
    best_acc_model, best_loss_model, best_acc, best_loss, best_within_1, best_within_3, best_within_5, best_reread_trigger = train_model(
        model=model, 
        criterion=criterion, 
        optimizer=optimizer, 
        scheduler=scheduler,
        fe_epochs=FLAGS.fe_epochs,
        ft_epochs=FLAGS.ft_epochs,
        fe_colder_lr=FLAGS.fe_colder_lr,
        ft_layers=ft_layers,
        ft_learning_rates=ft_learning_rates,
        early_stopping=FLAGS.early_stopping,
        dataloaders=dataloaders, 
        dataset_sizes=dataset_sizes, 
        device=device,
        writer=writer,
        age_normalization=age_normalization,
        model_type=FLAGS.model_type
    )

    # Writing plots for various metrics, with both default and rounded outputs.
    tags = {
        'Plots/Best Accuracy' : best_acc.data,
        'Plots/Best Loss' : best_loss.data,
        'Plots/Best Within 1 Accuracy' : best_within_1.data,
        'Plots/Best Within 3 Accuracy' : best_within_3.data,
        'Plots/Best Within 5 Accuracy' : best_within_5.data,
        'Plots/Best Reread Trigger Accuracy' : best_reread_trigger.data
    }
    for tag in tags:
        reread_trigger = (tag == 'Plots/Best Reread Trigger Accuracy')
        write_best_metrics_plot(
            tag=tag,
            data=tags[tag]['default'],
            reread_trigger=reread_trigger,
            writer=writer
        )

        rounded_tag = f'{tag} - rounded outputs'
        write_best_metrics_plot(
            tag=rounded_tag,
            data=tags[tag]['rounded'],
            reread_trigger=reread_trigger,
            writer=writer
        )

    # Saving highest accuracy and highest loss models.
    save_models(
        best_acc=best_acc.data['default']['acc'],
        best_acc_model=best_acc_model,
        best_loss=best_loss.data['default']['loss'],
        best_loss_model=best_loss_model,
        model_path=model_path
    )

    # Writing final metrics.
    metrics = {
        'best_acc': best_acc.data['default']['acc'],
        'best_acc_r': best_acc.data['rounded']['acc']
    }
    writer.add_hparams(hparams, metrics, run_name='hparams')

    writer.flush()
    writer.close()
    
if __name__ == '__main__':
    app.run(main)