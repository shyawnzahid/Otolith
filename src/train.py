import os
import time
import torch

from tempfile import TemporaryDirectory
from src.utils import Metric, BestEpoch, denormalize, update_running_metrics, calculate_epoch_metrics, update_best_epoch_metrics

def train_model(model, criterion, optimizer, scheduler, fe_epochs, ft_epochs, fe_colder_lr, ft_layers, ft_learning_rates, early_stopping, dataloaders, dataset_sizes, device, writer, age_normalization, model_type):
    since = time.time()
    total_epochs = fe_epochs + ft_epochs

    with TemporaryDirectory() as tempdir:
        # Save models for epoch with best accuracy and loss
        best_model_r2_path = os.path.join(tempdir, 'best_model_r2.pt')
        best_model_loss_path = os.path.join(tempdir, 'best_model_loss.pt')

        torch.save(
            obj=model.state_dict(),
            f=best_model_r2_path
        )

        torch.save(
            obj=model.state_dict(),
            f=best_model_loss_path
        )

        best_r2 = BestEpoch()
        best_loss = BestEpoch()

        best_within_1 = BestEpoch()
        best_within_3 = BestEpoch()
        best_within_5 = BestEpoch()

        best_reread_trigger = BestEpoch()

        fe_early_stopping = False
        cur_best = -100
        prev_best = -100

        fine_tuning = False

        # Training loop.
        for epoch in range(total_epochs):
            print(f'EPOCH {epoch+1}', flush=True)

            train_loss = Metric()
            train_acc = Metric()

            val_loss = Metric()
            val_acc = Metric()

            within_1_acc = Metric()
            within_3_acc = Metric()
            within_5_acc = Metric()
            reread_trigger_acc = Metric()

            # Switch from feature extraction stage to fine tuning stage: unfreeze frozen layers, update learning rates, and load best epoch from feature extractive stage
            if (epoch+1 == fe_epochs+1 or fe_early_stopping) and not fine_tuning:
                print(f'FINE TUNING: EPOCH {epoch+1}')
                model.load_state_dict(torch.load(best_model_r2_path))
                fine_tuning = True

                for g in optimizer.param_groups:
                    g['lr'] = fe_colder_lr

                ft_params = model.unfreeze(
                    layers=ft_layers,
                    learning_rates=ft_learning_rates
                )

                for ft_param in ft_params:
                    optimizer.add_param_group(ft_param)

            preds_by_phase = {
                'train': [],
                'val': []
            }

            preds_by_phase_r = {
                'train': [],
                'val': []
            }

            targets_by_phase = {
                'train': [],
                'val': []
            }

            for phase in ['train', 'val']:
                # Enable gradient calculations for training phase, disable for validation phase.
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = Metric()

                running_within_1 = Metric()
                running_within_3 = Metric()
                running_within_5 = Metric()

                running_reread_trigger = Metric()

                # Iterate through batches.
                for inputs, targets in dataloaders[phase]:
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs) # normalized

                        normalized_loss = criterion(outputs, targets) # normalized

                        # Optimize using loss calculated from normalized ages.
                        if phase == 'train':
                            normalized_loss.backward()
                            optimizer.step()
                    
                    with torch.no_grad():
                        if model_type == 'regression':
                            preds = denormalize(outputs, age_normalization, model_type)
                            preds_r = torch.round(preds)

                            labels = denormalize(targets, age_normalization, model_type)

                            loss = criterion(preds, labels)
                            loss_r = criterion(preds_r, labels)
                        else:
                            preds = torch.argmax(outputs, dim=1)
                            preds = denormalize(preds, age_normalization, model_type)
                            preds_r = preds

                            labels = denormalize(targets, age_normalization, model_type)

                            loss = normalized_loss
                            loss_r = normalized_loss

                        targets_by_phase[phase] += [labels[i].item() for i in range(len(labels))]

                        preds_by_phase[phase] += [preds[i].item() for i in range(len(preds))]
                        preds_by_phase_r[phase] += [preds_r[i].item() for i in range(len(preds_r))]

                        running_loss.default, running_within_1.default, running_within_3.default, running_within_5.default, running_reread_trigger.default = update_running_metrics(
                            running_loss=running_loss.default, 
                            running_within_1=running_within_1.default,
                            running_within_3=running_within_3.default,
                            running_within_5=running_within_5.default,
                            running_reread_trigger=running_reread_trigger.default,
                            loss=loss,
                            input_size=inputs.size(0), 
                            labels=labels, 
                            batch_preds=preds, 
                        )

                        running_loss.rounded, running_within_1.rounded, running_within_3.rounded, running_within_5.rounded, running_reread_trigger.rounded = update_running_metrics(
                            running_loss=running_loss.rounded, 
                            running_within_1=running_within_1.rounded,
                            running_within_3=running_within_3.rounded,
                            running_within_5=running_within_5.rounded,
                            running_reread_trigger=running_reread_trigger.rounded,
                            loss=loss_r,
                            input_size=inputs.size(0), 
                            labels=labels, 
                            batch_preds=preds_r, 
                        )

                with torch.no_grad():
                    epoch_loss, epoch_acc, epoch_within_1, epoch_within_3, epoch_within_5, epoch_reread_trigger = calculate_epoch_metrics(
                        running_loss=running_loss.default,
                        running_within_1=running_within_1.default,
                        running_within_3=running_within_3.default,
                        running_within_5=running_within_5.default,
                        running_reread_trigger=running_reread_trigger.default,
                        dataset_sizes=dataset_sizes,
                        phase=phase,
                        targets=targets_by_phase[phase],
                        preds=preds_by_phase[phase],
                    )
    
                    epoch_loss_r, epoch_acc_r, epoch_within_1_r, epoch_within_3_r, epoch_within_5_r, epoch_reread_trigger_r = calculate_epoch_metrics(
                        running_loss=running_loss.rounded,
                        running_within_1=running_within_1.rounded,
                        running_within_3=running_within_3.rounded,
                        running_within_5=running_within_5.rounded,
                        running_reread_trigger=running_reread_trigger.rounded,
                        dataset_sizes=dataset_sizes,
                        phase=phase,
                        targets=targets_by_phase[phase],
                        preds=preds_by_phase_r[phase],
                    )
                    
                    if phase == 'train':
                        scheduler.step()
    
                        train_loss.default = epoch_loss
                        train_loss.rounded = epoch_loss_r
    
                        train_acc.default = epoch_acc
                        train_acc.rounded = epoch_acc_r
    
                    else:
                        # Storing and writing metrics for validation data.
                        val_loss.default = epoch_loss
                        val_loss.rounded = epoch_loss_r
    
                        val_acc.default = epoch_acc
                        val_acc.rounded = epoch_acc_r
    
                        within_1_acc.default = epoch_within_1
                        within_1_acc.rounded = epoch_within_1_r
    
                        within_3_acc.default = epoch_within_3
                        within_3_acc.rounded = epoch_within_3_r
    
                        within_5_acc.default = epoch_within_5
                        within_5_acc.rounded = epoch_within_5_r
    
                        reread_trigger_acc.default = epoch_reread_trigger
                        reread_trigger_acc.rounded = epoch_reread_trigger_r

                        best_r2.data, best_loss.data, best_within_1.data, best_within_3.data, best_within_5.data, best_reread_trigger.data = update_best_epoch_metrics(
                            model=model,
                            targets=targets_by_phase[phase],
                            preds=preds_by_phase[phase],
                            preds_r=preds_by_phase_r[phase],
                            best_r2_data=(epoch_acc, epoch_acc_r, best_r2.data),
                            best_loss_data=(epoch_loss, epoch_loss_r, best_loss.data),
                            best_within_1_data=(epoch_within_1, epoch_within_1_r, best_within_1.data),
                            best_within_3_data=(epoch_within_3, epoch_within_3_r, best_within_3.data),
                            best_within_5_data=(epoch_within_5, epoch_within_5_r, best_within_5.data),
                            best_reread_trigger_data=(epoch_reread_trigger, epoch_reread_trigger_r, best_reread_trigger.data),
                            best_model_r2_path=best_model_r2_path,
                            best_model_loss_path=best_model_loss_path
                        )

                        writer.add_scalar('Best Metrics/best_r2', best_r2.data['default']['acc'], global_step=epoch+1)
                        writer.add_scalar('Best Metrics/best_loss', best_loss.data['default']['loss'], global_step=epoch+1)
                        writer.add_scalar('Best Metrics/best_within_1', best_within_1.data['default']['within_1_acc'], global_step=epoch+1)
                        writer.add_scalar('Best Metrics/best_within_3', best_within_3.data['default']['within_3_acc'], global_step=epoch+1)
                        writer.add_scalar('Best Metrics/best_within_5', best_within_5.data['default']['within_5_acc'], global_step=epoch+1)
                        writer.add_scalar('Best Metrics/best_reread_trigger', best_reread_trigger.data['default']['reread_trigger_acc'], global_step=epoch+1)

                        writer.add_scalar('Best Metrics/best_r2_r', best_r2.data['rounded']['acc'], global_step=epoch+1)
                        writer.add_scalar('Best Metrics/best_loss_r', best_loss.data['rounded']['loss'], global_step=epoch+1)
                        writer.add_scalar('Best Metrics/best_within_1_r', best_within_1.data['rounded']['within_1_acc'], global_step=epoch+1)
                        writer.add_scalar('Best Metrics/best_within_3_r', best_within_3.data['rounded']['within_3_acc'], global_step=epoch+1)
                        writer.add_scalar('Best Metrics/best_within_5_r', best_within_5.data['rounded']['within_5_acc'], global_step=epoch+1)
                        writer.add_scalar('Best Metrics/best_reread_trigger_r', best_reread_trigger.data['rounded']['reread_trigger_acc'], global_step=epoch+1)

                        if epoch_acc > cur_best:
                            cur_best = epoch_acc
                        print(f'EPOCH {epoch+1} ACCURACY: {epoch_acc}', flush=True)

                        if ((epoch+1) % early_stopping == 0) and not fine_tuning:
                            print(f'CHECKING FOR EARLY STOPPING: EPOCH {epoch+1}', flush=True)
                            print(f'current best = {cur_best}, previous best = {prev_best}', flush=True)
                            if cur_best > prev_best:
                                prev_best = cur_best
                                cur_best = -100
                            else:
                                fe_early_stopping = True
                                print(f'STOPPING EARLY at EPOCH {epoch+1}', flush=True)
                        
                        writer.flush()

            write_loss = {
                'train_loss': train_loss.default,
                'val_loss': val_loss.default
            }
            writer.add_scalars('Loss & Accuracy/loss', write_loss, global_step=epoch+1)

            write_acc = {
                'train_acc': train_acc.default,
                'val_acc': val_acc.default
            }
            writer.add_scalars('Loss & Accuracy/acc', write_acc, global_step=epoch+1)

            write_within_acc = {
                'within_1': within_1_acc.default,
                'within_3': within_3_acc.default,
                'within_5': within_5_acc.default,
                'reread_trigger': reread_trigger_acc.default
            }
            writer.add_scalars('Loss & Accuracy/within_acc', write_within_acc, global_step=epoch+1)

            write_loss_r = {
                'train_loss_r': train_loss.rounded,
                'val_loss_r': val_loss.rounded
            }
            writer.add_scalars('Loss & Accuracy - rounded outputs/loss_r', write_loss_r, global_step=epoch+1)

            write_acc_r = {
                'train_acc_r': train_acc.rounded,
                'val_acc_r': val_acc.rounded
            }
            writer.add_scalars('Loss & Accuracy - rounded outputs/acc_r', write_acc_r, global_step=epoch+1)

            write_within_acc_r = {
                'within_1_r': within_1_acc.rounded,
                'within_3_r': within_3_acc.rounded,
                'within_5_r': within_5_acc.rounded,
                'reread_trigger_r': reread_trigger_acc.rounded
            }
            writer.add_scalars('Loss & Accuracy - rounded outputs/within_acc_r', write_within_acc_r, global_step=epoch+1)

            writer.flush()

        time_elapsed = time.time() - since
        print(f'Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

        best_acc_model = model
        best_loss_model = model

        best_acc_model.load_state_dict(torch.load(best_model_r2_path))
        best_loss_model.load_state_dict(torch.load(best_model_loss_path))
    
    return best_acc_model, best_loss_model, best_r2, best_loss, best_within_1, best_within_3, best_within_5, best_reread_trigger