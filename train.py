import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torchsummary

from runner.trainer import Epoch
from utils.alarm import *
from utils.utility import *
import os


def train(**kwargs):
    """
    Initiate a complete runner procedure
    :param kwargs: Parameters used to configure the runner procedure
    """

    print("Checking Runtime Parameters")
    config_pre_check(**kwargs)
    print("Application starts")

    # Hyper Parameters & Environment Settings
    fargs = {}
    for key in kwargs.keys():
        fargs[key] = kwargs[key]

    # Seeds
    set_seed(fargs['random_seed'])

    # Model Configurations
    workspace_path = get_workspace_dir(fargs["base_workspace_path"], fargs["experiment_name"])
    model = get_model(mode="train", **fargs).to(fargs["device"])
    if fargs['enable_resume']:
        model.load_state_dict(torch.load(fargs['resume_path']))
    preproc = get_preproc_func(**fargs)
    train_dataset = get_train_dataset(preproc, classes_x=[0, 1], **fargs)
    train_dataloader = get_dataloader(train_dataset, **fargs)
    test_dataset = get_test_dataset(preproc, classes_x=[0, 1], **fargs)
    test_dataloader = get_dataloader(test_dataset, **fargs)
    optimizer = get_optimizer(fargs['optim'], model.parameters(), fargs['lr'])
    if fargs['enable_resume']:
        optimizer.load_state_dict(torch.load(fargs['resume_path']+"_optim.pth"))
    scheduler = None
    cust_scheduler = None
    if fargs['lr_sched']:
        scheduler = get_lr_scheduler(optimizer, **fargs)
        cust_scheduler = get_customized_lr_scheduler(optimizer, len(train_dataloader), **fargs)
    loss = get_loss_func(**fargs)
    metrics, metrics_mask = get_metrics(**fargs)
    dltype = get_train_epoch_type(**fargs)
    epoch_start = 0
    if fargs['enable_resume']:
        epoch_start = fargs['resume_epoch']

    # Model Summary
    torchsummary.summary(model, (3, fargs["image_scale_h"], fargs["image_scale_w"]), fargs["batch_size"],
                         device=fargs["device"])
    # Dataset Prefetching
    if fargs["dataset_preload"]:
        preload_dataset(train_dataloader, "Loading train dataset")
        if fargs["enable_test"]:
            preload_dataset(test_dataloader, "Loading test dataset")
    else:
        print("Prefetch ignored")
    print("Train:", len(train_dataset), "Test:", len(test_dataset))
    save_train_config(workspace_path, fargs)

    # Runner Definition
    print("Training...")
    train_runner = Epoch("Train Phase", model, train_dataloader, optimizer, loss, metrics, workspace_path,
                         device=fargs['device'], dataloader_type=dltype, metrics_mask=metrics_mask,
                         scheduler=cust_scheduler, backprop_gap=fargs['backprop_iteration_interval'],
                         save_checkpoints=False)
    test_runner = Epoch("Val Phase", model, test_dataloader, optimizer, loss, metrics, workspace_path,
                        back_prop=False, device=fargs['device'], dataloader_type=dltype,
                        metrics_mask=metrics_mask)
    train_log, test_log = {}, {}

    # Model Resume
    if fargs['lr_sched']:
        for i in range(epoch_start):
            print("Epoch", i, "Lr=", optimizer.state_dict()['param_groups'][0]['lr'])
            print("Epoch skipped")
            scheduler.step()

    # Training
    for i in range(epoch_start, fargs["epochs"]):
        print("Epoch", i, "Lr=", optimizer.state_dict()['param_groups'][0]['lr'])
        train_result = train_runner.run(i)
        train_log[i] = train_result
        save_train_log(workspace_path, train_log, name="train_result")
        if fargs["enable_test"]:
            test_result = test_runner.run(i)
            test_log[i] = test_result
            save_train_log(workspace_path, test_log, name="test_result")
        if fargs['lr_sched']:
            scheduler.step()


if __name__ == "__main__":
    train(**get_config_json())
