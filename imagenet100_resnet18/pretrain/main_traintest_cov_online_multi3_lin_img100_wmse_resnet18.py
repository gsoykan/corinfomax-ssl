import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import data_utils_dist as data_utils
import optim_utils3 as optim_utils
import parsing_file_dist as parsing_file
import save_utils_con_multi as save_utils
from distributed import init_distributed_mode
from loss import CovarianceLossv2 as CovarianceLoss
from loss import invariance_loss
from metrics import grad_norm
from model_base_wmse_resnet18 import CovModel6 as CovModel
from pretrain_base_dist_multigpulinear import pretrain_cov_dist, Covdet


def train_test(args):
    set_seed(10)
    # torch.backends.cudnn.benchmark = True
    init_distributed_mode(args)
    gpu = torch.device(args.device)

    print(f"============== ARGS ==============\n\n{args}\n\n==================================")

    pretrain_data, lin_train_data, lin_test_data = data_utils.make_data(args.dataset, args.subset, args.subset_type)

    sampler = torch.utils.data.distributed.DistributedSampler(pretrain_data, shuffle=True)
    print(args.batch_size % args.world_size, args.batch_size, args.world_size)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size

    pre_train_loader = DataLoader(pretrain_data, batch_size=per_device_batch_size, num_workers=args.n_workers,
                                  sampler=sampler, pin_memory=True, drop_last=True)

    # Initialize loss and classifier
    covariance_loss = CovarianceLoss(args)

    # Model and optimizer setup
    pre_model = CovModel(args).cuda(gpu)
    lin_classifier = None
    # Model and loss
    pre_model = Covdet(pre_model, covariance_loss, args, lin_classifier)
    pre_model = nn.SyncBatchNorm.convert_sync_batchnorm(pre_model)
    pre_model = DDP(pre_model, device_ids=[gpu], find_unused_parameters=True)

    pre_optimizer = optim_utils.make_optimizer(pre_model, args, pretrain=True)
    pre_scheduler = optim_utils.make_scheduler(pre_optimizer, args, pretrain=True)

    if args.rank == 0:
        save_results = save_utils.SaveResults(args)
        writer, save_name_linear = save_results.create_results()
        save_results.save_parameters()

    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)

        # Training 
        train_loss, train_cov_loss, train_sim_loss, break_code = pretrain_cov_dist(pre_model, pre_train_loader,
                                                                                   pre_optimizer, invariance_loss,
                                                                                   epoch)
        if break_code:
            test_acc1 = 0.0
            break
        # get learning rate from optimizer
        pre_curr_lr = pre_optimizer.param_groups[0]['lr']
        total_norm = grad_norm(pre_model)

        if args.rank == 0:
            save_results.update_results(train_loss, train_cov_loss, train_sim_loss, pre_curr_lr, total_norm)
            R_eigs = save_results.save_eigs(covariance_loss, epoch)
            save_utils.update_tensorboard(writer, train_loss, train_cov_loss, train_sim_loss, pre_curr_lr, total_norm,
                                          R_eigs, epoch)
            save_results.save_stats(epoch)

            writer.flush()

            if (epoch % 100 == 0) | (epoch == args.epochs):
                save_results.save_model(pre_model, pre_optimizer, train_loss, epoch)

        # Scheduler action
        if args.pre_scheduler == "None":
            pass
        elif args.pre_scheduler == "reduce_plateau":
            pre_scheduler.step(train_loss)
        else:
            pre_scheduler.step()

    if args.rank == 0:
        save_results.save_model_resnet(pre_model)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == '__main__':
    """
    torchrun 
    --nproc_per_node=8
     main_traintest_cov_online_multi3_lin_img100_wmse_resnet18.py 
    """
    use_preset_values = True
    preset_values = {
        'epochs': 400,
        'learning_rate': 1.0,
        'lin_epochs': 200,
        'lin_batch_size': 256,
        'R_ini': 1.0,
        'projector': '4096-4096-128',
        'batch_size': 1024,
        'sim_loss_weight': 500.0,
        'cov_loss_weight': 1.0,
        'la_R': 0.01,
        'la_mu': 0.01,
        'R_eps_weight': 1e-08,
        'w_decay': 1e-4,
        'pre_optimizer': 'SGD',
        'pre_scheduler': 'lin_warmup_cos',
        'lin_optimizer': 'SGD',
        'lin_learning_rate': 0.2,
        'lin_w_decay': 0,
        'lin_scheduler': 'lin_warmup_cos',
        'dataset': 'comics_crops_bodies',
        'lin_dataset': None,
        'con_name': 'cov_imagenet100_ep400_norm_resnet18',
        'model_name': 'resnet18',
        'n_workers': 12,
        'normalize_on': None,
        'min_lr': 5e-3
    }
    parser = parsing_file.create_parser()
    arguments = parser.parse_args()

    if use_preset_values:
        for k, v in preset_values.items():
            arguments.__setattr__(k, v)

    arguments.lin_epochs = arguments.epochs
    train_test(arguments)
