from torch.utils.data import DataLoader

import data_utils
import optim_utils
import parsing_file
import save_utils_con as save_utils
from loss import CovarianceLoss
from metrics import grad_norm
from model_base_resnet import CovModel
from pretrain_base import pretrain_cov


def train_test(args):
    """
    Pretrain for args.epochs, then linear evaluation (train and test) for args.lin_epochs. 
    """
    pretrain_data, _, _ = data_utils.make_data(args.dataset, args.subset, args.subset_type)

    pre_train_loader = DataLoader(pretrain_data, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=True,
                                  pin_memory=True, drop_last=True)

    # Initialize corinfomax loss 
    covariance_loss = CovarianceLoss(args)

    # Model and optimizer setup
    pre_model = CovModel(args).cuda()
    pre_optimizer = optim_utils.make_optimizer(pre_model, args, pretrain=True)
    pre_scheduler = optim_utils.make_scheduler(pre_optimizer, args, pretrain=True)

    save_results = save_utils.SaveResults(args)
    writer = save_results.create_results()
    save_results.save_parameters()

    for epoch in range(1, args.epochs + 1):
        # Pretraining 
        train_loss, train_cov_loss, train_sim_loss, break_code = pretrain_cov(pre_model, pre_train_loader,
                                                                              covariance_loss, pre_optimizer,
                                                                              args.cov_loss_weight,
                                                                              args.sim_loss_weight, epoch)
        if break_code:
            test_acc1 = 0.0
            break
        # Scheduler action
        if args.pre_scheduler == "None":
            pass
        elif args.pre_scheduler == "reduce_plateau":
            pre_scheduler.step(train_loss)
        else:
            pre_scheduler.step()
        # get learning rate from optimizer
        pre_curr_lr = pre_optimizer.param_groups[0]['lr']
        total_norm = grad_norm(pre_model)

        save_results.update_results(train_loss, train_cov_loss, train_sim_loss, pre_curr_lr, total_norm)
        R_eigs = save_results.save_eigs(covariance_loss, epoch)

        save_utils.update_tensorboard(writer, train_loss, train_cov_loss, train_sim_loss, pre_curr_lr, total_norm,
                                      R_eigs, epoch)
        save_results.save_stats(epoch)

        writer.flush()

        if epoch % 200 == 0:
            save_results.save_model(pre_model, pre_optimizer, train_loss, epoch)


if __name__ == '__main__':
    parser = parsing_file.create_parser()
    arguments = parser.parse_args()

    train_test(arguments)
