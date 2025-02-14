from models.padim_weights2 import PadimWeights2Trainer

import argparse
import os
import time

import random
from functools import partial
from torch.nn import functional as F
import torch.utils.data

from models.padim_weights import PadimWeightsTrainer

from tools.datasets import get_datasets, default_filename_sanitizer, get_full_dataset, get_train_dataset_for_testing
from tools.stats import *
from tools.parser import parse_args
from models.unet import Trainer as UNETTrainer
from models.padim import PadimTrainer
from models.ganomaly import GanomalyTrainer


def get_summary_writer_log_dir(args: argparse.Namespace) -> str:
    tb_log_dir_prefix = (
        f"CNN_bn_"
        f"bs={args.batch_size}_"
        f"lr={args.learning_rate}_"
        f"run_"
    )
    tb_log_dir = args.log_dir / (tb_log_dir_prefix + time.strftime('%m_%d_%H_%M_%S'))
    return str(tb_log_dir)


def log_args(logger: dict, args: argparse.Namespace):
    logger['batch_size'] = args.batch_size
    logger['epochs'] = args.epochs
    logger['has_nan'] = args.has_nan
    logger['seed'] = args.seed
    logger['dataset'] = args.dataset
    logger['subfolder'] = args.subfolder
    logger['beta_vae'] = args.beta_vae
    logger['lr'] = args.lr
    logger['image_size'] = args.image_size
    logger['latent_dim'] = args.latent_dim
    logger['result_dir'] = args.result_dir
    logger['cc'] = args.cc
    logger['model_type'] = args.model_type


def setup_args(logger, args):
    for key in logger:
        if key == 'result_dir':
            continue
        setattr(args, key, logger[key])


def get_model_name(args: argparse.Namespace, stats_saver):
    model_name = args.model_name
    if args.model_name.isdigit():
        logger = stats_saver.get_column(args.model_name)
        model_name = logger['model_name']
        return model_name

    if '.pth' not in args.model_name:
        # it's a path
        for file in os.listdir(args.model_name):
            if '.pth' in file:
                if '.pth' in model_name:
                    # find the latest model
                    date_1 = time.strptime(model_name.split('.')[0][-14:], '%m_%d_%H_%M_%S')
                    date_2 = time.strptime(file.split('.')[0][-14:], '%m_%d_%H_%M_%S')
                    if date_2 > date_1:
                        model_name = args.model_name + '/' + file
                else:
                    model_name = args.model_name + '/' + file
    return model_name


def main(args: argparse.Namespace):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(0)

    stats_saver = StatsSaver()

    if args.log:
        logger = {}
        log_args(logger, args)
        logger['info'] = ''
    else:
        logger = None

    train_dataset, test_normal, test_abnormal = get_datasets(args, logger)
    train_dataset_for_testing = get_train_dataset_for_testing(args, logger=logger, load_from_logger=False)
    full_dataset = get_full_dataset(args)

    if not os.path.isdir(args.result_dir):
        os.mkdir(args.result_dir)

    if not os.path.isdir('npz'):
        os.mkdir('npz')

    start_time = time.strftime('%y_%m_%d_%H_%M_%S')

    if 'padim_weights2' in args.model_type:
        trainer = PadimWeights2Trainer(args=args, train_loader=train_dataset, test_loader=test_normal, logger=logger)
    elif 'padim_weights' in args.model_type:
        trainer = PadimWeightsTrainer(args=args, train_loader=train_dataset, test_loader=test_normal, logger=logger)
    elif 'padim' in args.model_type:
        trainer = PadimTrainer(args=args, train_loader=train_dataset, test_loader=test_normal, logger=logger)
    elif 'ganomaly' in args.model_type:
        trainer = GanomalyTrainer(args=args, train_loader=train_dataset, test_loader=test_normal, logger=logger)
    elif 'diffusion' in args.model_type:
        trainer = UNETTrainer(args=args, train_loader=train_dataset, test_loader=test_normal, logger=logger)
    else:
        trainer = PadimTrainer(args=args, train_loader=train_dataset, test_loader=test_normal, logger=logger)

    trainer.start_time = start_time

    logger['time'] = start_time

    if args.validate:
        assert args.__contains__('model_name')
        # model_name = get_model_name(args, stats_saver)
        # logger = stats_saver.get_column(args.model_name)
        # setup_args(logger, args)

        trainer.load_model('res/stages-20_30-30_128/model_net_10_23_06_30_18_44_04.pt')
        # trainer.compute_latent_distribution()

        normal_losses, normal_prints = trainer.test()
        normal_losses = np.array(normal_losses)
        normal_prints = np.array(normal_prints)

        trainer.test_loader = test_abnormal
        abnormal_losses, abnormal_prints = trainer.test()
        abnormal_losses = np.array(abnormal_losses)
        abnormal_prints = np.array(abnormal_prints)

        # trainer.test_loader = test_normal
        # normal_losses, normal_prints = trainer.test()
        # normal_losses = np.array(normal_losses)
        # normal_prints = np.array(normal_prints)

        # trainer.test_loader = train_dataset_for_testing
        # train_losses, train_prints = trainer.test()
        # train_losses = np.array(train_losses)
        # train_prints = np.array(train_prints)

        stats_obj = {}
        # plot_bins(normal_losses, abnormal_losses,
        #           f"images/bins/{args.result_dir.split('/')[-1]}_run={start_time}")
        plot_auroc(normal_losses, abnormal_losses,
                   f"images/auroc/{args.result_dir.split('/')[-1]}_run={start_time}",
                   print_results=True, logger=logger, stats_obj=stats_obj)
        get_f1(normal_losses, abnormal_losses, print_results=True, logger=logger, stats_obj=stats_obj)
        get_accuracy(normal_losses, abnormal_losses, print_results=True, logger=logger)
        # print(compute_threshold(normal_losses, abnormal_losses))
        thresold = compute_real_threshold(normal_losses, abnormal_losses)

        stats_obj['normal_losses'] = normal_losses
        stats_obj['abnormal_losses'] = abnormal_losses
        stats_obj['normal_prints'] = normal_prints
        stats_obj['abnormal_prints'] = abnormal_prints

        np.savez(f"npz/stats_{start_time}", **stats_obj)
        # np.savez('prints', normal_prints=normal_prints, abnormal_prints=abnormal_prints, train_prints=train_prints, threshold=np.array(thresold))
    else:
        trainer.train()

        trainer.train_loader = train_dataset_for_testing
        train_losses, threshold, train_maps = trainer.get_threshold()
        train_maps = np.asarray(train_maps)

        normal_losses, normal_prints, normal_maps = trainer.test()
        normal_losses = np.array(normal_losses)
        normal_prints = np.array(normal_prints)
        normal_maps = np.array(normal_maps)

        trainer.test_loader = test_abnormal
        abnormal_losses, abnormal_prints, abnormal_maps = trainer.test()
        abnormal_losses = np.array(abnormal_losses)
        abnormal_prints = np.array(abnormal_prints)
        abnormal_maps = np.array(abnormal_maps)

        stats_obj = {}
        # plot_bins(normal_losses, abnormal_losses,
        #           f"images/bins/{args.result_dir.split('/')[-1]}_run={start_time}")

        # plot_auroc(normal_losses, abnormal_losses,
        #            f"images/auroc/{args.result_dir.split('/')[-1]}_run={start_time}",
        #            print_results=True, logger=logger, stats_obj=stats_obj)
        # get_accuracy(normal_losses, abnormal_losses, print_results=True, logger=logger)
        # get_f1(normal_losses, abnormal_losses, print_results=True, logger=logger, stats_obj=stats_obj)

        stats_obj['train_losses'] = train_losses
        stats_obj['threshold'] = threshold
        stats_obj['train_maps'] = train_maps
        stats_obj['normal_losses'] = normal_losses
        stats_obj['abnormal_losses'] = abnormal_losses
        stats_obj['normal_prints'] = normal_prints
        stats_obj['abnormal_prints'] = abnormal_prints
        stats_obj['normal_maps'] = normal_maps
        stats_obj['abnormal_maps'] = abnormal_maps

        if args.no_heatmap:
            custom_auroc_no_heatmap(args.dataset, args.subfolder, stats_obj)
        else:
            custom_auroc(args.dataset, args.subfolder, stats_obj)

        np.savez(f"npz/stats_{args.dataset}_{args.subfolder}_{args.model_type}_no_heatmap:{args.no_heatmap}_{start_time}", **stats_obj)

    stats_saver.add_column(logger)
    stats_saver.close_stats()


if __name__ == '__main__':
    main(parse_args())
    # plot_multiple_models(parse_args())
    # run_models(parse_args())