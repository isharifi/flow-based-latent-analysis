import os
import pdb
import traceback
import argparse
import random
import torch
import numpy as np
from itertools import product

import matplotlib
import optuna

matplotlib.use("Agg")

from constants import DEFORMATOR_TYPE_DICT, SHIFT_DISTRIDUTION_DICT, WEIGHTS
from loading import load_generator
from latent_deformator import LatentDeformator
from latent_shift_predictor import LatentShiftPredictor, LeNetShiftPredictor
from trainer import Trainer, Params
from visualization import inspect_all_directions, inspect_all_directions_per_direction
from utils import make_noise, save_command_run_params
from random_generator import random_generator
from pseudo_label.DVN import compute_DVN

from MyDeformator import MyDeformator


def main():
    parser = argparse.ArgumentParser(description='Latent space rectification')
    for key, val in Params().__dict__.items():
        target_type = type(val) if val is not None else int
        parser.add_argument('--{}'.format(key), type=target_type, default=None)

    parser.add_argument('--out', type=str, required=True, help='results directory')
    parser.add_argument('--gan_type', type=str, choices=WEIGHTS.keys(), help='generator model type')
    parser.add_argument('--gan_weights', type=str, default=None, help='path to generator weights')
    parser.add_argument('--target_class', nargs='+', type=int, default=[239],
                        help='classes to use for conditional GANs')

    parser.add_argument('--warm_start', type=str, default=None, help='path to pretrained checkpoint')

    parser.add_argument('--deformator', type=str, default='ortho',
                        choices=DEFORMATOR_TYPE_DICT.keys(), help='deformator type')
    parser.add_argument('--deformator_random_init', type=bool, default=True)

    parser.add_argument('--shift_predictor_size', type=int, help='reconstructor resolution')
    parser.add_argument('--shift_predictor', type=str,
                        choices=['ResNet', 'LeNet'], default='ResNet', help='reconstructor type')
    parser.add_argument('--shift_distribution_key', type=str,
                        choices=SHIFT_DISTRIDUTION_DICT.keys())

    parser.add_argument('--seed', type=int, default=random.randint(0,1000))
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--multi_gpu', type=bool, default=False,
                        help='Run generator in parallel. Be aware of old pytorch versions:\
                              https://github.com/pytorch/pytorch/issues/17345')
    # model-specific
    parser.add_argument('--w_shift', type=bool, default=True,
                        help='latent directions search in w-space for StyleGAN')

    args = parser.parse_args()

    torch.cuda.set_device(args.device)

    torch.backends.cudnn.deterministic = False
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed + 1)
    torch.random.manual_seed(args.seed + 1)
    np.random.seed(args.seed + 1)

    save_command_run_params(args)

    # init models
    if args.gan_weights is not None:
        weights_path = args.gan_weights
    else:
        if args.gan_type is not "GLOW_pt_celeba":
            weights_path = WEIGHTS[args.gan_type]

    def objective(trial: optuna.trial.Trial):
        print(f"[Optuna]: Trial #{trial.number}")

        G = load_generator(args.__dict__, weights_path, args.w_shift)
        params = Params(**args.__dict__)

        params.batch_size = 128
        params.directions_count = 200
        params.deformator_lr = trial.suggest_float("deformator_lr", 1e-4, 1e-4, log=True)
        params.shift_predictor_lr = params.deformator_lr
        # params.torch_grad = True if trial.suggest_int("torch_grad", 0, 1) else False

        deformator = LatentDeformator(shift_dim=G.dim_shift,
                                      input_dim=params.directions_count,
                                      out_dim=params.max_latent_dim,
                                      type=DEFORMATOR_TYPE_DICT[args.deformator],
                                      random_init=args.deformator_random_init
                                      ).cuda()

        # deformator = MyDeformator(48*8*8, 48*8*8)

        if args.shift_predictor == 'ResNet':
            shift_predictor = LatentShiftPredictor(params.directions_count, args.shift_predictor_size).cuda()
        elif args.shift_predictor == 'LeNet':
            shift_predictor = LeNetShiftPredictor(
                params.directions_count, 1 if args.gan_type == 'SN_MNIST' else 3).cuda()

        # training
        args.shift_distribution = SHIFT_DISTRIDUTION_DICT[args.shift_distribution_key]

        # update dims with respect to the deformator if some of params are None
        params.directions_count = int(deformator.input_dim)
        params.max_latent_dim = int(deformator.out_dim)

        trainer = Trainer(params, out_dir=args.out, verbose=True)
        loss = trainer.train(G, deformator, shift_predictor, multi_gpu=args.multi_gpu, trial=trial)

        # my_save_results_charts(G, deformator, params, trainer.log_dir)
        compute_DVN(G, deformator, f'DVN_{args.gan_type}_{args.deformator}_{params.directions_count}.pt')

        return loss

    print("[Optuna]: Creating the study ...")
    # study_name = "glow_voynov"
    # study_name = "glow_voynov_id"
    study_name = f'{args.gan_type}_{args.deformator}'
    study = optuna.create_study(study_name=study_name,
                                storage=f"sqlite:///optuna_{study_name}.db",
                                load_if_exists=True,
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5,
                                                                   n_warmup_steps=450,
                                                                   interval_steps=20))
    run_specific_trial = False

    if run_specific_trial:
        for i in range(len(study.trials)):
            print(f'trial id: {i}', study.trials[i].params, study.trials[i].state)

        print("running a specific trial ...")

        # random
        # trial = study.trials[7]

        # id
        # trial = study.trials[11]

        # ours
        trial = study.trials[31]

        # trial.params['deformator_lr'] = 0.001

        # trial.params['deformator_lr'] += 0.000001
        objective(trial)
    else:
        print("[Optuna]: Starting the optimization ...")
        study.optimize(objective, n_trials=1)


def my_save_results_charts(G, deformator, params, out_dir):
    print("[Charts]: Starting creating visualization charts:")
    deformator.eval()
    G.eval()
    z = None  # make_noise(3, G.dim_z, params.z_std, params.truncation).cuda()
    inspect_all_directions_per_direction(
        G, deformator, os.path.join(out_dir, 'charts_s{}'.format(int(2 * params.shift_scale))),
        directions_count=params.directions_count, zs=z, std=params.z_std, shifts_r=2 * params.shift_scale)
    inspect_all_directions_per_direction(
        G, deformator, os.path.join(out_dir, 'charts_s{}'.format(int(params.shift_scale))),
        directions_count=params.directions_count, zs=z, std=params.z_std, shifts_r=params.shift_scale)
    inspect_all_directions_per_direction(
        G, deformator, os.path.join(out_dir, 'charts_s{}'.format(int(3 * params.shift_scale))),
        directions_count=params.directions_count, zs=z, std=params.z_std, shifts_r=3 * params.shift_scale)


def save_results_charts(G, deformator, params, out_dir):
    print("[Charts]: Starting creating visualization charts:")
    deformator.eval()
    G.eval()
    z = make_noise(3, G.dim_z, params.z_std, params.truncation).cuda()
    inspect_all_directions_per_direction(
        G, deformator, os.path.join(out_dir, 'charts_s{}'.format(int(2 * params.shift_scale))),
        directions_count=params.directions_count, zs=z, std=params.z_std, shifts_r=2 * params.shift_scale)
    inspect_all_directions_per_direction(
        G, deformator, os.path.join(out_dir, 'charts_s{}'.format(int(params.shift_scale))),
        directions_count=params.directions_count, zs=z, std=params.z_std, shifts_r=params.shift_scale)
    inspect_all_directions_per_direction(
        G, deformator, os.path.join(out_dir, 'charts_s{}'.format(int(3 * params.shift_scale))),
        directions_count=params.directions_count, zs=z, std=params.z_std, shifts_r=3 * params.shift_scale)


if __name__ == '__main__':
    main()
