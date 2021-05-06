import os
import time
import json
from enum import Enum

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch_tools.modules import DataParallelPassthrough
from torch.optim import lr_scheduler
import torchvision
import optuna

from utils import make_noise, is_conditional
from train_log import MeanTracker
from visualization import make_interpolation_chart, fig_to_image
from latent_deformator import DeformatorType


class ShiftDistribution(Enum):
    NORMAL = 0,
    UNIFORM = 1,


class Params(object):
    def __init__(self, **kwargs):
        self.shift_scale = 6.0
        self.min_shift = 0.5
        self.shift_distribution = ShiftDistribution.UNIFORM

        self.z_std = 1.0

        self.torch_grad = False

        self.deformator_lr = 0.0001
        self.shift_predictor_lr = 0.0001
        self.n_steps = int(139500 + 1)
        self.batch_size = 8

        self.directions_count = 100
        self.max_latent_dim = None

        self.label_weight = 1.0
        self.shift_weight = 0.25

        self.steps_per_log = 10
        self.steps_per_save = 500  # 10000 # models
        self.steps_per_img_log = 1000  # 2000
        self.steps_per_backup = 500  # 2000 # checkpoint
        self.steps_per_weight_log = 1000  # 500

        self.truncation = None

        for key, val in kwargs.items():
            if val is not None:
                self.__dict__[key] = val


class Trainer(object):
    def __init__(self, params=Params(), out_dir='', verbose=False):
        if verbose:
            print('Trainer inited with:\n{}'.format(str(params.__dict__)))
        self.p = params
        self.cross_entropy = nn.CrossEntropyLoss()

        try_name = f"BS={self.p.batch_size} sft-lr={self.p.shift_predictor_lr:.7f} def-lr={self.p.deformator_lr:.7f} dir_n={self.p.directions_count} def-type={self.p.deformator}"
        self.log_dir = os.path.join(out_dir, try_name, 'logs')
        tb_dir = os.path.join(out_dir, try_name)
        self.writer = SummaryWriter(tb_dir)

        self.models_dir = os.path.join(out_dir, try_name, 'models')
        self.images_dir = os.path.join(self.log_dir, 'images')
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)

        self.checkpoint = os.path.join(out_dir, try_name, 'checkpoint.pt')
        if hasattr(params, 'warm_start'):
            self.warm_start = params.warm_start
        else:
            self.warm_start = self.checkpoint
        self.out_json = os.path.join(self.log_dir, 'stat.json')
        self.fixed_test_noise = None

    def make_shifts(self, latent_dim):
        target_indices = torch.randint(
            0, self.p.directions_count, [self.p.batch_size], device='cuda')
        if self.p.shift_distribution == ShiftDistribution.NORMAL:
            shifts = torch.randn(target_indices.shape, device='cuda')
        elif self.p.shift_distribution == ShiftDistribution.UNIFORM:
            shifts = 2.0 * torch.rand(target_indices.shape, device='cuda') - 1.0

        shifts = self.p.shift_scale * shifts
        shifts[(shifts < self.p.min_shift) & (shifts > 0)] = self.p.min_shift
        shifts[(shifts > -self.p.min_shift) & (shifts < 0)] = -self.p.min_shift

        try:
            latent_dim[0]
            latent_dim = list(latent_dim)
        except Exception:
            latent_dim = [latent_dim]

        z_shift = torch.zeros([self.p.batch_size] + latent_dim, device='cuda')
        for i, (index, val) in enumerate(zip(target_indices, shifts)):
            z_shift[i][index] += val

        return target_indices, shifts, z_shift

    def log_train(self, step, should_print=True, stats=()):
        if should_print:
            out_text = '{}% [step {}]'.format(int(100 * step / self.p.n_steps), step)
            for named_value in stats:
                out_text += (' | {}: {:.5f}'.format(*named_value))
            print(out_text)
        for named_value in stats:
            self.writer.add_scalar(named_value[0], named_value[1], step)

        with open(self.out_json, 'w') as out:
            stat_dict = {named_value[0]: named_value[1] for named_value in stats}
            json.dump(stat_dict, out)

    def log_interpolation(self, G, deformator, step):
        noise = make_noise(1, G.dim_z, self.p.z_std, self.p.truncation).cuda()
        if self.fixed_test_noise is None:
            self.fixed_test_noise = noise.clone()
        for z, prefix in zip([noise, self.fixed_test_noise], ['rand', 'fixed']):
            fig = make_interpolation_chart(
                G, deformator, z=z, std=self.p.z_std, shifts_r=self.p.shift_scale, shifts_count=3,
                dims_count=min(15, self.p.directions_count),
                dpi=500)
            self.writer.add_figure('{}_deformed_interpolation'.format(prefix), fig, step)
            fig_to_image(fig).convert("RGB").save(
                os.path.join(self.images_dir, '{}_{}.jpg'.format(prefix, step)))

    def start_from_checkpoint(self, deformator, shift_predictor, deformator_opt, shift_predictor_opt, scheduler_def,
                              scheduler_pred):
        step = 0
        if os.path.isfile(self.warm_start):
            state_dict = torch.load(self.warm_start)
            step = state_dict['step']
            deformator.load_state_dict(state_dict['deformator'])
            shift_predictor.load_state_dict(state_dict['shift_predictor'])
            if deformator_opt is not None: deformator_opt.load_state_dict(state_dict['deformator_opt'])
            shift_predictor_opt.load_state_dict(state_dict['shift_predictor_opt'])
            if scheduler_def is not None: scheduler_def.load_state_dict(state_dict['scheduler_def'])
            # scheduler_pred.load_state_dict(state_dict['scheduler_pred'])
            print('starting from step {}'.format(step))
        return step

    def start_from_saved_model(selfself, deformator, shift_predictor, deformator_opt, shift_predictor_opt,
                               scheduler_def, scheduler_pred, saved_model_folder_path, desired_step):
        deformator_path = os.path.join(saved_model_folder_path, f'deformator_{desired_step}.pt')
        predictor_path = os.path.join(saved_model_folder_path, f'shift_predictor_{desired_step}.pt')
        deformator_opt_path = os.path.join(saved_model_folder_path, f'deformator_opt_{desired_step}.pt')
        shift_predictor_opt_path = os.path.join(saved_model_folder_path, f'shift_predictor_opt_{desired_step}.pt')
        scheduler_def_path = os.path.join(saved_model_folder_path, f'scheduler_def_{desired_step}.pt')
        scheduler_pred_path = os.path.join(saved_model_folder_path, f'scheduler_pred_{desired_step}.pt')
        if os.path.isfile(deformator_path) and os.path.isfile(predictor_path):
            deformator.load_state_dict(torch.load(deformator_path))
            shift_predictor.load_state_dict(torch.load(predictor_path))
            deformator_opt.load_state_dict(torch.load(deformator_opt_path))
            shift_predictor_opt.load_state_dict(torch.load(shift_predictor_opt_path))
            scheduler_def.load_state_dict(torch.load(scheduler_def_path))
            scheduler_pred.load_state_dict(torch.load(scheduler_pred_path))
            print(f'|| Saved model loaded. Starting from step {desired_step}')
        return desired_step

    def save_checkpoint(self, deformator, shift_predictor, deformator_opt, shift_predictor_opt, scheduler_def,
                        scheduler_pred, step):
        state_dict = {
            'step': step,
            'deformator': deformator.state_dict(),
            'shift_predictor': shift_predictor.state_dict(),
            'deformator_opt': deformator_opt.state_dict() if deformator_opt is not None else None,
            'shift_predictor_opt': shift_predictor_opt.state_dict(),
            'scheduler_def': scheduler_def.state_dict() if scheduler_def is not None else None,
            'scheduler_pred': scheduler_pred.state_dict()
        }
        torch.save(state_dict, self.checkpoint)
        print("[Log] Checkpoint saved.")

    def save_models(self, deformator, shift_predictor, deformator_opt, shift_predictor_opt, scheduler_def,
                    scheduler_pred, step):
        torch.save(deformator.state_dict(),
                   os.path.join(self.models_dir, 'deformator_{}.pt'.format(step)))
        torch.save(shift_predictor.state_dict(),
                   os.path.join(self.models_dir, 'shift_predictor_{}.pt'.format(step)))
        if deformator_opt is not None: torch.save(deformator_opt.state_dict(),
                   os.path.join(self.models_dir, 'deformator_opt_{}.pt'.format(step)))
        torch.save(shift_predictor_opt.state_dict(),
                   os.path.join(self.models_dir, 'shift_predictor_opt_{}.pt'.format(step)))
        if scheduler_def is not None: torch.save(scheduler_def.state_dict(),
                   os.path.join(self.models_dir, 'scheduler_def_{}.pt'.format(step)))
        torch.save(scheduler_pred.state_dict(),
                   os.path.join(self.models_dir, 'scheduler_pred_{}.pt'.format(step)))

    def log_accuracy(self, G, deformator, shift_predictor, step):
        deformator.eval()
        shift_predictor.eval()

        accuracy = validate_classifier(G, deformator, shift_predictor, trainer=self)
        self.writer.add_scalar('accuracy', accuracy.item(), step)

        deformator.train()
        shift_predictor.train()
        return accuracy

    def log_weights(self, defmormator, shift_predictor, step):
        try:
            for name, weight in shift_predictor.named_parameters():
                self.writer.add_histogram(f"sft-{name}", weight, step)
                if type(weight.grad) != type(None):
                    self.writer.add_histogram(f"sft-{name}.grad", weight.grad, step)
                else:
                    self.writer.add_histogram(f"sft-{name}.grad", torch.ones_like(weight) * (-3.14), step)
        except:
            print("EXCEPTION")

        try:
            for name, weight in defmormator.named_parameters():
                self.writer.add_histogram(f"def-{name}", weight, step)
                if type(weight.grad) != type(None):
                    self.writer.add_histogram(f"def-{name}.grad", weight.grad, step)
                else:
                    self.writer.add_histogram(f"def-{name}.grad", torch.ones_like(weight) * (-3.14), step)
        except:
            print(("EXCEPTION"))

    def log_generated_images(self, G, step):
        generated_imgs = []
        for _ in range(2):
            z = make_noise(8, G.dim_z, self.p.z_std, self.p.truncation).cuda()
            imgs = G(z=z, reverse=True).clamp_(0, 1).cuda()
            generated_imgs.append(imgs)
        image_grid = torchvision.utils.make_grid(generated_imgs)
        self.writer.add_image("Generated Images", image_grid, step)

    def log(self, G, deformator, shift_predictor, deformator_opt, shift_predictor_opt, scheduler_def, scheduler_pred,
            step, avgs):

        values = {'loss': 0}

        if step % self.p.steps_per_log == 0:
            values = [avg.flush() for avg in avgs]
            self.log_train(step, True, values)

        if step % self.p.steps_per_backup == 0 and step > 0:
            self.save_checkpoint(deformator, shift_predictor, deformator_opt, shift_predictor_opt, scheduler_def,
                                 scheduler_pred, step)
            # self.accuracy = self.log_accuracy(G, deformator, shift_predictor, step)
            # print('[Log] Step {} accuracy: {:.3}'.format(step, self.accuracy.item()))

        if step % self.p.steps_per_save == 0 and step > 0:
            self.save_models(deformator, shift_predictor, deformator_opt, shift_predictor_opt, scheduler_def,
                             scheduler_pred, step)
            print('[Log] Model Saved.')

        if step % self.p.steps_per_weight_log == 0:
            self.log_weights(deformator, shift_predictor, step)

        if step % self.p.steps_per_img_log == 0 and step > 0:
            # self.log_interpolation(G, deformator, step)
            print('[Log] Image Interpolation is Done.')
            # self.log_generated_images(G, step)

        if step == self.p.n_steps - 1:
            self.writer.add_hparams(
                {"bs": self.p.batch_size, "sft-lr": self.p.shift_predictor_lr, "dfr-lr": self.p.deformator_lr},
                {"loss": values['loss']})

    def train(self, G, deformator, shift_predictor, trial, multi_gpu=False):
        # torch.autograd.set_detect_anomaly(True)
        G.cuda().eval()
        deformator.cuda().train()
        shift_predictor.cuda().train()

        should_gen_classes = is_conditional(G)
        if multi_gpu:
            G = DataParallelPassthrough(G, device_ids=[0]).to('cuda:0')

        # Optimizers
        deformator_opt = torch.optim.Adam(deformator.parameters(), lr=self.p.deformator_lr) \
            if deformator.type not in [DeformatorType.ID, DeformatorType.RANDOM] else None
        shift_predictor_opt = torch.optim.Adam(
            shift_predictor.parameters(), lr=self.p.shift_predictor_lr)

        # Optimization Scheduler
        scheduler_def = lr_scheduler.ReduceLROnPlateau(deformator_opt, 'min', min_lr=0.00001, factor=0.5, patience=1000,
                                                       verbose=True, threshold=0.001,
                                                       cooldown=500) if deformator.type not in [DeformatorType.ID,
                                                                                                DeformatorType.RANDOM] else None
        scheduler_pred = lr_scheduler.ReduceLROnPlateau(shift_predictor_opt, 'min', min_lr=0.00001, patience=1000,
                                                        verbose=True, threshold=0.001, cooldown=500)

        # Measures Trackers
        avgs = MeanTracker('class_correct_percent'), MeanTracker('loss'), MeanTracker('direction_loss'), MeanTracker(
            'shift_loss'), MeanTracker('learning_rate')
        avg_correct_percent, avg_loss, avg_label_loss, avg_shift_loss, avg_lr = avgs

        # Load the checkpoint (deformator's weights, shift_predictor's weight, step)

        recovered_step = self.start_from_checkpoint(deformator, shift_predictor, deformator_opt, shift_predictor_opt,
                                                    scheduler_def, scheduler_pred)
        # recovered_step = self.start_from_saved_model(deformator, shift_predictor,deformator_opt, shift_predictor_opt,
        #                                              scheduler_def, scheduler_pred,
        #                                              './anime_results_dir/BS=256 sft-lr=0.0100000 def-lr=0.0100000 dir_n=100 def-type=proj/models',
        #                                              5000)
        shift_predictor_opt.param_groups[0]['lr'] = 0.0001
        if recovered_step == self.p.n_steps - 1:
            print("[Trainer]: the model has been trained before.", "Trying next hyperparameters")
            return

        for step in range(recovered_step, self.p.n_steps, 1):
            if step < 11 or False: begin = time.time()

            G.zero_grad()
            deformator.zero_grad()
            shift_predictor.zero_grad()

            if deformator.type == DeformatorType.ID:
                target_indices, shifts, basis_shift = self.make_shifts(48 * 8 * 8)
            else:
                target_indices, shifts, basis_shift = self.make_shifts(deformator.input_dim)

            # Deformation
            shift = deformator(basis_shift)
            shift = shift.view([-1] + G.dim_z)

            # Image Generation
            # z = make_noise(self.p.batch_size, G.dim_z, self.p.z_std, self.p.truncation).cuda()
            # with torch.set_grad_enabled(self.p.torch_grad):
            #     imgs = G(z=z, reverse=True)  # .clamp_(0, 1)
            #     imgs_shifted = G.nvp_shifted(z, shift, reverse=True)  # .clamp_(0, 1)

            # Sometimes the generator make infinite values in output images and cause gradient explosion. This part checks
            # if there is inf values in images and so substitue the invalid images vith valid images (have no inf).

            # if (torch.sum(torch.isinf(imgs)) + torch.sum(torch.isinf(imgs_shifted))) > 0:
            #     print('| Invalid Images')
            #     valid = False
            #     while not valid:
            #         for i in range(len(imgs)):
            #             if torch.sum(torch.isinf(imgs[i])) + torch.sum(torch.isinf(imgs_shifted[i])) > 0:
            #                 print(f'| image #{i} has Inf values.')
            #                 zz = make_noise(1, G.dim_z, self.p.z_std, self.p.truncation).cuda()
            #                 imgs[i] = G(z=zz, reverse=True)[0]
            #                 imgs_shifted[i] = G.nvp_shifted(zz, shift[i].unsqueeze(0), reverse=True)[0]
            #         torch.cuda.empty_cache()
            #         valid = (torch.sum(torch.isinf(imgs)) + torch.sum(torch.isinf(imgs_shifted))) == 0

            while True:
                z = make_noise(self.p.batch_size, G.dim_z, self.p.z_std, self.p.truncation).cuda()
                with torch.set_grad_enabled(self.p.torch_grad):
                    imgs = G(z=z, reverse=True).to('cuda:0')
                    imgs_shifted = G(z=z + shift, reverse=True).to('cuda:0')

                if (torch.sum(torch.isinf(imgs)) + torch.sum(torch.isinf(imgs_shifted))) == 0:
                    imgs = torch.clamp(imgs, min=0, max=1)
                    imgs_shifted = torch.clamp(imgs_shifted, min=0, max=1)
                    break
                print("Inf values in generated images!   Repeat image generation")
                del imgs
                del imgs_shifted
                torch.cuda.empty_cache()

            if step % (self.p.steps_per_img_log / 2) == 0:
                image_grid = torchvision.utils.make_grid(list(imgs[0:8]) + list(imgs_shifted[0:8]))
                self.writer.add_image("Training Shifted Images", image_grid, step)

            logits, shift_prediction = shift_predictor(imgs, imgs_shifted)
            logit_loss = self.p.label_weight * self.cross_entropy(logits, target_indices)
            shift_loss = self.p.shift_weight * torch.mean(torch.abs(shift_prediction - shifts))

            # total loss
            loss = logit_loss + shift_loss
            self.loss = loss.detach().item()
            loss.backward()

            if deformator_opt is not None:
                deformator_opt.step()
            shift_predictor_opt.step()

            if deformator.type not in [DeformatorType.ID, DeformatorType.RANDOM] : scheduler_def.step(loss)
            # scheduler_pred.step(loss)

            # update statistics trackers
            avg_correct_percent.add(torch.mean(
                (torch.argmax(logits, dim=1) == target_indices).to(torch.float32)).detach())
            avg_loss.add(loss.item())
            avg_label_loss.add(logit_loss.item())
            avg_shift_loss.add(shift_loss)
            avg_lr.add(shift_predictor_opt.param_groups[0]['lr'])

            # Optuna, report intermediate objective value
            trial.report(avg_loss.mean(), step)

            self.log(G, deformator, shift_predictor, deformator_opt, shift_predictor_opt, scheduler_def, scheduler_pred,
                     step, avgs)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                self.writer.add_hparams(
                    {"bs": self.p.batch_size, "sft-lr": self.p.shift_predictor_lr, "dfr-lr": self.p.deformator_lr},
                    {"loss": self.loss})
                print("%%%% The trial was prouned %%%%")
                raise optuna.TrialPruned()

            if step < 11 or False: end = time.time(); print(f"Epoch {step} = {(end - begin):.2f} seconds.")

        return self.loss


@torch.no_grad()
def validate_classifier(G, deformator, shift_predictor, params_dict=None, trainer=None):
    n_steps = 10
    if trainer is None:
        trainer = Trainer(params=Params(**params_dict), verbose=False)

    percents = torch.empty([n_steps])
    for step in range(n_steps):
        z = make_noise(trainer.p.batch_size, G.dim_z, trainer.p.z_std, trainer.p.truncation).cuda()
        target_indices, shifts, basis_shift = trainer.make_shifts(deformator.input_dim)

        shift = deformator(basis_shift)

        with torch.set_grad_enabled(trainer.p.torch_grad):
            imgs = G(z=z, reverse=True).clamp_(0, 1).to('cuda:0')
        shift = shift.view([-1] + G.dim_z)
        imgs_shifted = G.nvp_shifted(z, shift, reverse=True).clamp_(0, 1).to('cuda:0')

        logits, _ = shift_predictor(imgs, imgs_shifted)
        percents[step] = (torch.argmax(logits, dim=1) == target_indices).to(torch.float32).mean()

    return percents.mean()
