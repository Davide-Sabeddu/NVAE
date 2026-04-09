# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
# ... (license unchanged)
# ---------------------------------------------------------------

import sys, os
sys.path.insert(0, '/lustre/fs1/home/da389032/NVAE')

import argparse
import numpy as np
from copy import deepcopy
from math import floor, sqrt

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.multiprocessing import Process
from torch.cuda.amp import autocast, GradScaler
import torchvision
from torchvision.models import inception_v3, Inception_V3_Weights
import matplotlib
matplotlib.use('Agg')          # ← FIX: non-interactive backend for HPC nodes
import matplotlib.pyplot as plt

from model import AutoEncoder
from thirdparty.adamax import Adamax
import utils
import datasets

from fid.fid_score import compute_statistics_of_generator, load_statistics, calculate_frechet_distance
from fid.inception import InceptionV3

from projorg import checkpointsdir, datadir, setup_environment, upload_to_cloud
from sips.utils import plot_loss, plotsdir


class NVAETrainer:

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

        self.logging = utils.Logger(args.global_rank, args.save)
        self.writer  = utils.Writer(args.global_rank, args.save)

        self.train_queue, self.valid_queue, num_classes = datasets.get_loaders(args)
        args.num_total_iter = len(self.train_queue) * args.epochs
        self.warmup_iters   = len(self.train_queue) * args.warmup_epochs
        self.swa_start      = len(self.train_queue) * (args.epochs - 1)

        self.arch_instance = utils.get_arch_cells(args.arch_instance)
        self.model = AutoEncoder(args, self.writer, self.arch_instance).cuda()

        self.logging.info('args = %s', args)
        self.logging.info('param size = %fM', utils.count_parameters_in_M(self.model))
        self.logging.info(
            'groups per scale: %s, total_groups: %d',
            self.model.groups_per_scale,
            sum(self.model.groups_per_scale),
        )

        if args.fast_adamax:
            self.optimizer = Adamax(
                self.model.parameters(), args.learning_rate,
                weight_decay=args.weight_decay, eps=1e-3,
            )
        else:
            self.optimizer = torch.optim.Adamax(
                self.model.parameters(), args.learning_rate,
                weight_decay=args.weight_decay, eps=1e-3,
            )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            float(args.epochs - args.warmup_epochs - 1),
            eta_min=args.learning_rate_min,
        )
        self.grad_scalar = GradScaler(2 ** 10)

        num_output     = utils.num_output(args.dataset)
        self.bpd_coeff = 1.0 / np.log(2.0) / num_output

        self.train_obj       = []
        self.train_obj_epoch = []
        self.val_obj         = []
        self.val_epochs      = []   # ← FIX: track which epoch each val entry belongs to

        self._metrics = {
            'train_nelbo':     [],
            'valid_nelbo':     [],
            'valid_neg_log_p': [],
            'valid_bpd_elbo':  [],
            'valid_bpd_log_p': [],
        }

        weights = Inception_V3_Weights.DEFAULT
        self.feature_net = inception_v3(weights=weights, transform_input=True)
        self.feature_net.fc = nn.Identity()
        self.feature_net.eval().to('cuda')

    # ------------------------------------------------------------------ #
    #  Checkpoint helpers                                                  #
    # ------------------------------------------------------------------ #

    def save_checkpoint(self, epoch: int, global_step: int) -> None:
        if self.args.global_rank != 0:
            return

        ckpt_dir = checkpointsdir(self.args.save)
        os.makedirs(ckpt_dir, exist_ok=True)

        payload = {
            'epoch':             epoch + 1,
            'global_step':       global_step,
            'state_dict':        self.model.state_dict(),
            'optimizer':         self.optimizer.state_dict(),
            'grad_scalar':       self.grad_scalar.state_dict(),
            'scheduler':         self.scheduler.state_dict(),
            'args':              self.args,
            'arch_instance':     self.arch_instance,
            'train_obj':         self.train_obj,
            'train_obj_epoch':   self.train_obj_epoch,
            'val_obj':           self.val_obj,
            'val_epochs':        self.val_epochs,   # ← FIX: persist val epoch indices
            '_metrics':          self._metrics,
        }

        epoch_path = os.path.join(ckpt_dir, f'checkpoint_{epoch:05d}.pth')
        torch.save(payload, epoch_path)
        self.logging.info('saved epoch snapshot → %s', epoch_path)

        latest_path = os.path.join(ckpt_dir, 'checkpoint_latest.pth')
        torch.save(payload, latest_path)

    def load_checkpoint(self, epoch: int | None = None) -> tuple[int, int]:
        ckpt_dir = checkpointsdir(self.args.save)

        if epoch is not None:
            file_to_load = os.path.join(ckpt_dir, f'checkpoint_{epoch:05d}.pth')
            if not os.path.exists(file_to_load):
                self.logging.info('requested checkpoint not found: %s', file_to_load)
                return 0, 0
        else:
            latest = os.path.join(ckpt_dir, 'checkpoint_latest.pth')
            if os.path.exists(latest):
                file_to_load = latest
            else:
                snaps = [
                    f for f in os.listdir(ckpt_dir)
                    if f.startswith('checkpoint_') and f.endswith('.pth')
                ]
                if not snaps:
                    self.logging.info('no checkpoint found, starting from scratch')
                    return 0, 0
                file_to_load = os.path.join(
                    ckpt_dir,
                    max(snaps, key=lambda f: int(f.split('_')[-1].split('.')[0])),
                )

        self.logging.info('loading checkpoint: %s', file_to_load)
        ckpt = torch.load(file_to_load, map_location='cpu', weights_only=False)

        self.model.load_state_dict(ckpt['state_dict'])
        self.model.cuda()
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.grad_scalar.load_state_dict(ckpt['grad_scalar'])
        self.scheduler.load_state_dict(ckpt['scheduler'])

        self.train_obj       = ckpt.get('train_obj',       [])
        self.train_obj_epoch = ckpt.get('train_obj_epoch', [])
        self.val_obj         = ckpt.get('val_obj',         [])
        self.val_epochs      = ckpt.get('val_epochs',      [])   # ← FIX: restore
        self._metrics        = ckpt.get('_metrics',        self._metrics)

        global_step = ckpt.get('global_step', 0)
        init_epoch  = ckpt.get('epoch', 0)

        self.logging.info(
            'resumed from epoch %d  (global_step=%d, '
            'train_step entries=%d, train_epoch entries=%d, val entries=%d)',
            init_epoch, global_step,
            len(self.train_obj), len(self.train_obj_epoch), len(self.val_obj),
        )

        for step, val in self._metrics['train_nelbo']:
            self.writer.add_scalar('train/nelbo', val, step)
        for ep, val in self._metrics['valid_nelbo']:
            self.writer.add_scalar('val/nelbo', val, ep)
        for ep, val in self._metrics['valid_neg_log_p']:
            self.writer.add_scalar('val/neg_log_p', val, ep)
        for ep, val in self._metrics['valid_bpd_elbo']:
            self.writer.add_scalar('val/bpd_elbo', val, ep)
        for ep, val in self._metrics['valid_bpd_log_p']:
            self.writer.add_scalar('val/bpd_log_p', val, ep)

        return init_epoch, global_step

    # ------------------------------------------------------------------ #
    #  Feature extraction                                                  #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor, chunk_size: int = 16) -> torch.Tensor:
        self.feature_net.eval()
        feats = []
        for i in range(0, x.shape[0], chunk_size):
            chunk = x[i : i + chunk_size].cuda()
            chunk = nn.functional.interpolate(
                chunk, size=(299, 299), mode='bilinear', align_corners=False
            )
            feats.append(self.feature_net(chunk).cpu())
        return torch.cat(feats, dim=0)

    # ------------------------------------------------------------------ #
    #  Validation                                                          #
    # ------------------------------------------------------------------ #

    def _validate(self, num_samples: int = 10) -> tuple[float, float]:
        if self.args.distributed:
            dist.barrier()

        nelbo_avg     = utils.AvgrageMeter()
        neg_log_p_avg = utils.AvgrageMeter()
        self.model.eval()

        for step, x in enumerate(self.valid_queue):
            x = x[0] if len(x) > 1 else x
            x = x.cuda()
            x = utils.pre_process(x, self.args.num_x_bits)

            with torch.no_grad():
                nelbo_samples, log_iw_samples = [], []
                for _ in range(num_samples):
                    logits, log_q, log_p, kl_all, _ = self.model(x)
                    output      = self.model.decoder_output(logits)
                    recon_loss  = utils.reconstruction_loss(output, x, crop=self.model.crop_output)
                    balanced_kl, _, _ = utils.kl_balancer(kl_all, kl_balance=False)
                    nelbo_samples.append(recon_loss + balanced_kl)
                    log_iw_samples.append(
                        utils.log_iw(output, x, log_q, log_p, crop=self.model.crop_output)
                    )

                nelbo = torch.mean(torch.stack(nelbo_samples, dim=1))
                log_p = torch.mean(
                    torch.logsumexp(torch.stack(log_iw_samples, dim=1), dim=1)
                    - np.log(num_samples)
                )

            nelbo_avg.update(nelbo.data, x.size(0))
            neg_log_p_avg.update(-log_p.data, x.size(0))

        utils.average_tensor(nelbo_avg.avg, self.args.distributed)
        utils.average_tensor(neg_log_p_avg.avg, self.args.distributed)

        if self.args.distributed:
            dist.barrier()

        self.logging.info(
            'val step %d  NELBO %.4f  neg_log_p %.4f',
            step, nelbo_avg.avg, neg_log_p_avg.avg,
        )
        return float(neg_log_p_avg.avg), float(nelbo_avg.avg)

    # ------------------------------------------------------------------ #
    #  Test / visualisation                                                #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def test(self, epoch: int, global_step: int, n_gen: int = 64) -> None:
        if self.args.global_rank != 0:
            return

        self.model.eval()
        plot_root = plotsdir(self.args.save)
        os.makedirs(plot_root, exist_ok=True)

        n_side = int(floor(sqrt(n_gen)))
        n_gen  = n_side * n_side

        # ── 1. Sample grids ───────────────────────────────────────────
        for temp in [0.7, 0.8, 0.9, 1.0]:
            logits     = self.model.sample(n_gen, temp)
            output     = self.model.decoder_output(logits)
            output_img = (
                output.mean
                if isinstance(output, torch.distributions.Bernoulli)
                else output.sample(temp)
            )
            grid_path = os.path.join(
                plot_root, f'{epoch:05d}_samples_t{temp:.1f}.png'
            )
            torchvision.utils.save_image(output_img, grid_path, nrow=n_side, normalize=True)
            self.logging.info('saved sample grid → %s', grid_path)
            tiled = utils.tile_image(output_img, n_side)
            self.writer.add_image(f'generated_{temp:.1f}', tiled, global_step)

        # ── 2. Nearest-neighbour comparison ───────────────────────────
        logits   = self.model.sample(n_gen, 0.7)
        output   = self.model.decoder_output(logits)
        gen_imgs = (
            output.mean
            if isinstance(output, torch.distributions.Bernoulli)
            else output.mean()
        )

        val_pool, count = [], 0
        for x_batch, *_ in self.valid_queue:
            val_pool.append(x_batch)
            count += x_batch.shape[0]
            if count >= 5000:
                break
        val_pool = torch.cat(val_pool, dim=0)[:5000]

        def to_01(t):
            lo, hi = t.min(), t.max()
            return (t - lo) / (hi - lo + 1e-8)

        gen_01 = to_01(gen_imgs.cpu())
        val_01 = to_01(val_pool.cpu())

        self.logging.info('extracting Inception features for NN comparison …')
        gen_feat = self.extract_features(gen_01)
        val_feat = self.extract_features(val_01)

        dists   = torch.cdist(gen_feat, val_feat, p=2)
        nn_idx  = torch.argmin(dists, dim=1)
        nearest = val_01[nn_idx]

        comparison = torch.empty((n_gen * 2, *gen_01.shape[1:]), dtype=gen_01.dtype)
        comparison[0::2] = gen_01
        comparison[1::2] = nearest

        nn_path = os.path.join(plot_root, f'{epoch:05d}_gen_vs_nearest_val.png')
        torchvision.utils.save_image(comparison, nn_path, nrow=n_side * 2, padding=2)
        self.logging.info('saved NN comparison → %s', nn_path)

        # ── 3. Loss curves ────────────────────────────────────────────
        # FIX: train and val are sampled at different frequencies, so we
        # must give each curve its own x-axis instead of assuming they
        # share the same integer indices.
        #
        # train_obj_epoch[i] is always from epoch i (0-indexed).
        # val_obj[i]         is from val_epochs[i]  (sparse subset).
        #
        # We fall back gracefully if val_epochs is empty (first epoch).
        if len(self.train_obj_epoch) > 0:
            fig, ax = plt.subplots(figsize=(8, 5))

            train_x = list(range(len(self.train_obj_epoch)))
            ax.plot(train_x, self.train_obj_epoch, label='train NELBO', color='steelblue')

            if len(self.val_obj) > 0:
                # val_epochs may be missing in old checkpoints; fall back to
                # evenly-spaced positions so the plot still renders.
                if len(self.val_epochs) == len(self.val_obj):
                    val_x = self.val_epochs
                else:
                    val_x = np.linspace(0, len(self.train_obj_epoch) - 1,
                                        len(self.val_obj)).tolist()
                ax.plot(val_x, self.val_obj, label='val NELBO',
                        color='tomato', marker='o', markersize=4)

            ax.set_xlabel('epoch')
            ax.set_ylabel('NELBO')
            ax.set_title(f'Loss curves — epoch {epoch}')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plot_path = os.path.join(plot_root, f'{epoch:05d}_loss.png')
            fig.savefig(plot_path, dpi=120, bbox_inches='tight')
            plt.close(fig)
            self.logging.info('saved loss plot → %s', plot_path)

    # ------------------------------------------------------------------ #
    #  Single training step                                                #
    # ------------------------------------------------------------------ #

    def _train_step(self, x, global_step, alpha_i):
        args = self.args

        if global_step < self.warmup_iters:
            lr = args.learning_rate * float(global_step) / self.warmup_iters
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr

        self.optimizer.zero_grad()
        with autocast():
            logits, log_q, log_p, kl_all, kl_diag = self.model(x)
            output = self.model.decoder_output(logits)

            kl_coeff = utils.kl_coeff(
                global_step,
                args.kl_anneal_portion * args.num_total_iter,
                args.kl_const_portion  * args.num_total_iter,
                args.kl_const_coeff,
            )

            recon_loss  = utils.reconstruction_loss(output, x, crop=self.model.crop_output)
            balanced_kl, kl_coeffs, kl_vals = utils.kl_balancer(
                kl_all, kl_coeff, kl_balance=True, alpha_i=alpha_i
            )

            nelbo_batch = recon_loss + balanced_kl
            loss        = torch.mean(nelbo_batch)
            norm_loss   = self.model.spectral_norm_parallel()
            bn_loss     = self.model.batchnorm_loss()

            if args.weight_decay_norm_anneal:
                assert args.weight_decay_norm_init > 0 and args.weight_decay_norm > 0
                wdn_coeff = np.exp(
                    (1.0 - kl_coeff) * np.log(args.weight_decay_norm_init)
                    + kl_coeff        * np.log(args.weight_decay_norm)
                )
            else:
                wdn_coeff = args.weight_decay_norm

            loss = loss + norm_loss * wdn_coeff + bn_loss * wdn_coeff

        self.grad_scalar.scale(loss).backward()
        utils.average_gradients(self.model.parameters(), args.distributed)
        self.grad_scalar.step(self.optimizer)
        self.grad_scalar.update()

        return loss, norm_loss, bn_loss, kl_coeff, kl_coeffs, kl_vals, kl_diag, output, recon_loss, kl_all, wdn_coeff

    # ------------------------------------------------------------------ #
    #  Training loop                                                       #
    # ------------------------------------------------------------------ #

    def train(self) -> None:
        args = self.args
        init_epoch, global_step = 0, 0

        if args.cont_training:
            init_epoch, global_step = self.load_checkpoint()

        alpha_i = utils.kl_balancer_coeff(
            num_scales=self.model.num_latent_scales,
            groups_per_scale=self.model.groups_per_scale,
            fun='square',
        )
        nelbo = utils.AvgrageMeter()

        for epoch in range(init_epoch, args.epochs):

            if args.distributed:
                self.train_queue.sampler.set_epoch(global_step + args.seed)
                self.valid_queue.sampler.set_epoch(0)

            if epoch > args.warmup_epochs:
                self.scheduler.step()

            self.logging.info('epoch %d', epoch)
            self.model.train()
            nelbo.reset()

            for step, x in enumerate(self.train_queue):
                x = x[0] if len(x) > 1 else x
                x = x.cuda()
                x = utils.pre_process(x, args.num_x_bits)

                if step % 100 == 0:
                    utils.average_params(self.model.parameters(), args.distributed)

                (
                    loss, norm_loss, bn_loss,
                    kl_coeff, kl_coeffs, kl_vals,
                    kl_diag, output, recon_loss, kl_all, wdn_coeff,
                ) = self._train_step(x, global_step, alpha_i)

                nelbo.update(loss.data, 1)
                self.train_obj.append(float(loss.data))

                if (global_step + 1) % 100 == 0:
                    if (global_step + 1) % 1000 == 0:
                        n_sq = int(floor(sqrt(x.size(0))))
                        x_img      = x[:n_sq * n_sq]
                        output_img = (
                            output.mean
                            if isinstance(output, torch.distributions.Bernoulli)
                            else output.sample()
                        )[:n_sq * n_sq]
                        in_out = torch.cat(
                            (utils.tile_image(x_img, n_sq),
                             utils.tile_image(output_img, n_sq)),
                            dim=2,
                        )
                        self.writer.add_image('reconstruction', in_out, global_step)

                    self.writer.add_scalar('train/norm_loss',  norm_loss, global_step)
                    self.writer.add_scalar('train/bn_loss',    bn_loss,   global_step)
                    self.writer.add_scalar('train/norm_coeff', wdn_coeff, global_step)

                    utils.average_tensor(nelbo.avg, args.distributed)
                    self.logging.info('train %d %f', global_step, nelbo.avg)
                    self.writer.add_scalar('train/nelbo_avg',  nelbo.avg, global_step)
                    self.writer.add_scalar(
                        'train/lr',
                        self.optimizer.state_dict()['param_groups'][0]['lr'],
                        global_step,
                    )
                    self.writer.add_scalar('train/nelbo_iter', loss,       global_step)
                    self.writer.add_scalar(
                        'train/kl_iter',
                        torch.mean(sum(kl_all)),
                        global_step,
                    )
                    self.writer.add_scalar(
                        'train/recon_iter',
                        torch.mean(utils.reconstruction_loss(output, x, crop=self.model.crop_output)),
                        global_step,
                    )
                    self.writer.add_scalar('kl_coeff/coeff', kl_coeff, global_step)

                    total_active = 0
                    for i, kl_diag_i in enumerate(kl_diag):
                        utils.average_tensor(kl_diag_i, args.distributed)
                        num_active    = torch.sum(kl_diag_i > 0.1).detach()
                        total_active += num_active
                        self.writer.add_scalar(f'kl/active_{i}',      num_active,    global_step)
                        self.writer.add_scalar(f'kl_coeff/layer_{i}', kl_coeffs[i], global_step)
                        self.writer.add_scalar(f'kl_vals/layer_{i}',  kl_vals[i],   global_step)
                    self.writer.add_scalar('kl/total_active', total_active, global_step)

                global_step += 1

            # ── End-of-epoch bookkeeping ──────────────────────────────
            utils.average_tensor(nelbo.avg, args.distributed)
            train_nelbo = float(nelbo.avg)
            self.logging.info('train_nelbo %f', train_nelbo)

            self.train_obj_epoch.append(train_nelbo)
            self._metrics['train_nelbo'].append((global_step, train_nelbo))
            self.writer.add_scalar('train/nelbo', train_nelbo, global_step)

            self.model.eval()
            eval_freq = 1 if args.epochs <= 50 else 20
            do_eval   = (epoch % eval_freq == 0) or (epoch == args.epochs - 1)

            if do_eval:
                valid_neg_log_p, valid_nelbo = self._validate(num_samples=10)
                self.logging.info('valid_nelbo %f',     valid_nelbo)
                self.logging.info('valid neg log p %f', valid_neg_log_p)
                self.logging.info('valid bpd elbo %f',  valid_nelbo * self.bpd_coeff)
                self.logging.info('valid bpd log p %f', valid_neg_log_p * self.bpd_coeff)

                self.val_obj.append(valid_nelbo)
                self.val_epochs.append(epoch)   # ← FIX: record the epoch index

                self._metrics['valid_nelbo'].append((epoch, valid_nelbo))
                self._metrics['valid_neg_log_p'].append((epoch, valid_neg_log_p))
                self._metrics['valid_bpd_elbo'].append((epoch, valid_nelbo * self.bpd_coeff))
                self._metrics['valid_bpd_log_p'].append((epoch, valid_neg_log_p * self.bpd_coeff))

                self.writer.add_scalar('val/neg_log_p', valid_neg_log_p,                   epoch)
                self.writer.add_scalar('val/nelbo',     valid_nelbo,                       epoch)
                self.writer.add_scalar('val/bpd_log_p', valid_neg_log_p * self.bpd_coeff, epoch)
                self.writer.add_scalar('val/bpd_elbo',  valid_nelbo     * self.bpd_coeff, epoch)

                self.test(epoch, global_step)

            save_freq = int(np.ceil(args.epochs / 100))
            if epoch % save_freq == 0 or epoch == args.epochs - 1:
                self.logging.info('saving the model.')
                self.save_checkpoint(epoch, global_step)

        # ── Final validation ──────────────────────────────────────────
        valid_neg_log_p, valid_nelbo = self._validate(num_samples=1000)
        self.logging.info('final valid nelbo %f',     valid_nelbo)
        self.logging.info('final valid neg log p %f', valid_neg_log_p)

        self.val_obj.append(valid_nelbo)
        self.val_epochs.append(args.epochs)   # ← FIX

        self._metrics['valid_nelbo'].append((args.epochs, valid_nelbo))
        self._metrics['valid_neg_log_p'].append((args.epochs, valid_neg_log_p))
        self._metrics['valid_bpd_elbo'].append((args.epochs, valid_nelbo * self.bpd_coeff))
        self._metrics['valid_bpd_log_p'].append((args.epochs, valid_neg_log_p * self.bpd_coeff))

        self.writer.add_scalar('val/neg_log_p', valid_neg_log_p,                   args.epochs)
        self.writer.add_scalar('val/nelbo',     valid_nelbo,                       args.epochs)
        self.writer.add_scalar('val/bpd_log_p', valid_neg_log_p * self.bpd_coeff, args.epochs)
        self.writer.add_scalar('val/bpd_elbo',  valid_nelbo     * self.bpd_coeff, args.epochs)

        self.test(args.epochs, global_step)
        self.writer.close()

    # ------------------------------------------------------------------ #
    #  FID (unchanged)                                                     #
    # ------------------------------------------------------------------ #

    def _create_generator(self, batch_size, num_total_samples):
        num_iters = int(np.ceil(num_total_samples / batch_size))
        for _ in range(num_iters):
            with torch.no_grad():
                logits     = self.model.sample(batch_size, 1.0)
                output     = self.model.decoder_output(logits)
                output_img = (
                    output.mean
                    if isinstance(output, torch.distributions.Bernoulli)
                    else output.mean()
                )
            yield output_img.float()

    def compute_fid(self, total_fid_samples):
        args     = self.args
        dims     = 2048
        device   = 'cuda'
        num_gpus  = args.num_process_per_node * args.num_proc_node
        n_per_gpu = int(np.ceil(total_fid_samples / num_gpus))

        g = self._create_generator(args.batch_size, n_per_gpu)
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        inc_model = InceptionV3([block_idx], model_dir=args.fid_dir).to(device)
        m, s = compute_statistics_of_generator(
            g, inc_model, args.batch_size, dims, device, max_samples=n_per_gpu
        )

        m = torch.from_numpy(m).cuda()
        s = torch.from_numpy(s).cuda()
        utils.average_tensor(m, args.distributed)
        utils.average_tensor(s, args.distributed)
        m, s = m.cpu().numpy(), s.cpu().numpy()

        path   = os.path.join(args.fid_dir, args.dataset + '.npz')
        m0, s0 = load_statistics(path)
        return calculate_frechet_distance(m0, s0, m, s)


# ─────────────────────────────────────────────────────────────────────────────
#  Distributed helpers
# ─────────────────────────────────────────────────────────────────────────────

def init_processes(rank, size, fn, args):
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = '6020'
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(
        backend='nccl', init_method='env://', rank=rank, world_size=size
    )
    fn(args)
    dist.destroy_process_group()


def run(args):
    trainer = NVAETrainer(args)

    if args.phase == 'train':
        trainer.train()

    if args.phase in ('test', 'eval'):
        epoch_to_load = getattr(args, 'testing_epoch', None)
        trainer.load_checkpoint(epoch=epoch_to_load)
        trainer.model.eval()
        trainer.test(epoch=epoch_to_load or 0, global_step=0, n_gen=64)

    if getattr(args, 'upload', False):
        upload_to_cloud(args, rclone_remote='UCFOneDrive')


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    args = setup_environment("nvae_cifar10.json")
    args.save = args.root + '/eval-' + args.save
    utils.create_exp_dir(args.save)

    size = args.num_process_per_node
    if size > 1:
        args.distributed = True
        processes = []
        for rank in range(size):
            args.local_rank  = rank
            args.global_rank = rank + args.node_rank * size
            global_size      = args.num_proc_node * size
            print(f'Node rank {args.node_rank}, local proc {rank}, global proc {args.global_rank}')
            p = Process(target=init_processes,
                        args=(args.global_rank, global_size, run, args))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        print('starting in debug mode (single GPU)')
        args.distributed = True
        init_processes(0, size, run, args)