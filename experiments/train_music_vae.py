import numpy as np
from pathlib import Path
import time
import functools
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sacred import Experiment
from sacred.observers import FileStorageObserver
import pypianoroll as pp

from smg.common.loggers import CompositeLogger, TensorBoardLogger, SacredLogger
from smg.music.melody_lib import melody_to_midi, melody_to_pianoroll
from smg.datasets.melody_dataset import MelodyDataset, MelodyEncode, MelodyDecode, FixedLengthMelodyDataset
from smg.models.music_vae.music_vae import MusicVAE
from smg.models.music_vae import encoder as smg_encoder
from smg.models.music_vae import decoder as smg_decoder
from smg.ingredients.data import data_ingredient, dataset_train_valid_split

try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False

MAX_N_RESULTS = 4

# create experiment
ex = Experiment('train_music_vae', ingredients=[data_ingredient])


def save_checkpoint(file_path, epoch, model, optimizer, use_apex=False):
    """Save current training state"""
    ckpt = {
        "epoch": epoch,
        "model": model.create_ckpt(),
        "optimizer": optimizer.state_dict()
    }
    if use_apex:
        ckpt["apex"] = amp.state_dict()
    torch.save(ckpt, file_path)


def load_checkpoint(ckpt_path, model, opt, device, use_apex=False):
    ckpt = torch.load(ckpt_path, map_location=device)
    model_ckpt = ckpt["model"]
    # load model state
    model.encoder.load_state_dict(model_ckpt["encoder"]["state"])
    model.decoder.load_state_dict(model_ckpt["decoder"]["state"])

    # load optimizer state
    opt.load_state_dict(ckpt["optimizer"])

    if use_apex:
        if "apex" not in ckpt:
            raise ValueError("No apex state found in checkpoint")
        amp.load_state_dict(ckpt["apex"])

    train_step = ckpt["train_step"] if "train_step" in ckpt else None
    return ckpt["epoch"], train_step


def calc_sampling_probabilty(step, rate, schedule):
    if schedule == "constant":
        return rate
    k = torch.tensor(rate, dtype=torch.float32)
    return 1.0 - k / (k + torch.exp(step / k))


def calc_beta(step, beta_rate, max_beta):
    """"Formula taken from magenta music vae implementation.
    (see magenta/models/music_vae/base_model.py class MusicVAE method _compute_model_loss)"""
    return (1.0 - beta_rate ** step) * max_beta


def calc_loss(x_hat, mu, sigma, x, alpha=1.0, beta=1.0, free_bits=0):
    variance = sigma.pow(2)
    log_variance = variance.log()

    # NOTE: this parameter depends on z_dim
    free_nats = free_bits * np.log(2.0)

    # Sum up kl div per time step, then substract the free nats and remove negative values, then calc mean
    # this should be consistent with magenta implementation
    kl_div = -0.5 * torch.sum(1 + log_variance - mu.pow(2) - variance, dim=1)
    kl_cost = torch.mean(torch.max(kl_div - free_nats, torch.zeros_like(kl_div)))

    n_features = x_hat.size(-1)
    x = x.argmax(dim=-1)
    r_loss = F.cross_entropy(x_hat.view(-1, n_features), x.view(-1), reduction="mean")

    # NOTE: to be consistent with the magenta implementation,
    #   alpha must be set to sequence length
    elbo_loss = alpha * r_loss + beta * kl_cost

    return elbo_loss, r_loss.item(), kl_cost.item(), torch.mean(kl_div.detach()).item()


def get_grad_norm(params):
    # alternative https://discuss.pytorch.org/t/check-the-norm-of-gradients/27961/2
    # total_norm = 0.0
    # for p in params:
    #     param_norm = p.grad.data.norm(2)
    #     total_norm += param_norm.item() ** 2
    # return total_norm ** (1. / 2)
    # see https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html
    norm_type = 2
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type) for p in params]), norm_type)
    return total_norm.item()


def train(epoch, global_step, model, data_loader, loss_fn, opt,
          lr_scheduler, device, print_log_interval, logger, beta_fn,
          sampling_prob_fn, accumulated_grad_steps, use_apex):
    model.train()
    start_time = time.time()

    n_batches = len(data_loader)
    n_batches = n_batches // accumulated_grad_steps
    data_gen = iter(data_loader)

    # metrics I don't want to log every update step
    advanced_logging_interval = n_batches // 4

    for batch_idx in range(n_batches):

        # set sampling prob
        sampling_prob = sampling_prob_fn(global_step)
        model.decoder.set_sampling_probability(sampling_prob)

        # calculate beta
        beta = beta_fn(global_step)

        opt.zero_grad()

        # keep track of some stats
        stats = {"loss": 0.0, "r_loss": 0.0, "kl_loss": 0.0, "kl_div": 0.0, "sample_ratio": 0.0}
        means = []
        sigmas = []
        z_arr = []

        # make 'accumulated_grad_steps' forward and backward steps
        # TODO: when used accumulated_grad_steps=16 loss and r_loss returned nan, maybe there
        #   is still a bug in the implementation
        for _ in range(accumulated_grad_steps):
            x = next(data_gen).to(device)

            x_hat, mu, sigma, sample_ratio, z = model.forward(x)

            loss, r_loss, kl_cost, kl_div = loss_fn(x_hat, mu, sigma, x, beta=beta)
            # normalize loss
            loss = loss / accumulated_grad_steps

            if use_apex:
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            stats["loss"] += loss.item()
            stats["r_loss"] += r_loss
            stats["kl_loss"] += kl_cost
            stats["kl_div"] += kl_div
            stats["sample_ratio"] += sample_ratio
            means.append(mu.detach())
            sigmas.append(sigma.detach())
            z_arr.append(z.detach())

        # set learning rate
        lr = lr_scheduler(global_step)
        opt.param_groups[0]['lr'] = lr
        opt.step()

        # before logging, normalize stats, note loss is already normalized
        stats["r_loss"] /= accumulated_grad_steps
        stats["kl_loss"] /= accumulated_grad_steps
        stats["kl_div"] /= accumulated_grad_steps
        stats["sample_ratio"] /= accumulated_grad_steps

        # log statistics
        batch_num = batch_idx + 1
        if batch_num % advanced_logging_interval == 0:
            # log histogram
            logger.add_histogram("train.encoder/mean", torch.cat(means), global_step)
            logger.add_histogram("train.encoder/sigma", torch.cat(sigmas), global_step)
            logger.add_histogram("train.encoder/z", torch.cat(z_arr), global_step)

            # log gradient norm
            logger.add_scalar("train.grad_norm/global", get_grad_norm(model.parameters()), global_step)
            logger.add_scalar("train.grad_norm/encoder", get_grad_norm(model.encoder.parameters()), global_step)
            logger.add_scalar("train.grad_norm/decoder", get_grad_norm(model.decoder.parameters()), global_step)

        logger.add_scalar("train.loss", stats["loss"], global_step)
        logger.add_scalar("train.losses/kl_loss", stats["kl_loss"], global_step)
        logger.add_scalar("train.losses/kl_div", stats["kl_div"], global_step)
        logger.add_scalar("train.losses/r_loss", stats["r_loss"], global_step)
        logger.add_scalar("sampling/train_ratio", stats["sample_ratio"], global_step)

        logger.add_scalar("train.losses/kl_beta", beta, global_step)
        logger.add_scalar("sampling/sampling_probability", sampling_prob, global_step)
        logger.add_scalar("lr", lr, global_step)

        if batch_num % print_log_interval == 0 or batch_num == n_batches:
            # epoch | batch_num/n_batches (finsihed% ) | loss | r_loss | kl_loss | sec
            print("{:3d} | {:4d}/{} ({:3.0f}%) | {:.6f} | {:.6f} | {:.6f} | {:.4f} sec.".format(
                epoch, batch_num, n_batches, 100. * batch_num / n_batches,
                stats["loss"], stats["r_loss"], stats["kl_loss"], time.time() - start_time))

        global_step += 1
    return global_step


def evaluate(epoch, model, data_loader, loss_fn, device, logger, best_loss, beta):
    model.eval()

    total_loss = 0.0
    total_r_loss = 0.0
    total_kl_loss = 0.0
    total_kl_div = 0.0
    total_sampled_ratio = 0.0
    means = []
    sigmas = []
    z_arr = []

    start_time = time.time()
    with torch.no_grad():
        for batch_idx, x in enumerate(data_loader):
            x = x.to(device)
            x_hat, mu, sigma, sampled_ratio, z = model.forward(x)
            loss, r_loss, kl_cost, kl_div = loss_fn(x_hat, mu, sigma, x, beta=beta)

            total_loss += loss.item()
            total_r_loss += r_loss
            total_kl_loss += kl_cost
            total_kl_div += kl_div
            total_sampled_ratio += sampled_ratio
            means.append(mu.detach())
            sigmas.append(sigma.detach())
            z_arr.append(z.detach())

    avg_loss = total_loss / len(data_loader)
    avg_r_loss = total_r_loss / len(data_loader)
    avg_kl_loss = total_kl_loss / len(data_loader)
    avg_kl_div = total_kl_div / len(data_loader)
    avg_sampled_ratio = total_sampled_ratio / len(data_loader)

    logger.add_histogram("eval.encoder/mean", torch.cat(means), epoch)
    logger.add_histogram("eval.encoder/sigma", torch.cat(sigmas), epoch)
    logger.add_histogram("eval.encoder/z", torch.cat(z_arr), epoch)

    logger.add_scalar("eval.loss", avg_loss, epoch)
    logger.add_scalar("eval.losses/kl_beta", beta, epoch)
    logger.add_scalar("eval.losses/kl_loss", avg_kl_loss, epoch)
    logger.add_scalar("eval.losses/kl_div", avg_kl_div, epoch)
    logger.add_scalar("eval.losses/r_loss", avg_r_loss, epoch)
    logger.add_scalar("sampling/eval_ratio", avg_sampled_ratio, epoch)

    new_best = avg_loss < best_loss
    if new_best:
        best_loss = avg_loss

    print('====> evaluation loss={:.6f}, r_loss={:.6f}, kl_loss={:.6f} ({:.4f} sec.){}'.format(
        avg_loss, avg_r_loss, avg_kl_loss, time.time() - start_time,
        " *new best*" if new_best else ""))

    return best_loss, new_best


@ex.capture
def get_run_dir(_run, run_dir):
    # file storage observer dir has higher priority than specified run_dir
    for obs in _run.observers:
        if isinstance(obs, FileStorageObserver):
            return obs.dir
    # TODO make subdir for current run
    return run_dir


@ex.capture
def run(_run,
        batch_size, accumulated_grad_steps,
        num_epochs, num_workers,
        initial_lr, lr_decay_rate, min_lr,
        melody_dir, melody_length, n_classes,
        sampling_schedule, sampling_rate,
        log_interval,
        encoder_params, use_hier, decoder_params,
        beta_rate, max_beta, free_bits, alpha=1.0,
        use_apex=False, opt_level="O1", ckpt_path=None):

    if use_apex and not APEX_AVAILABLE:
        use_apex = False
        print("Mixed precision training is not available, install apex.")

    run_dir = get_run_dir()
    ckpt_dir = Path(run_dir) / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger = CompositeLogger()
    log_dir = run_dir
    logger.add(TensorBoardLogger(log_dir=log_dir))
    logger.add(SacredLogger(ex, _run))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = smg_encoder.BidirectionalLstmEncoder(**encoder_params)

    if use_hier:
        decoder_params = dict(decoder_params)
        output_decoder_params = decoder_params.pop("output_decoder_params")
        output_decoder = smg_decoder.LstmDecoder(**output_decoder_params)
        decoder = smg_decoder.HierarchicalDecoder(output_decoder=output_decoder, **decoder_params)
    else:
        decoder = smg_decoder.LstmDecoder(**decoder_params)

    model = MusicVAE(encoder=encoder, decoder=decoder)
    model = model.to(device)

    # set betas, even this are the default values, to make it explicit that we use
    # the same values as in the music vae implementation.
    opt = optim.Adam(model.parameters(), lr=initial_lr, betas=(0.9, 0.999))
    lr_scheduler = lambda step: (initial_lr - min_lr) * lr_decay_rate ** step + min_lr

    if use_apex:
        keep_batchnorm_fp32 = True if opt_level != "O1" else None
        model, opt = amp.initialize(
            model, opt, opt_level=opt_level,
            keep_batchnorm_fp32=keep_batchnorm_fp32, loss_scale="dynamic"
        )

    # dataset = MelodyDataset(melody_dir=melody_dir, melody_length=melody_length,
    #                        transforms=MelodyEncode(n_classes=n_classes))
    dataset = FixedLengthMelodyDataset(melody_dir=melody_dir, transforms=MelodyEncode(n_classes=n_classes,
                                                                                      num_special_events=0))
    ds_train, ds_eval = dataset_train_valid_split(dataset, valid_split=0.2)

    dl_train = DataLoader(ds_train, batch_size=batch_size, num_workers=num_workers, drop_last=True, shuffle=True)
    dl_eval = DataLoader(ds_eval, batch_size=batch_size, num_workers=num_workers, drop_last=True, shuffle=True)

    # calc beta based on train step
    beta_fn = functools.partial(calc_beta, beta_rate=beta_rate, max_beta=max_beta)
    loss_fn = functools.partial(calc_loss, free_bits=free_bits, alpha=alpha)
    sampling_prob_fn = functools.partial(calc_sampling_probabilty, rate=sampling_rate, schedule=sampling_schedule)

    start_epoch = 0
    train_step = 0
    if ckpt_path is not None:
        start_epoch, train_step = load_checkpoint(ckpt_path, model, opt, device, use_apex)
        if train_step is None:
            # if train step was not saved, use start of epoch
            train_step = int(start_epoch * (len(dl_train) // accumulated_grad_steps))

    best_loss = float('inf')
    try:
        print("Start training from epoch {} with step {}".format(start_epoch, train_step))
        for epoch in range(start_epoch + 1, num_epochs + 1):
            train_step = train(epoch, train_step, model, dl_train, loss_fn, opt, lr_scheduler, device, log_interval,
                               logger, beta_fn, sampling_prob_fn, accumulated_grad_steps, use_apex)
            best_loss, new_best = evaluate(epoch, model, dl_eval, loss_fn, device, logger, best_loss, beta=max_beta)
            if new_best:
                save_checkpoint(str(ckpt_dir / "model_ckpt_best.pth"), epoch, model, opt)
            else:
                # save the current model anyway
                save_checkpoint(str(ckpt_dir / "model_ckpt_current.pth"), epoch, model, opt)
    except KeyboardInterrupt:
        print("Keyboard interrupt on epoch {}".format(epoch))
        save_checkpoint(str(ckpt_dir / "model_ckpt_interrupt.pth"), epoch, model, opt)


@ex.config
def config():
    batch_size = 16
    num_workers = 0


@ex.automain
def main():
    run()
