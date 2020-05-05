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

MAX_N_RESULTS = 4

# create experiment
ex = Experiment('train_music_vae', ingredients=[data_ingredient])


def save_checkpoint(file_path, epoch, model, optimizer):
    """Save current training state"""
    ckpt = {
        'epoch': epoch,
        'model': model.create_ckpt(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(ckpt, file_path)


def interpolate(model, start_melody, end_melody, num_steps, device):

    def _slerp(p0, p1, t):
        """Spherical linear interpolation."""
        omega = np.arccos(np.dot(np.squeeze(p0/np.linalg.norm(p0)),
                                 np.squeeze(p1/np.linalg.norm(p1))))
        so = np.sin(omega)
        return np.sin((1.0-t)*omega) / so * p0 + np.sin(t*omega)/so * p1

    endpoints = torch.stack((start_melody, end_melody)).to(device)
    with torch.no_grad():
        mu, sigma = model.encode(endpoints)
        z = model.reparameterize(mu, sigma)  # maybe directly use mu
        z = z.cpu()
        z = torch.stack([_slerp(z[0], z[1], t)
                         for t in np.linspace(0, 1, num_steps)]).to(device)

    return model.decode(z, endpoints.size(1))


def save_interpolation(samples, file_path_without_suffix):
    melody_decode = MelodyDecode()

    _, samples = torch.max(samples.cpu(), dim=-1)

    samples = melody_decode(samples).cpu().numpy().astype(np.int32)
    pianorolls = []
    for sample in samples:
        pianorolls.append(melody_to_pianoroll(sample[:32]))

    merged_pianoroll = np.concatenate(pianorolls, axis=0)

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    pp.plot_pianoroll(ax, merged_pianoroll)
    ax.set_xticks(np.arange(32, merged_pianoroll.shape[0], 32))

    fig.savefig(file_path_without_suffix + ".png")
    plt.close()


def save_reconstruction(origin, recon, file_path_without_suffix):
    pm = melody_to_midi(origin)
    orig_filepath = file_path_without_suffix + "_orig.mid"
    pm.write(orig_filepath)

    pm = melody_to_midi(recon.astype(np.int8))
    recon_filepath = file_path_without_suffix + "_recon.mid"
    pm.write(recon_filepath)

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(211)
    pp.plot_pianoroll(ax, melody_to_pianoroll(origin))
    plt.title('original')

    ax = fig.add_subplot(212)
    pp.plot_pianoroll(ax, melody_to_pianoroll(recon))
    plt.title('reconstruction')

    fig.savefig(file_path_without_suffix + "_recon.png")
    plt.close()


def save_melody(melody, file_path_without_suffix):
    pm = melody_to_midi(melody)
    midi_filepath = file_path_without_suffix + ".mid"
    pm.write(midi_filepath)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    pp.plot_pianoroll(ax, melody_to_pianoroll(melody))
    fig.savefig(file_path_without_suffix + ".png")
    plt.close()


def calc_sampling_probabilty(step, rate, schedule):
    if schedule == "constant":
        return rate
    k = torch.tensor(rate, dtype=torch.float32)
    return 1.0 - k / (k + torch.exp(step / k))


def calc_beta(step, beta_rate, max_beta):
    """"Formula taken from magenta music vae implementation.
    (see magenta/models/music_vae/base_model.py class MusicVAE method _compute_model_loss)"""
    return (1.0 - beta_rate**step) * max_beta


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

    return elbo_loss, r_loss, kl_cost, torch.mean(kl_div)


def train(epoch, global_step, model, data_loader, loss_fn, opt, lr_scheduler, device, log_interval, logger, beta_fn, sampling_prob_fn):
    model.train()
    train_loss = 0.0
    start_time = time.time()
    for batch_idx, x in enumerate(data_loader):
        x = x.to(device)

        # set sampling prob
        sampling_prob = sampling_prob_fn(global_step)
        model.decoder.sampling_probability = sampling_prob

        opt.zero_grad()
        x_hat, mu, sigma, sampled_ratio = model.forward(x)

        beta = beta_fn(global_step)
        loss, r_loss, kl_cost, kl_div = loss_fn(x_hat, mu, sigma, x, beta=beta)
        loss.backward()

        # set learning rate
        lr = lr_scheduler(global_step)
        opt.param_groups[0]['lr'] = lr
        opt.step()

        train_loss += loss.item()
        logger.add_scalar("train.loss", loss.item(), global_step)
        logger.add_scalar("train.losses/kl_beta", beta, global_step)
        logger.add_scalar("train.losses/kl_bits", kl_div / np.log(2.0), global_step)
        logger.add_scalar("train.losses/kl_loss", kl_cost.item(), global_step)
        logger.add_scalar("train.losses/r_loss", r_loss.item(), global_step)
        logger.add_scalar("sampling/sampling_probability", sampling_prob, global_step)
        logger.add_scalar("sampling/train_ratio", sampled_ratio, global_step)
        logger.add_scalar("lr", lr, global_step)

        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t{:.4f} sec.'.format(
                epoch, (batch_idx + 1) * len(x), len(data_loader.dataset),
                100. * (batch_idx + 1) / len(data_loader),
                loss.item() / len(x),
                time.time() - start_time))
        global_step += 1

    print('====> Epoch: {} Average loss: {:.6f} ({:.4f} sec.)'.format(epoch, train_loss / len(data_loader),
                                                                      time.time() - start_time))
    return global_step


def evaluate(epoch, global_step, model, data_loader, loss_fn, device, reconstruct_dir,
             interpolate_dir, melody_decode, logger, beta):
    model.eval()
    eval_loss = 0.0
    start_time = time.time()
    with torch.no_grad():
        for batch_idx, x in enumerate(data_loader):
            x = x.to(device)
            x_hat, mu, sigma, sampled_ratio = model.forward(x)
            loss, r_loss, kl_cost, kl_div = loss_fn(x_hat, mu, sigma, x, beta=beta)

            eval_loss += loss.item()
            logger.add_scalar("eval.loss", loss.item(), global_step)
            logger.add_scalar("eval.losses/kl_beta", beta, global_step)
            logger.add_scalar("eval.losses/kl_bits", kl_div / np.log(2.0), global_step)
            logger.add_scalar("eval.losses/kl_loss", kl_cost.item(), global_step)
            logger.add_scalar("eval.losses/r_loss", r_loss.item(), global_step)
            logger.add_scalar("sampling/eval_ratio", sampled_ratio, global_step)

            # if batch_idx == 0:
            #     recon_epoch_dir = reconstruct_dir / str(epoch)
            #     recon_epoch_dir.mkdir()
            #     # reconstruction
            #     _, recon = torch.max(x_hat.cpu(), dim=-1)
            #     for i, (orig_melody, recon_melody) in enumerate(zip(x[:MAX_N_RESULTS],
            #                                                         recon[:MAX_N_RESULTS])):
            #         orig_melody = orig_melody.cpu().squeeze(dim=-1).numpy().astype(np.int8)
            #         recon_melody = recon_melody.cpu().numpy().astype(np.int8)
            #
            #         save_reconstruction(melody_decode(orig_melody),
            #                             melody_decode(recon_melody),
            #                             str(recon_epoch_dir / "{}".format(i)))
            #
            #     # interpolate
            #     if x.size(0) > 1:
            #         start_melody, end_melody = x[0], x[1]
            #         samples = interpolate(model, start_melody, end_melody, 7, device)
            #         save_interpolation(samples, str(interpolate_dir / "epoch_{}".format(epoch)))

            global_step += 1

    print('====> Evaluation set loss: {:.6f}  ({:.4f} sec.)'.format(eval_loss / len(data_loader),
                                                                    time.time() - start_time))
    return global_step, eval_loss / len(data_loader)


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
        batch_size, num_epochs, num_workers,
        initial_lr, lr_decay_rate, min_lr,
        melody_dir, melody_length, n_classes,
        sampling_schedule, sampling_rate,
        log_interval,
        z_dim, encoder_params, decoder_params,
        beta_rate, max_beta, free_bits, alpha=1.0):

    run_dir = get_run_dir()
    sample_dir = Path(run_dir) / "results/samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    reconstruct_dir = Path(run_dir) / "results/reconstruction"
    reconstruct_dir.mkdir(parents=True, exist_ok=True)
    interpolate_dir = Path(run_dir) / "results/interpolation"
    interpolate_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(run_dir) / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger = CompositeLogger()
    log_dir = run_dir + "/tensorboard"
    logger.add(TensorBoardLogger(log_dir=log_dir))
    logger.add(SacredLogger(ex, _run))

    melody_decode = MelodyDecode()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = smg_encoder.BidirectionalLstmEncoder(**encoder_params)
    decoder = smg_decoder.LstmDecoder(**decoder_params)
    model = MusicVAE(encoder=encoder, decoder=decoder)
    model = model.to(device)

    # set betas, even this are the default values, to make it explicit that we use
    # the same values as in the music vae implementation.
    opt = optim.Adam(model.parameters(), lr=initial_lr, betas=(0.9, 0.999))
    lr_scheduler =lambda step: (initial_lr - min_lr) * lr_decay_rate**step + min_lr

    #dataset = MelodyDataset(melody_dir=melody_dir, melody_length=melody_length,
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

    train_step = eval_step = 0
    best_loss = float('inf')
    for epoch in range(1, num_epochs+1):
        train_step = train(epoch, train_step, model, dl_train, loss_fn, opt, lr_scheduler, device, log_interval,
                           logger, beta_fn, sampling_prob_fn)
        eval_step, eval_loss = evaluate(epoch, eval_step, model, dl_eval, loss_fn, device,
                                        reconstruct_dir,
                                        interpolate_dir,
                                        melody_decode,
                                        logger, beta=max_beta)
        if eval_loss < best_loss:
            print("====> Got a new best model! From {:.6f} to {:.6f}".format(best_loss, eval_loss))
            best_loss = eval_loss
            save_checkpoint(str(ckpt_dir / "model_ckpt_best.pth"), epoch, model, opt)


        # model.eval()
        # with torch.no_grad():
        #     sample_epoch_dir = sample_dir / str(epoch)
        #     sample_epoch_dir.mkdir()
        #
        #     # sample from z space
        #     z = torch.randn(MAX_N_RESULTS, z_dim).to(device)
        #     sample = model.decode(z, melody_length).cpu()
        #     _, sample_argmax = torch.max(sample, dim=-1)
        #     for i, melody in enumerate(sample_argmax):
        #         melody = melody.cpu().numpy().astype(np.int8)
        #         save_melody(melody_decode(melody),
        #                     str(sample_epoch_dir / "sample_{}".format(i)))


@ex.config
def config():
    batch_size = 16
    num_workers = 0


@ex.automain
def main():
    run()
