import numpy as np
from pathlib import Path
import time
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import transforms

from sacred import Experiment
from sacred.observers import FileStorageObserver

import pypianoroll as pp

from smg.common.loggers import CompositeLogger, TensorBoardLogger, SacredLogger
from smg.music.melody_lib import melody_to_midi, melody_to_pianoroll
from smg.datasets.melody_dataset import MelodyDataset, MelodyEncode, MelodyDecode
from smg.models.music_vae.music_vae import MusicVAE
from smg.models.music_vae import encoder as smg_encoder
from smg.models.music_vae import decoder as smg_decoder
from smg.ingredients.data import data_ingredient, dataset_train_valid_split

# create experiment
ex = Experiment('train_music_vae', ingredients=[data_ingredient])


def interpolate(model, start_melody, end_melody, num_steps, device):

    def _slerp(p0, p1, t):
        """Spherical linear interpolation."""
        omega = np.arccos(np.dot(np.squeeze(p0/np.linalg.norm(p0)),
                                 np.squeeze(p1/np.linalg.norm(p1))))
        so = np.sin(omega)
        return np.sin((1.0-t)*omega) / so * p0 + np.sin(t*omega)/so * p1

    endpoints = torch.stack((start_melody, end_melody)).to(device)
    with torch.no_grad():
        mu, log_var = model.encode(endpoints)
        z = model.reparameterize(mu, log_var)  # maybe directly use mu
        z = torch.stack([_slerp(z[0], z[1], t)
                         for t in np.linspace(0, 1, num_steps)]).to(device)

    return model.decode(z, endpoints.size(1))


def save_interpolation(samples, file_path_without_suffix):
    melody_decode = MelodyDecode()

    _, samples = torch.max(samples.cpu(), dim=-1)

    samples = melody_decode(samples).numpy().astype(np.int32)
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


def calc_loss(x_hat, mu, logvar, x, beta=1.0, free_bits=0):

    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    free_nats = free_bits * np.log(2.0)
    # TODO fix this
    kl_cost = kl_div
    # kl_cost = torch.max(kl_div - free_nats, torch.tensor(0, dtype=kl_div.dtype))

    n_steps = x_hat.size()[1]
    r_loss = None
    for t in range(n_steps):
        if r_loss is None:
            r_loss = F.cross_entropy(x_hat[:, t], x[:, t].squeeze(dim=-1).type(torch.LongTensor), reduction='sum')
        else:
            r_loss += F.cross_entropy(x_hat[:, t], x[:, t].squeeze(dim=-1).type(torch.LongTensor), reduction='sum')

    elbo_loss = r_loss + beta * kl_cost

    return elbo_loss, r_loss, kl_cost, kl_div


def train(epoch, global_step, model, data_loader, loss_fn, opt, device, log_interval, logger):
    model.train()
    train_loss = 0.0
    start_time = time.time()
    for batch_idx, x in enumerate(data_loader):
        x = x.to(device)

        opt.zero_grad()
        x_hat, mu, logvar = model.forward(x)
        loss, _, _, _ = loss_fn(x_hat, mu, logvar, x)
        loss.backward()
        opt.step()

        # TODO log all the losses
        logger.add_scalar("train.loss", loss.item(), global_step)
        train_loss += loss.item()

        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t{:.4f} sec.'.format(
                epoch, (batch_idx + 1) * len(x), len(data_loader.dataset),
                100. * (batch_idx + 1) / len(data_loader),
                loss.item() / len(x),
                time.time() - start_time))
        global_step += 1

    print('====> Epoch: {} Average loss: {:.4f} ({:.4f} sec.)'.format(epoch, train_loss / len(data_loader.dataset),
                                                                      time.time() - start_time))
    return global_step


def evaluate(epoch, global_step, model, data_loader, loss_fn, device, reconstruct_dir,
             interpolate_dir, melody_decode, logger):
    model.eval()
    eval_loss = 0.0
    start_time = time.time()
    with torch.no_grad():
        for batch_idx, x in enumerate(data_loader):
            x = x.to(device)
            x_hat, mu, logvar = model.forward(x)
            loss, _, _, _ = loss_fn(x_hat, mu, logvar, x)

            # TODO log all the losses
            eval_loss += loss.item()
            logger.add_scalar("eval.loss", loss.item(), global_step)

            if batch_idx == 0:
                # reconstruction
                _, recon = torch.max(x_hat.cpu(), dim=-1)
                for i, (orig_melody, recon_melody) in enumerate(zip(x, recon)):
                    save_reconstruction(melody_decode(orig_melody.squeeze(dim=-1).numpy().astype(np.int8)),
                                        melody_decode(recon_melody.numpy().astype(np.int8)),
                                        str(reconstruct_dir / "{}".format(i)))

                # interpolate
                if x.size(0) > 1:
                    start_melody, end_melody = x[0], x[1]
                    samples = interpolate(model, start_melody, end_melody, 7, device)
                    save_interpolation(samples, str(interpolate_dir / "epoch_{}".format(epoch)))

            global_step += 1

    print('====> Evaluation set loss: {:.4f}  ({:.4f} sec.)'.format(eval_loss / len(data_loader.dataset),
                                                                    time.time() - start_time))
    return global_step


@ex.capture
def get_run_dir(_run, run_dir):
    # file storage observer dir has higher priority than specified run_dir
    for obs in _run.observers:
        if isinstance(obs, FileStorageObserver):
            return obs.dir
    # TODO make subdir for current run
    return run_dir


@ex.capture
def run(_run, batch_size, num_epochs, num_workers, learning_rate, z_size, melody_dir, melody_length, log_interval):
    run_dir = get_run_dir()
    sample_dir = Path(run_dir) / "results/samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    reconstruct_dir = Path(run_dir) / "results/reconstruction"
    reconstruct_dir.mkdir(parents=True, exist_ok=True)

    interpolate_dir = Path(run_dir) / "results/interpolation"
    interpolate_dir.mkdir(parents=True, exist_ok=True)

    logger = CompositeLogger()
    log_dir = run_dir + "/tensorboard"
    logger.add(TensorBoardLogger(log_dir=log_dir))
    logger.add(SacredLogger(ex, _run))

    melody_decode = MelodyDecode()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    params = {
        "input_size": 1,
        "hidden_size": z_size,
        "z_size": z_size,
        "num_layers": 1
    }
    encoder = smg_encoder.BidirectionalLstmEncoder(**params)
    decoder = smg_decoder.LstmDecoder(130, params["z_size"], teacher_forcing=False)
    model = MusicVAE(encoder=encoder, decoder=decoder)

    model = model.to(device)

    opt = optim.Adam(model.parameters(), lr=learning_rate)

    transform = transforms.Compose([MelodyEncode(),
                                    lambda x: x[:, np.newaxis]])
    dataset = MelodyDataset(melody_dir=melody_dir, melody_length=melody_length, transforms=transform)
    ds_train, ds_eval = dataset_train_valid_split(dataset, valid_split=0.2)

    dl_train = DataLoader(ds_train, batch_size=batch_size, num_workers=num_workers, drop_last=True, shuffle=True)
    dl_eval = DataLoader(ds_eval, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    train_step = eval_step = 0
    for epoch in range(1, num_epochs+1):
        train_step = train(epoch, train_step, model, dl_train, calc_loss, opt, device, log_interval, logger)
        eval_step = evaluate(epoch, eval_step, model, dl_eval, calc_loss, device,
                            reconstruct_dir,
                            interpolate_dir,
                            melody_decode,
                            logger)

        # TODO log statistics
        with torch.no_grad():
            sample_epoch_dir = sample_dir / str(epoch)
            sample_epoch_dir.mkdir()

            # sample from z space
            z = torch.randn(16, z_size).to(device)
            sample = model.decode(z, melody_length).cpu()
            _, sample_argmax = torch.max(sample, dim=-1)
            for i, melody in enumerate(sample_argmax):
                save_melody(melody_decode(melody.numpy().astype(np.int8)),
                            str(sample_epoch_dir / "sample_{}".format(i)))


@ex.config
def config():
    # run_dir = "../runs/train_music_vae"
    # log_interval = 100

    batch_size = 16
    num_workers = 0

    # num_epochs = 2
    # learning_rate = 1e-3
    # z_size = 512
    # melody_dir = "../data/melodies/"
    # melody_length = 256


@ex.automain
def main():
    run()
