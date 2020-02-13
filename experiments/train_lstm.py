from pathlib import Path
import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Subset
from datetime import datetime

from smg.datasets.lpd_5_cleansed import LPD5Cleansed
from smg.models.recurrent import RecurrentSMG
from smg.common.loggers import CompositeLogger, TensorBoardLogger, SacredLogger
from smg.common.generate_music import generate_pianoroll


# create experiment
ex = Experiment('train_lstm')
ex.captured_out_filter = lambda captured_output: "Output capturing turned off."

# global variables
pbar_update_interval = 100

@ex.capture
def get_data_loader(dataset, batch_size, n_workers=4, shuffle=True):
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=n_workers,
        batch_size=batch_size,
        drop_last=True,
        shuffle=shuffle
    )
    return data_loader


@ex.capture
def get_datasets(data_obj_file, valid_step_size):
    if data_obj_file is None:
        dataset = get_dataset_from_dir()
        return dataset_train_valid_split(dataset)
    else:
        import pickle

        with open(str(data_obj_file), 'rb') as f:
            data_obj = pickle.load(f)
        
        ds_train = get_dataset_from_obj(data_obj, 'train')
        ds_valid = get_dataset_from_obj(data_obj, 'valid', step_size=valid_step_size)

        return ds_train, ds_valid


@ex.capture
def get_dataset_from_obj(data_obj, phase, lowest_pitch, n_pitches, beat_resolution, in_seq_length, out_seq_length, step_size, instruments):
    instruments = list(instruments)

    assert(data_obj['lowest_pitch'] == lowest_pitch)
    assert(data_obj['n_pitches'] == n_pitches)
    assert(data_obj['beat_resolution'] == beat_resolution)
    assert(data_obj['instruments'] == instruments)
    
    phase_obj = dict(data_obj, **{'samples': data_obj['samples'][phase], 'names': data_obj['names'][phase]})
    kwargs = {
        "data_obj": phase_obj,
        "in_seq_length": in_seq_length,
        "out_seq_length": out_seq_length,
        "step_size": step_size,
    }
    return LPD5Cleansed(**kwargs)


@ex.capture
def get_dataset_from_dir(data_dir, lowest_pitch, n_pitches, beat_resolution, in_seq_length, out_seq_length, step_size, instruments):
    instruments = list(instruments)
    kwargs = {
        "data_dir": data_dir, 
        "lowest_pitch": lowest_pitch,
        "n_pitches": n_pitches,
        "beat_resolution": beat_resolution,
        "in_seq_length": in_seq_length,
        "out_seq_length": out_seq_length,
        "step_size": step_size,
        "instruments": instruments
    }
    return LPD5Cleansed(**kwargs)


@ex.capture
def dataset_train_valid_split(dataset, valid_split):
    ds_size = len(dataset)
    indices = np.arange(ds_size)
    train_size = int(ds_size * (1 - valid_split))

    ds_train = Subset(dataset, indices[:train_size])
    ds_valid = Subset(dataset, indices[train_size:])

    assert(len(ds_train) + len(ds_valid) == ds_size)

    return ds_train, ds_valid


@ex.capture
def get_model(hidden_size, num_layers, out_seq_length, dense_layer_hiddens, instruments, n_pitches):
    instruments = list(instruments)
    dense_layer_hiddens = list(dense_layer_hiddens)

    kwargs = {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dense_layer_hiddens": dense_layer_hiddens,
        "out_seq_length": out_seq_length, # part of out features
        "instruments": instruments, # part of out features
        "n_pitches": n_pitches, # part of out features
    }
    return RecurrentSMG(**kwargs)


@ex.capture
def get_checkpoint_dir(_run):
    ckpt_dir = ""
    if len(_run.observers) > 0 and isinstance(_run.observers[0], FileStorageObserver):
        ckpt_dir = Path(_run.observers[0].dir + "/ckpt")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


def save_checkpoint(file_name, epoch, model, optimizer):
    """Save current training state"""

    ckpt = {
        'epoch': epoch,
        'model': model.create_ckpt(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(ckpt, file_name)


@ex.capture
def load_checkpoint(model, optimizer, _run, checkpoint, device=None):
    # add checkpoint file to resources and open it
    with _run.open_resource(checkpoint, mode='rb') as f:
        ckpt = torch.load(f, map_location=device)

        # assert model params are equal and load state dict
        assert(model.kwargs == ckpt['model']['kwargs'])
        model.load_state_dict(ckpt['model']['state'])

        # load state dict of optimizer
        optimizer.load_state_dict(ckpt['optimizer'])

    return ckpt['epoch']

    
@ex.capture
def run(model, dl_train, dl_valid, dev, _run, logger, checkpoint=None, num_epochs=10, lr=1e-2):
    ckpt_dir = get_checkpoint_dir()

    optimizer = optim.Adam(model.parameters(), lr=lr)
        
    epoch = 0
    if checkpoint is not None:
        epoch = load_checkpoint(model, optimizer, device=dev)
        print("Load checkpoint from '{}' trained for {} epoch(s).".format(checkpoint, epoch))

    calc_loss = nn.MSELoss()

    audio_interval = 2 # num_epochs // 10 if num_epochs > 10 else 5
    best_total_loss = float('inf')

    # save dataload size to later visualize epochs
    logger.add_scalar("train.size", len(dl_train), 0)
    logger.add_scalar("valid.size", len(dl_valid), 0)

    train_step = 0
    valid_step = 0
    for epoch in range(epoch + 1, num_epochs+1):
        # train model for one epoch
        model.train()
        total_loss_train = 0.0
       
        with tqdm(desc="{:4d}. epoch".format(epoch), total=len(dl_train)) as pbar:
            for step, (x, y) in enumerate(dl_train):
                x, y = x.to(dev), y.to(dev)
                
                optimizer.zero_grad()
                y_hat = model.forward(x)
                loss = calc_loss(y_hat, y)
                loss.backward()
                optimizer.step()
                
                #_run.log_scalar("train.loss", loss.item())
                logger.add_scalar("train.loss", loss.item(), train_step)
                total_loss_train += loss.item()
                
                if (step + 1) % pbar_update_interval == 0:
                    pbar.set_postfix(loss=loss.item())
                    pbar.update(pbar_update_interval)
                
                train_step += 1
            
            # update the rest
            pbar.update((step + 1) % pbar_update_interval)
            

        # validate model
        model.eval()
        total_loss_valid = 0.0
        with torch.no_grad():
            for x, y in dl_valid:
                x, y = x.to(dev), y.to(dev)
                y_hat = model.forward(x)
                loss = calc_loss(y_hat, y)

                # _run.log_scalar("valid.loss", loss.item())
                logger.add_scalar("valid.loss", loss.item(), valid_step)
                total_loss_valid += loss.item()

                valid_step += 1

        # create a checkpoint of current train state
        save_checkpoint(str(ckpt_dir / "model_ckpt.pth"), epoch, model, optimizer)

        # save current best model (on validation dataset)
        new_best = False
        if total_loss_valid < best_total_loss:
            best_total_loss = total_loss_valid
            new_best = True
            save_checkpoint(str(ckpt_dir / "model_ckpt_best.pth"), epoch, model, optimizer)

        print("\n---- {:4d}. epoch loss (train/valid): {:.5f} / {:.5f}{} ----".format(epoch, total_loss_train,
            total_loss_valid,
            " *NEW BEST MODEL*" if new_best else ""))

        # every now and then create a sample audio file
        if epoch % audio_interval == 0:
            for phase, data_loader in zip(['train', 'valid'], [dl_train, dl_valid]):
                x, _ = data_loader.dataset[np.random.choice(len(data_loader.dataset))]
                x = torch.tensor(x).unsqueeze(0).to(dev)
                # y = torch.tensor(y).unsqueeze(0).to(dev)

                pianoroll = generate_pianoroll(model, x, x.size(1) * 2)
                logger.add_pianoroll_img("{}.sample".format(phase), pianoroll, epoch)
                # do not log audio for now as this cost quite some time and the result
                # is not good right know
                #logger.add_pianoroll_audio("{}.sample".format(phase), pianoroll, epoch)

    
    save_checkpoint(str(ckpt_dir / "model_ckpt_finished.pth"), epoch, model, optimizer)
    logger.close()


@ex.config
def config():
    # data_loader config
    batch_size = 8
    n_workers = 0

    # dataset config
    data_obj_file = None # "../data/lpd_5_subsets/4_24_72_piano/lpd_5_subset_10"
    data_dir = "../data/examples" # #"../../../data/lpd_5"
    valid_split = 0.3

    # model configs
    hidden_size = 200
    num_layers = 3
    dense_layer_hiddens = []

    # train configs
    num_epochs = 10
    lr = 1e-3

    # general configs
    
    # NOTE: Sacred converts a list to a custom object which could
    # have negative side effect, for example when trying to serialize the list,
    # as is the case in the RecurrentSmg save_ckpt function, which resulted in a
    # error when loading a model from a saved checkpoint file.
    # So just to be save I try to convert the parameter back to a list when used in a captured
    # function.
    # This sucks and is definitely a drawback of the sacred magic, as I was not
    # expecting a list to be suddenly a differnt type. 

    # instruments = ['drums', 'piano', 'guitar', 'bass', 'strings']
    instruments = ['piano'] 
    lowest_pitch = 24
    n_pitches = 72
    beat_resolution = 4

    measures_per_sample = 4
    beats_per_measure = 4

    in_seq_length = beat_resolution * beats_per_measure * measures_per_sample
    step_size = beat_resolution * beats_per_measure
    out_seq_length = 1
    # step size for validation dataset
    valid_step_size = beat_resolution * beats_per_measure
    # step size for validation dataset
    valid_step_size = beat_resolution * beats_per_measure

    # path to checkpoint file to continue training
    checkpoint = None


@ex.automain
def main(_run, instruments, lowest_pitch, n_pitches, beat_resolution):
    instruments = list(instruments)

    ds_train, ds_valid = get_datasets()

    dl_train = get_data_loader(ds_train, shuffle=True)
    dl_valid = get_data_loader(ds_valid, shuffle=False)

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_model()
    log_dir = "../runs/tensorboard_log/" + str(datetime.timestamp(datetime.now()))
    if len(_run.observers) > 0 and isinstance(_run.observers[0], FileStorageObserver):
        log_dir = Path(_run.observers[0].dir + "/tensorboard_log")

    logger = CompositeLogger()
    logger.add(TensorBoardLogger(log_dir=log_dir, track_info={
        "instruments": instruments,
        "lowest_pitch": lowest_pitch, 
        "n_pitches": n_pitches, 
        "beat_resolution": beat_resolution
    }))
    logger.add(SacredLogger(ex, _run))

    model = model.to(dev)

    run(model, dl_train, dl_valid, dev, logger=logger)