from pathlib import Path
import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Subset
from smg.datasets.lpd_5_cleansed import LPD5Cleansed
from smg.models.recurrent import RecurrentSMG


pbar_update_interval = 100

ex = Experiment('train_lstm')

ex.captured_out_filter = lambda captured_output: "Output capturing turned off."

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
def get_dataset(data_dir, lowest_pitch, n_pitches, beat_resolution, in_seq_length, out_seq_length, step_size, instruments):
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
def get_model(hidden_size, num_layers, out_seq_length, instruments, n_pitches):
    instruments = list(instruments)

    kwargs = {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
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


@ex.capture
def run(model, dl_train, dl_valid, dev, _run, num_epochs=10, lr=1e-2):
    ckpt_dir = get_checkpoint_dir()

    opt = optim.Adam(model.parameters(), lr=lr)
    calc_loss = nn.MSELoss()

    ckpt_interval = num_epochs // 10 if num_epochs > 10 else 5
    best_total_loss = float('inf')

    for epoch in range(1, num_epochs+1):
        # train model for one epoch
        model.train()
        total_loss_train = 0.0
       
        with tqdm(desc="{:4d}. epoch".format(epoch), total=len(dl_train)) as pbar:
            for step, (x, y) in enumerate(dl_train):
                x, y = x.to(dev), y.to(dev)
                
                opt.zero_grad()
                y_hat = model.forward(x)
                loss = calc_loss(y_hat, y)
                loss.backward()
                opt.step()
                
                _run.log_scalar("train.loss", loss.item())
                total_loss_train += loss.item()
                
                
                if (step + 1) % pbar_update_interval == 0:
                    pbar.set_postfix(loss=loss.item())
                    pbar.update(pbar_update_interval)
            
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

                _run.log_scalar("valid.loss", loss.item())
                total_loss_valid += loss.item()
    
        # save every now and then
        if epoch % ckpt_interval == 0:
            model.save_ckpt(file_name=ckpt_dir / "model_ckpt_{}.pth".format(epoch))

        # save current best model (on validation dataset)
        new_best = False
        if total_loss_valid < best_total_loss:
            best_total_loss = total_loss_valid
            new_best = True
            model.save_ckpt(file_name=ckpt_dir / "model_ckpt_best.pth")

        print("\n{:4d}. epoch loss (train/valid): {:.5f} / {:.5f}{}".format(epoch, total_loss_train,
            total_loss_valid,
            " *NEW BEST MODEL*" if new_best else ""))
    
    model.save_ckpt(file_name=ckpt_dir / "model_ckpt_finished.pth")


@ex.config
def config():
    # data_loader config
    batch_size = 8
    n_workers = 0

    # dataset config
    data_dir = "../data/examples" # #"../../../data/lpd_5"
    valid_split = 0.3

    # model configs
    hidden_size = 200
    num_layers = 3

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
    

@ex.automain
def main():
    dataset = get_dataset()
    ds_train, ds_valid = dataset_train_valid_split(dataset)
    dl_train = get_data_loader(ds_train, shuffle=True)
    dl_valid = get_data_loader(ds_valid, shuffle=False)

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_model()
    model = model.to(dev)

    run(model, dl_train, dl_valid, dev)