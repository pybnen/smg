import numpy as np
from sacred import Experiment
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Subset
from smg.datasets.lpd_5_cleansed import LPD5Cleansed
from smg.models.recurrent import RecurrentSMG

ex = Experiment('train_lstm')

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
def run(model, dl_train, dl_valid, dev, num_epochs=10, lr=1e-2):
    opt = optim.Adam(model.parameters(), lr=lr)
    calc_loss = nn.MSELoss()

    ckpt_interval = num_epochs // 10 if num_epochs > 10 else 5
    best_total_loss = float('inf')
    
    for epoch in range(1, num_epochs+1):
        # train model for one epoch
        model.train()
        total_loss_train = 0.0
        pbar = tqdm(dl_train, desc="{:4d}. epoch".format(epoch))
        for x, y in pbar:
            x, y = x.to(dev), y.to(dev)
            
            opt.zero_grad()
            y_hat = model.forward(x)
            loss = calc_loss(y_hat, y)
            loss.backward()
            opt.step()
            
            total_loss_train += loss.item()
            pbar.set_postfix(loss=loss.item())

        # validate model
        model.eval()
        total_loss_valid = 0.0
        with torch.no_grad():
            for x, y in dl_valid:
                x, y = x.to(dev), y.to(dev)
                y_hat = model.forward(x)
                loss = calc_loss(y_hat, y)
                total_loss_valid += loss.item()
    
        # save every now and then
        if epoch % ckpt_interval == 0:
            model.save_ckpt(file_name="lstm_snapshot_{}.pth".format(epoch))

        # save current best model (on validation dataset)
        new_best = False
        if total_loss_valid < best_total_loss:
            best_total_loss = total_loss_valid
            new_best = True
            model.save_ckpt(file_name="lstm_best.pth")

        print("\nLoss (train/valid): {:.5f} / {:.5f}{}".format(total_loss_train,
            total_loss_valid,
            " *NEW BEST MODEL*" if new_best else ""))
    
    model.save_ckpt(file_name="lstm_finished.pth")


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