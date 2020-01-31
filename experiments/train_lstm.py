from sacred import Experiment
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
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
def get_dataset(data_dir, lowest_pitch, n_pitches, beat_resolution, in_seq_length, out_seq_length, step_size):
    kwargs = {
        "data_dir": data_dir, 
        "lowest_pitch": lowest_pitch,
        "n_pitches": n_pitches,
        "beat_resolution": beat_resolution,
        "in_seq_length": in_seq_length,
        "out_seq_length": out_seq_length,
        "step_size": step_size
    }
    return LPD5Cleansed(**kwargs)


@ex.capture
def get_model(hidden_size, num_layers, in_seq_length, out_seq_length, n_instruments, n_pitches):
    kwargs = {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "in_seq_length": in_seq_length, 
        "out_seq_length": out_seq_length, # part of out features
        "n_instruments": n_instruments, # part of out features
        "n_pitches": n_pitches, # part of out features
    }
    return RecurrentSMG(**kwargs)


@ex.capture
def run(model, data_loader, dev, num_epochs=10, lr=1e-2):
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    ckpt_interval = num_epochs // 10 if num_epochs > 10 else 5
    
    for epoch in range(1, num_epochs+1):
        pbar = tqdm(data_loader, desc="{:4d}. epoch".format(epoch))
        for x, y in pbar:
            x, y = x.to(dev), y.to(dev)
            
            opt.zero_grad()
            y_hat = model.forward(x)
            loss = loss_fn(y_hat, y) * 1e3
            loss.backward()
            opt.step()
            
            pbar.set_postfix(loss=loss.item())
        
        # save every now and then
        if epoch % ckpt_interval == 0:
            torch.save(model.state_dict(), 'lstm_snapshot_{}.pth'.format(epoch))
    
    torch.save(model.state_dict(), 'lstm_finished.pth')


@ex.config
def config():
    # data_loader config
    batch_size = 8
    n_workers = 0

    # dataset config
    data_dir = "../data/examples" # #"../../../data/lpd_5",

    # model configs
    hidden_size = 200
    num_layers = 3

    # train configs
    num_epochs = 10
    lr = 1e-2

    # general configs
    n_instruments = 5
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
    data_loader = get_data_loader(dataset, shuffle=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model()
    model = model.to(dev)

    run(model, data_loader, dev)