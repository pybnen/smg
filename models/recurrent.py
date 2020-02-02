import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import copy


class RecurrentSMG(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.kwargs = copy(kwargs)
        self.n_pitches = kwargs.pop('n_pitches')
        self.instruments = kwargs.pop('instruments')
        self.n_instruments = len(self.instruments)
        self.out_seq_length = kwargs.pop('out_seq_length')


        self.input_size = self.n_pitches * self.n_instruments
        self.hidden_size = kwargs.pop('hidden_size')
        self.num_layers = kwargs.pop('num_layers', 1)
        self.dropout = kwargs.pop('dropout', 0)
        
        self.lstm0 = nn.LSTM(self.input_size, self.hidden_size,
            num_layers=self.num_layers, dropout=self.dropout)

        self.dense = nn.Linear(self.hidden_size, self.out_seq_length * self.n_instruments * self.n_pitches)


    def forward(self, x):
        # x.size() == [batch_size, in_seq_length, n_instruments, n_pitches]
        batch_size, seq_length, n_instruments, n_pitches = x.size()
        x = x.permute(1, 0, 2, 3).reshape(seq_length, batch_size, -1)

        x, (hn, cn) = self.lstm0.forward(x)
        assert(torch.all(x[-1] == hn[-1]))

        x = self.dense(x[-1])
        # x.size() == [batch_size, out_seq_length, n_instruments, n_pitches]
        return x.view(batch_size, self.out_seq_length, n_instruments, self.n_pitches)

    def save_ckpt(self, file_name, info=None):
        ckpt = {}
        if info is not None:
            ckpt['info'] = info
        ckpt['state'] = self.state_dict()
        ckpt['kwargs'] = self.kwargs

        torch.save(ckpt, file_name)

    @classmethod
    def load_from_ckpt(clazz, ckpt_file, device=None):
        ckpt  = torch.load(ckpt_file, map_location=device)
        model = clazz(**ckpt['kwargs'])
        model.load_state_dict(ckpt['state'])

        return model, ckpt.pop('info', None)


if __name__ == "__main__":
    save = True

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    beat_resolution = 4
    beats_per_measure = 4
    measures_per_sample = 4

    in_seq_length = beat_resolution * beats_per_measure * measures_per_sample
    
    if save:

        step_size = beat_resolution * beats_per_measure
        out_seq_length = 1
        instruments = ['piano', 'drums']
        n_pitches = 72

        kwargs = {
            "hidden_size": 200,
            "num_layers": 1,
            "out_seq_length": out_seq_length, # part of out features
            "instruments": instruments, # part of out features
            "n_pitches": n_pitches, # part of out features
        }

        model =  RecurrentSMG(**kwargs)
    else:
        model, info = RecurrentSMG.load_from_ckpt('ckpt_test.pth')
        n_pitches = model.n_pitches

        print("info", info)
    
    model = model.to(dev)

    batch_size = 8
    n_instruments = len(model.instruments)
    seq = torch.rand((batch_size, in_seq_length, n_instruments, n_pitches))

    pred = model.forward(seq)
    print("input size:".ljust(15), seq.size())
    print("output size:".ljust(15), pred.size())

    if save:
        model.save_ckpt('ckpt_test.pth', {'in_seq_length': in_seq_length})