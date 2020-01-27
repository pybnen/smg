import torch
import torch.nn as nn
import torch.nn.functional as F

class RecurrentSMG(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.timesteps_per_measure = kwargs.pop('timesteps_per_measure')
        self.n_pitches = kwargs.pop('n_pitches')
        self.n_instruments = kwargs.pop('n_instruments')
        self.seq_length = kwargs.pop('seq_length')


        self.input_size = self.n_pitches * self.n_instruments
        self.hidden_size = kwargs.pop('hidden_size')
        self.num_layers = kwargs.pop('num_layers', 1)
        self.dropout = kwargs.pop('dropout', 0)
        
        self.lstm0 = nn.LSTM(self.input_size, self.hidden_size,
            num_layers=self.num_layers, dropout=self.dropout)

        self.dense = nn.Linear(self.hidden_size, self.timesteps_per_measure * self.n_pitches * self.n_instruments)


    def forward(self, x):
        # seq_length = timesteps_per_measure * n_measures
        # x.size() == [batch_size, n_instruments, seq_length, n_pitches]
        batch_size, n_instruments, seq_length, n_pitches = x.size()
        x = x.permute(2, 0, 1, 3).reshape(seq_length, batch_size, -1)

        x, (hn, cn) = self.lstm0.forward(x)
        assert(torch.all(x[-1] == hn[-1]))

        x = self.dense(x[-1])
        # x.size() == [batch_size, n_instruments, timesteps_per_measure, n_pitches]
        return x.view(batch_size, n_instruments, self.timesteps_per_measure, self.n_pitches)


    #TODO save_ckpt
    #TODO load_from_ckpt
