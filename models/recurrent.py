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

        # lstm
        self.hidden_size = kwargs.pop('hidden_size')
        self.num_layers = kwargs.pop('num_layers', 1)
        self.dropout = kwargs.pop('dropout', 0)

        # dense layer
        self.dense_layer_hiddens = kwargs.pop('dense_layer_hiddens')

        in_features = self.n_pitches * self.n_instruments
        out_features = self.out_seq_length * self.n_instruments * self.n_pitches
        
        self.lstm0 = nn.LSTM(in_features, self.hidden_size,
            num_layers=self.num_layers, dropout=self.dropout)


        dense_layer_sizes = [self.hidden_size, *self.dense_layer_hiddens, out_features]
        dense_layers = []
        dense_in = dense_layer_sizes[0]
        for i in range(1, len(dense_layer_sizes)):
            # add relu activation between layers
            if len(dense_layers) > 0:
                dense_layers.append(nn.ReLU())
            
            dense_out = dense_layer_sizes[i]
            dense_layers.append(nn.Linear(dense_in, dense_out))
            dense_in = dense_out
        self.dense = nn.Sequential(*dense_layers)
        

    def forward(self, x):
        # x.size() == [batch_size, in_seq_length, n_instruments, n_pitches]
        batch_size, seq_length, n_instruments, n_pitches = x.size()
        x = x.permute(1, 0, 2, 3).reshape(seq_length, batch_size, -1)

        x, (hn, cn) = self.lstm0.forward(x)
        assert(torch.all(x[-1] == hn[-1]))

        x = self.dense(x[-1])
        # x.size() == [batch_size, out_seq_length, n_instruments, n_pitches]
        return x.view(batch_size, self.out_seq_length, n_instruments, self.n_pitches)

    def create_ckpt(self):
        is_training = self.training
        # set to train, not sure if needed, saw in forum post
        # https://discuss.pytorch.org/t/loading-a-saved-model-for-continue-training/17244/2
        if not is_training:
            self.train()

        ckpt = {}
        ckpt['state'] = self.state_dict()
        ckpt['kwargs'] = self.kwargs
        
        # set back to previous state
        if not is_training:
            self.eval()
        
        return ckpt

    @classmethod
    def load_from_ckpt(clazz, ckpt_file, device=None):
        ckpt  = torch.load(ckpt_file, map_location=device)
        
        model_ckpt = ckpt['model']
        model = clazz(**model_ckpt['kwargs'])
        model.load_state_dict(model_ckpt['state'])

        return model


if __name__ == "__main__":
    import sys

    load_model = len(sys.argv) == 2 and sys.argv[1] == 'load'

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    beat_resolution = 4
    beats_per_measure = 4
    measures_per_sample = 4

    in_seq_length = beat_resolution * beats_per_measure * measures_per_sample
    
    if not load_model:
        step_size = beat_resolution * beats_per_measure
        out_seq_length = 16
        instruments = ['piano']
        n_pitches = 72

        kwargs = {
            "hidden_size": 200,
            "dense_layer_hiddens": [],
            "num_layers": 1,
            "out_seq_length": out_seq_length, # part of out features
            "instruments": instruments, # part of out features
            "n_pitches": n_pitches, # part of out features
        }

        model =  RecurrentSMG(**kwargs)
    else:
        model = RecurrentSMG.load_from_ckpt('ckpt_test.pth')
        n_pitches = model.n_pitches

    model = model.to(dev)

    batch_size = 8
    n_instruments = len(model.instruments)
    seq = torch.rand((batch_size, in_seq_length, n_instruments, n_pitches))

    pred = model.forward(seq)
    print(model)
    print("input size:".ljust(15), seq.size())
    print("output size:".ljust(15), pred.size())

    if not load_model:
        ckpt = model.create_ckpt()
        torch.save({'model': ckpt}, 'ckpt_test.pth')