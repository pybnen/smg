import torch
import torch.nn as nn


class RandomDecoder(nn.Module):
    def __init__(self, output_size) -> None:
        super().__init__()
        self.output_size = output_size

    def forward(self, z):
        batch_size, _ = z.size()

        return torch.randn(size=(batch_size, *self.output_size))


class LstmDecoder(nn.Module):
    def __init__(self, sample_features, z_size, teacher_forcing=True):
        super().__init__()

        self.sample_features = sample_features
        self.z_size = z_size
        self.teacher_forcing = teacher_forcing

        # TODO confirm that lstmcell in a loop yields the same
        #  result as an lstm layer, for teacher_forcing lstmlayer could be used for performance
        #  reasons
        self.lstm_cell = nn.LSTMCell(self.sample_features, self.z_size, bias=True)
        self.linear = nn.Linear(self.z_size, self.sample_features, bias=True)

    def forward(self, z, seq_length=None, x=None):
        assert(not self.teacher_forcing or x is not None)
        assert(seq_length or x)

        batch_size, _ = z.size()

        if seq_length is None:
            # assume sequence length is in second dimension
            seq_length = x.size(1)

        # init hidden with latent vector
        h_t = z
        c_t = torch.zeros_like(z, dtype=z.dtype)
        x_t = torch.zeros((batch_size, self.sample_features), dtype=z.dtype)

        output = []
        for t in range(seq_length):
            sample = x[:, t] if self.teacher_forcing else x_t
            h_t, c_t = self.lstm_cell(sample, (h_t, c_t))
            # x_t = nn.functional.log_softmax(self.linear(h_t), dim=-1)
            x_t = self.linear(h_t)
            output.append(x_t)

        return torch.stack(output, dim=1)


if __name__ == "__main__":
    import numpy as np

    batch_size = 8
    z_size = 512

    z = torch.randn(batch_size, z_size)

    sample_features = 130

    decoder = LstmDecoder(sample_features, z_size, teacher_forcing=False)

    seq_length = 32
    x_hat = decoder(z, seq_length=seq_length)
    print("x_hat.shape = {}".format(x_hat.size()))




