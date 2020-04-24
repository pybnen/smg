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
    def __init__(self, in_features, out_features, z_size, teacher_forcing=True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.z_size = z_size
        self.teacher_forcing = teacher_forcing
        self.kwargs = {"in_features": in_features,
                       "out_features": out_features,
                       "z_size": z_size,
                       "teacher_forcing": teacher_forcing}

        # TODO confirm that lstmcell in a loop yields the same
        #  result as an lstm layer, for teacher_forcing lstmlayer could be used for performance
        #  reasons
        self.lstm_cell = nn.LSTMCell(self.in_features, self.z_size, bias=True)
        self.linear = nn.Linear(self.z_size, self.out_features, bias=True)

    def forward(self, z, seq_length=None, x=None):
        # only use teacher forcing for training
        teacher_forcing = self.teacher_forcing and self.training
        assert(not teacher_forcing or x is not None)
        assert(seq_length or x)

        batch_size, _ = z.size()

        if teacher_forcing:
            seq_length = x.size(1)

        # init hidden with latent vector
        h_t = z
        c_t = torch.zeros_like(z).type(z.type())

        # start with zero as input
        cur_input = torch.zeros((batch_size, self.in_features)).type(z.type())

        output = []
        for t in range(seq_length):
            h_t, c_t = self.lstm_cell(cur_input, (h_t, c_t))
            # x_t = nn.functional.log_softmax(self.linear(h_t), dim=-1)
            x_t = self.linear(h_t)
            output.append(x_t)

            if teacher_forcing:
                cur_input = x[:, t]
            else:
                cur_input = x_t.argmax(dim=-1).type(z.type()).view(-1, self.in_features)

        return torch.stack(output, dim=1)

    def create_ckpt(self):
        ckpt = {"clazz": ".".join([self.__module__, self.__class__.__name__]),
                "state": self.state_dict(),
                "kwargs": self.kwargs}
        return ckpt


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




