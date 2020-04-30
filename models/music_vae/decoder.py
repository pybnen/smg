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
    def __init__(self, in_features, out_features, z_size, num_layers=1, teacher_forcing=True, temperature=1.0):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.z_size = z_size
        self.num_layers = num_layers
        self.teacher_forcing = teacher_forcing
        self.temperature = temperature
        self.kwargs = {"in_features": in_features,
                       "out_features": out_features,
                       "z_size": z_size,
                       "num_layers": num_layers,
                       "teacher_forcing": teacher_forcing,
                       "temperature": temperature}

        # TODO confirm that lstmcell in a loop yields the same
        #  result as an lstm layer, for teacher_forcing lstmlayer could be used for performance
        #  reasons
        cells = []
        for i in range(num_layers):
            if i == 0:
                cell = nn.LSTMCell(self.in_features, self.z_size, bias=True)
            else:
                cell = nn.LSTMCell(self.z_size, self.z_size, bias=True)
            cells.append(cell)
        self.cells = nn.ModuleList(cells)

        def cells_forward(cell_input, states):
            h_t, c_t = states

            for i, cell in enumerate(self.cells):
                h_t[i], c_t[i] = cell(cell_input, (h_t[i], c_t[i]))
                cell_input = h_t[i]

            return h_t, c_t

        self.cells_forward = cells_forward

        self.linear = nn.Linear(self.z_size, self.out_features, bias=True)

    def forward(self, z, seq_length=None, x=None):
        # only use teacher forcing for training
        teacher_forcing = self.teacher_forcing and self.training
        assert(not teacher_forcing or x is not None)
        assert(seq_length is not None or x is not None)

        batch_size, _ = z.size()

        if teacher_forcing:
            seq_length = x.size(1)

        # init hidden with latent vector

        # TODO z must first be activated through a fully connected layer with tanh

        # TODO check with magenta implementation if all layers of stacked lstm are initialized with z or only the first.
        #   [UPDATE] checked and nope, z is input for linear layer with output size of the sum of all hidden and
        #            cell units (e.g. for 2 layers with hidden 512 the output size is 2048) the output is then used
        #            to init hidden and cell state of lstm cell (no na)

        h_t = [z] * self.num_layers
        c_t = [torch.zeros_like(z).type(z.type())] * self.num_layers

        # start with zero as input
        # TODO check if init all dim with zero is correct
        cur_input = torch.zeros((batch_size, self.in_features)).type(z.type())

        output = []
        for t in range(seq_length):
            h_t, c_t = self.cells_forward(cur_input, (h_t, c_t))
            # x_t = nn.functional.log_softmax(self.linear(h_t), dim=-1)
            x_t = self.linear(h_t[-1])
            output.append(x_t)

            if teacher_forcing:
                cur_input = x[:, t]
            else:
                # Got the following error on cp student server:
                #  "RuntimeError: Expected isFloatingType(grads[i].scalar_type()) to be true, but got false."
                #  I could narrow the error down to the argmax operation, and detaching the output of the network x_t
                #  fixes the error.
                #  on the student server pytorch version 1.5.0+cu101 is used
                logits = x_t.detach()
                sampler = torch.distributions.one_hot_categorical.OneHotCategorical(logits=logits / self.temperature)
                cur_input = sampler.sample()
                #cur_input = x_t.detach().argmax(dim=-1).type(z.type()).view(-1, self.in_features)

        output = torch.stack(output, dim=1)
        return output

    def create_ckpt(self):
        ckpt = {"clazz": ".".join([self.__module__, self.__class__.__name__]),
                "state": self.state_dict(),
                "kwargs": self.kwargs}
        return ckpt


if __name__ == "__main__":
    from copy import deepcopy
    import numpy as np
    torch.manual_seed(0)

    batch_size = 20
    z_size = 10
    seq_len = 5

    in_features = 10
    out_features = 5
    num_layers = 10

    decoder = LstmDecoder(in_features, out_features, z_size, num_layers=num_layers, teacher_forcing=True)

    # now i want to check if an lstm would net me the same result
    lstm = nn.LSTM(in_features, z_size, num_layers=num_layers)
    # copy weights
    for i in range(num_layers):
        cell = decoder.cells[i]
        # set ih
        decoder_param = deepcopy(cell.weight_ih)
        setattr(lstm, "weight_ih_l{}".format(i), decoder_param)

        # set hh
        decoder_param = deepcopy(cell.weight_hh)
        setattr(lstm, "weight_hh_l{}".format(i), decoder_param)

        # set bias ih
        decoder_param = deepcopy(cell.bias_ih)
        setattr(lstm, "bias_ih_l{}".format(i), decoder_param)

        # set bias hh
        decoder_param = deepcopy(cell.bias_hh)
        setattr(lstm, "bias_hh_l{}".format(i), decoder_param)

    for _ in range(10):
        z = torch.randn(batch_size, z_size)
        x = torch.randint(0, 130, size=(batch_size, seq_len, in_features)).float()
        x_hat1 = decoder(z, x=x)

        new_x = torch.cat([torch.zeros((batch_size, 1, in_features)), x[:, :-1, :]], dim=1)
        c0 = torch.zeros((num_layers, batch_size, z_size))
        h0 = torch.stack([z]*num_layers, dim=0)

        seq_out, (h_t, c_t) = lstm(new_x.permute((1, 0, 2)), (h0, c0))
        seq_out = seq_out.permute((1, 0, 2))
        x_hat2 = []
        for i in range(seq_len):
            x_hat2.append(decoder.linear(seq_out[:, i, :]))
        x_hat2 = torch.stack(x_hat2, dim=1)

        assert(np.allclose(x_hat1.detach().numpy(), x_hat2.detach().numpy()))
    print('done')

    # Test teacher_forcing False
    # decoder = LstmDecoder(in_features, out_features, z_size, num_layers=1, teacher_forcing=False)
    # print(list(decoder.named_parameters()))
    #

    # x_hat = decoder(z, seq_length=seq_length)
    # print("x_hat.shape = {}".format(x_hat.size()))
    # print("x_hat", x_hat)






