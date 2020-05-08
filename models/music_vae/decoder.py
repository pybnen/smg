import torch
import torch.nn as nn


class TrainHelper:
    def __init__(self, input_shape, input_type,
                 inputs=None,
                 sequence_length=None,
                 sampling_probability=0.0,
                 aux_input=None,
                 next_inputs_fn=None):

        self.aux_input = aux_input
        self.sampler = torch.distributions.Bernoulli(probs=sampling_probability)
        self.next_inputs_fn = next_inputs_fn
        self.sampled_cnt = 0
        self.inputs = inputs
        self.sequence_length = sequence_length or self.inputs.size(1)
        self.input_sequence_length = self.inputs.size(1) if self.inputs is not None else 0

        self.zero_input = torch.zeros(input_shape).type(input_type)
        self.initial_input = torch.zeros(input_shape).type(input_type)
        if self.inputs is not None:
            self.initial_input = self.inputs[:, 0]

        if self.aux_input is not None:
            self.initial_input = torch.cat((self.initial_input, self.aux_input[:, 0]), dim=-1)

    def initialize(self):
        return self.initial_input

    def next_inputs(self, time, outputs):
        next_time = time + 1
        if next_time >= self.sequence_length:
            return self.zero_input

        teacher_forcing = self.sampler.sample() == 0 and next_time < self.input_sequence_length
        if teacher_forcing:
            next_inputs = self.inputs[:, next_time]
        else:
            self.sampled_cnt += 1
            next_inputs = outputs
            if self.next_inputs_fn is not None:
                next_inputs = self.next_inputs_fn(next_inputs)

        if self.aux_input is not None:
            next_inputs = torch.cat((next_inputs, self.aux_input[:, next_time]), dim=-1)

        return next_inputs


class LstmDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, z_dim, num_layers=1, temperature=1.0, use_z_as_input=True):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.z_dim = z_dim
        self.num_layers = num_layers
        self.temperature = temperature
        self.use_z_as_input = use_z_as_input
        self.sampling_probability = 0.0
        self.eval_allow_teacher_forcing = False

        self.kwargs = {"input_size": input_size, "hidden_size": hidden_size, "output_size": output_size,
                       "z_dim": z_dim, "num_layers": num_layers, "temperature": temperature,
                       "use_z_as_input": use_z_as_input}

        # mapping from latent space to initial state of decoder
        n_state_units = num_layers * hidden_size * 2  # each layer has cell and hidden units
        self.initial_state_embed = nn.Sequential(
            nn.Linear(z_dim, n_state_units, bias=True),
            nn.Tanh())

        self.lstm_layer = nn.LSTM(input_size, hidden_size, bias=True, num_layers=num_layers)
        self.output_layer = nn.Linear(hidden_size, output_size, bias=True)

    def next_inputs(self, inputs):
        logits = inputs / self.temperature

        sampler = torch.distributions.one_hot_categorical.OneHotCategorical(logits=logits)
        return sampler.sample()

    def forward(self, z, sequence_input=None, sequence_length=None):
        batch_size, _ = z.size()
        sequence_length = sequence_length or sequence_input.size(1)

        # get initial states from latent space z
        initial_states = self.initial_state_embed(z)
        ht, ct = initial_states.view(batch_size, self.num_layers, 2, -1).permute(2, 1, 0, 3).contiguous()

        aux_input = None
        if self.use_z_as_input:
            aux_input = z.detach().unsqueeze(dim=1).repeat(1, sequence_length, 1)

        if self.training or self.eval_allow_teacher_forcing:
            sampling_probability = self.sampling_probability
        else:
            sampling_probability = 1.0

        input_features = self.input_size - (self.z_dim if self.use_z_as_input else 0)
        helper = TrainHelper(
            input_shape=(batch_size, input_features),
            input_type=z.type(),
            inputs=sequence_input,
            sequence_length=sequence_length,
            sampling_probability=sampling_probability,
            aux_input=aux_input,
            next_inputs_fn=self.next_inputs)

        # init first outputs TODO maybe use some specific start token instead
        next_inputs = helper.initialize()
        sequence_output = []
        for t in range(sequence_length):
            _, (ht, ct) = self.lstm_layer(next_inputs.unsqueeze(dim=0), (ht, ct))
            current_output = self.output_layer(ht[-1])
            sequence_output.append(current_output)

            next_inputs = helper.next_inputs(t, current_output.detach())  # detach is very important

        sequence_output = torch.stack(sequence_output, dim=1)
        return sequence_output, helper.sampled_cnt / float(sequence_output.size(1) - 1)

    def create_ckpt(self):
        ckpt = {"clazz": ".".join([self.__module__, self.__class__.__name__]),
                "state": self.state_dict(),
                "kwargs": self.kwargs}
        return ckpt


if __name__ == "__main__":
    # TODO adapt to changes
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






