import torch
import torch.distributions
import torch.nn as nn


class TrainHelper:
    def __init__(self, input_shape, input_type,
                 inputs=None,
                 sequence_length=None,
                 sampling_probability=0.0,
                 aux_input=None,
                 next_inputs_fn=None,
                 initial_input=None):

        self.aux_input = aux_input
        self.sampler = torch.distributions.Bernoulli(probs=sampling_probability)
        self.next_inputs_fn = next_inputs_fn
        self.sampled_cnt = 0
        self.inputs = inputs
        self.sequence_length = sequence_length or self.inputs.size(1)
        self.input_sequence_length = self.inputs.size(1) if self.inputs is not None else 0

        self.zero_input = torch.zeros(input_shape).type(input_type)
        if initial_input is not None:
            self.initial_input = initial_input
        else:
            self.initial_input = self.zero_input

    def initialize(self):
        return self.next_inputs(-1, self.initial_input)

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
            # time == -1 should only been called from initialize method,
            #  in this case the output is the zeros tensor, and the next_inputs_fn method
            #  should not be applied
            if self.next_inputs_fn is not None and time != -1:
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

    def set_sampling_probability(self, sampling_probability):
        self.sampling_probability = sampling_probability

    def get_sampling_probability(self):
        return self.sampling_probability

    def set_temperature(self, temperature):
        self.temperature = temperature

    def get_temperature(self):
        return self.temperature

    def allow_teacher_forcing_for_evaluation(self, allow):
        self.eval_allow_teacher_forcing = allow

    def next_inputs(self, inputs):
        logits = inputs / self.temperature

        sampler = torch.distributions.one_hot_categorical.OneHotCategorical(logits=logits)
        return sampler.sample()

    def forward(self, z, sequence_input=None, sequence_length=None, initial_input=None):
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
            next_inputs_fn=self.next_inputs,
            initial_input=initial_input)

        # init first outputs TODO maybe use some specific start token instead
        next_inputs = helper.initialize()
        sequence_output = []
        sampled_output = []
        for t in range(sequence_length):
            _, (ht, ct) = self.lstm_layer(next_inputs.unsqueeze(dim=0), (ht, ct))
            current_output = self.output_layer(ht[-1])
            sequence_output.append(current_output)

            next_inputs = helper.next_inputs(t, current_output.detach())  # detach is very important
            sampled_output.append(next_inputs.detach()[:, :input_features])

        # replace the last element, by sampling from last output distribution
        sampled_output[-1] = self.next_inputs(current_output.detach())

        sequence_output = torch.stack(sequence_output, dim=1)
        return sequence_output, helper.sampled_cnt / float(sequence_output.size(1)), torch.stack(sampled_output, dim=1)

    def create_ckpt(self, with_state=True):
        ckpt = {"clazz": ".".join([self.__module__, self.__class__.__name__]),
                "kwargs": self.kwargs}
        if with_state:
            ckpt["state"] = self.state_dict()
        return ckpt


class HierarchicalDecoder(nn.Module):
    def __init__(self, num_subsequences, z_dim, hidden_size, output_decoder, num_layers=1):
        super().__init__()

        self.num_subsequences = num_subsequences
        self.z_dim = z_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.kwargs = {"num_subsequences": num_subsequences, "z_dim": z_dim, "hidden_size": hidden_size,
                       "num_layers": num_layers, "output_decoder_ckpt": output_decoder.create_ckpt(with_state=False)}

        n_state_units = num_layers * hidden_size * 2  # each layer has cell and hidden units
        self.initial_state_embed = nn.Sequential(
            nn.Linear(z_dim, n_state_units, bias=True),
            nn.Tanh())

        # magenta has option for an autoregressive conductor, but the config used for 16bar melody passes
        # a zero tensor as input, thus use input_size 1 (zero doesn't work)
        self.conductor_rnn = nn.LSTM(input_size=1, hidden_size=hidden_size, bias=True, num_layers=num_layers)

        # NOTE: from the MusicVAE paper
        # "[...] we use a two-layer unidirectional LSTM for the conductor with a hidden state size of
        # 1024 and 512 output dimensions."
        # In the magenta source code, I don't see the mapping from 1024 hidden size to 512 output size,
        # in my understanding (which could very well be wrong) this has to be achieved by a fully connected layer
        # on top of each recurrent output.
        # TODO debug the magenta code and see if output of hierarchical lstm has 1024 or really 512

        self.output_decoder = output_decoder

    def set_sampling_probability(self, sampling_probability):
        self.output_decoder.set_sampling_probability(sampling_probability)

    def get_sampling_probability(self):
        return self.output_decoder.get_sampling_probability()

    def set_temperature(self, temperature):
        self.output_decoder.temperature = temperature

    def get_temperature(self):
        return self.output_decoder.temperature

    def allow_teacher_forcing_for_evaluation(self, allow):
        self.output_decoder.allow_teacher_forcing_for_evaluation(allow)

    def forward(self, z, sequence_input=None, sequence_length=None):
        batch_size, _ = z.size()
        sequence_length = sequence_length or sequence_input.size(1)
        assert(sequence_length % self.num_subsequences == 0)
        subsequence_length = int(sequence_length / self.num_subsequences)

        # get initial states for conductor from latent space z
        initial_states = self.initial_state_embed(z)
        ht, ct = initial_states.view(batch_size, self.num_layers, 2, -1).permute(2, 1, 0, 3).contiguous()

        subsequences = None
        if sequence_input is not None:
            output_size = sequence_input.size(-1)
            subsequences = sequence_input.view(batch_size, self.num_subsequences, -1, output_size)
            assert(subsequences.size(2) == subsequence_length)

        sequence_output = []
        sampled_output = []
        sample_ratios = 0.0
        initial_output_decoder_input = None
        for i in range(self.num_subsequences):
            # get a "empty" input for one time step
            next_input = torch.zeros((1, batch_size, 1)).type(z.type())

            _, (ht, ct) = self.conductor_rnn(next_input, (ht, ct))
            # TODO: there may or may not be an linear output layer (see TODO in init method)
            current_c = ht[-1]

            subsequence_input = subsequences[:, i] if subsequences is not None else None
            subsequence_output, sample_ratio, subsequence_sampled_output = self.output_decoder(current_c,
                                                                                               sequence_input=subsequence_input,  # noqa
                                                                                               sequence_length=subsequence_length,  # noqa
                                                                                               initial_input=initial_output_decoder_input)  # noqa
            sample_ratios += sample_ratios
            sequence_output.append(subsequence_output)
            sampled_output.append(subsequence_sampled_output)
            initial_output_decoder_input = subsequence_sampled_output[:, -1]

        return torch.cat(sequence_output, dim=1),\
               sample_ratios / self.num_subsequences,\
               torch.cat(sampled_output, dim=1)

    def create_ckpt(self, with_state=True):
        ckpt = {"clazz": ".".join([self.__module__, self.__class__.__name__]),
                "kwargs": self.kwargs}
        if with_state:
            ckpt["state"] = self.state_dict()
        return ckpt


def get_hierarchical_decoder_for_testing():
    z_dim = 8
    n_classes = 5
    c_dim = 12
    hidden_size = 3

    core_decoder = LstmDecoder(input_size=n_classes + c_dim,
                               hidden_size=hidden_size,
                               output_size=n_classes,
                               num_layers=2,
                               z_dim=c_dim)
    return HierarchicalDecoder(num_subsequences=3, z_dim=z_dim,
                               hidden_size=c_dim, output_decoder=core_decoder, num_layers=2)


def get_lstm_decoder_for_testing():
    z_dim = 8
    n_classes = 5
    hidden_size = 2
    return LstmDecoder(n_classes + z_dim, hidden_size, n_classes, num_layers=2, z_dim=z_dim)


def test_decoder(decoder):
    import torch.nn.functional as F
    torch.manual_seed(0)

    print("Test decoder {}".format(repr(decoder)))
    print(repr(decoder.create_ckpt(with_state=False)))
    print("\n--------------------------------------------\n")

    z_dim = 8
    n_classes = 5
    batch_size = 6
    sequence_length = 9

    z = torch.randn(batch_size, z_dim)
    with torch.no_grad():
        # test decode without input (e.g. pure sampling)
        x_hat, sample_ratio = decoder(z, sequence_length=sequence_length)
        assert (sample_ratio == 1.0)
        assert (x_hat.size() == torch.Size([batch_size, sequence_length, n_classes]))

        # test decode with input (e.g. teacher forcing)
        decoder.set_sampling_probability(0.0)  # (= teacher forcing)
        x = torch.distributions.one_hot_categorical.OneHotCategorical(
            probs=torch.ones((batch_size, sequence_length, n_classes))).sample()  # noqa
        shifted_x = F.pad(x, pad=[0, 0, 1, 0, 0, 0])[:, :-1]  # add zero vector at beginning each sequence
        x_hat, sample_ratio = decoder(z, sequence_input=shifted_x)
        assert (sample_ratio == 0.0)
        assert (x_hat.size() == torch.Size([batch_size, sequence_length, n_classes]))

        # test decode with input, but dont use teacher forcing
        decoder.set_sampling_probability(1.0)
        x_hat, sample_ratio = decoder(z, sequence_input=shifted_x)
        assert (sample_ratio == 1.0)
        assert (x_hat.size() == torch.Size([batch_size, sequence_length, n_classes]))


if __name__ == "__main__":
    test_decoder(get_lstm_decoder_for_testing())
    test_decoder(get_hierarchical_decoder_for_testing())
