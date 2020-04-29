import torch
import torch.nn as nn


class RandomEncoder(nn.Module):
    def __init__(self, z_size) -> None:
        super().__init__()
        self.dummy_weight = nn.Parameter(torch.randn(3, 3), requires_grad=True)

        self.z_size = z_size

    def forward(self, x):
        # batch_size, seq_length, features
        batch_size, _ = x.size()
        mu, sigma = torch.randn((2, batch_size, self.z_size))
        return mu, sigma


class BidirectionalLstmEncoder(nn.Module):

    def __init__(self, input_size, hidden_size, z_size, num_layers=1):
        super().__init__()

        # TODO Look at this:
        #  https://medium.com/@sikdar_sandip/implementing-a-variational-autoencoder-vae-in-pytorch-4029a819ccb6
        #  In the article an embedding layer is used, may be interesting, why not?

        self.z_size = z_size
        self.kwargs = {"input_size": input_size,
                       "hidden_size": hidden_size,
                       "z_size": z_size,
                       "num_layers": num_layers}

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=True)

        # multiply hidden size by 2 because of bidirectional
        in_features = hidden_size * 2
        self.enc_mu = nn.Linear(in_features=in_features, out_features=z_size)
        self.enc_sigma = nn.Linear(in_features=in_features, out_features=z_size)

    def forward(self, x):
        # batch_size, seq_length, features
        batch_size, _, _ = x.size()
        # convert to time major, i.e. seq_length, batch_size, features
        x = x.permute((1, 0, 2))

        x, (hn, cn) = self.lstm(x)
        # TODO this blog post "https://towardsdatascience.com/understanding-bidirectional-rnn-in-pytorch-5bd25a5dd66"
        #  suggests to use the first timestep of the rev ersed lstm because is contains the information
        #  from the end to start, maybe ask advisor about this, or just implement and see the difference.
        #  I see the point for a one layer lstm, but for a stacked lstm the upper layers should get information
        #  from both directions (forward, reversed) by using the hiddens of the lower level
        x_T = x[-1]

        mu = self.enc_mu(x_T)
        # TODO paper uses softplus activation for sigma, other VAE implementations seen online
        #  just use linear layer and say the result is the log of the variance,
        #  why????? ask advisor
        sigma = nn.functional.softplus(self.enc_sigma(x_T))
        return mu, sigma

    def create_ckpt(self):
        ckpt = {"clazz": ".".join([self.__module__, self.__class__.__name__]),
                "state": self.state_dict(),
                "kwargs": self.kwargs}
        return ckpt


if __name__ == "__main__":
    from smg.datasets.melody_dataset import MelodyEncode, MelodyDataset
    from torch.utils.data import DataLoader
    params = {
        "input_size": 1,
        "hidden_size": 1048,
        "z_size": 1048,
        "num_layers": 2
    }
    z_size = params.get("z_size")
    input_size = params.get("input_size")
    seq_length = 32

    encoder = BidirectionalLstmEncoder(**params)

    batch_size = 8
    # x = torch.randint(0, 127, size=(batch_size, seq_length, input_size)).type(torch.FloatTensor)
    ds = MelodyDataset(melody_dir="../data/lmd_full_melody/", transforms=MelodyEncode())
    dl = DataLoader(ds, drop_last=True, num_workers=0, batch_size=batch_size)
    x = next(iter(dl)).unsqueeze(dim=-1)

    mu, log_var = encoder(x)
    print("mu.shape = {}".format(mu.size()))
    print("log_var.shape = {}".format(log_var.size()))
