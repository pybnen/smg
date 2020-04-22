import torch
import torch.nn as nn


class MusicVAE(nn.Module):

    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x):
        return self.encoder(x)

    def reparameterize(self, mu, sigma):
        # std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(sigma)
        return mu + eps * sigma

    def decode(self, z, seq_len):
        return self.decoder(z, seq_len)

    def forward(self, x, seq_len=None):
        seq_len = seq_len or x.size(1)

        mu, sigma = self.encode(x)
        z = self.reparameterize(mu, sigma)
        x_hat = self.decoder(z, seq_len)
        return x_hat, mu, sigma


if __name__ == "__main__":
    import torch
    from smg.datasets.melody_dataset import MelodyEncode, MelodyDataset
    from torch.utils.data import DataLoader
    from .encoder import RandomEncoder, BidirectionalLstmEncoder
    from .decoder import RandomDecoder, LstmDecoder

    batch_size = 8
    seq_length = 512
    features = 130

    ds = MelodyDataset(melody_dir="../data/lmd_full_melody/", transforms=MelodyEncode(), melody_length=seq_length)
    dl = DataLoader(ds, drop_last=True, num_workers=0, batch_size=batch_size)

    # encoder = RandomEncoder(z_size)
    # decoder = RandomDecoder(out_size))
    params = {
        "input_size": 1,
        "hidden_size": 1048,
        "z_size": 1048,
        "num_layers": 2
    }

    encoder = BidirectionalLstmEncoder(**params)
    decoder = LstmDecoder(features, params["z_size"], teacher_forcing=False)
    model = MusicVAE(encoder=encoder, decoder=decoder)

    #x = torch.randint(0, 127, size=(8, *out_size))
    x = next(iter(dl)).unsqueeze(dim=-1)
    x_hat, mu, var_log = model(x, seq_length)

    print("x.shape = {}".format(x.size()))
    print("x_hat.shape = {}".format(x_hat.size()))
    print("mu.shape = {}".format(mu.size()))
    print("var_log.shape = {}".format(var_log.size()))
