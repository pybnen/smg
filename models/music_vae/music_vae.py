import torch
import torch.nn as nn
import torch.nn.functional as F
from smg import common


class MusicVAE(nn.Module):

    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x):
        return self.encoder(x)

    def reparameterize(self, mu, sigma):
        # TODO: add parameter to handle this
        if self.training:
            eps = torch.randn_like(sigma)
            return mu + eps * sigma
        else:
            # for evaluation don't sample from q, just use mean
            return mu

    def decode(self, z, input_sequence=None, sequence_length=None):
        return self.decoder(z, input_sequence, sequence_length=sequence_length)

    def forward(self, input_sequence, seq_length=None):
        seq_length = seq_length or input_sequence.size(1)

        mu, sigma = self.encode(input_sequence)
        z = self.reparameterize(mu, sigma)

        # add a `start token` at the beginning of each sequence and truncate last time step
        decoder_input = F.pad(input_sequence, pad=[0, 0, 1, 0, 0, 0])[:, :-1]
        x_hat, sampled_ratio = self.decode(z, input_sequence=decoder_input, sequence_length=seq_length)
        return x_hat, mu, sigma, sampled_ratio

    def create_ckpt(self):
        ckpt = {"encoder": self.encoder.create_ckpt(),
                "decoder": self.decoder.create_ckpt()}
        return ckpt

    @classmethod
    def load_from_ckpt(clazz, ckpt):
        encoder = common.load_class_by_name(ckpt["encoder"]["clazz"], **ckpt['encoder']['kwargs'])
        encoder.load_state_dict(ckpt["encoder"]['state'])

        decoder_clazz = ckpt["decoder"]["clazz"]
        use_hier = decoder_clazz.find("HierarchicalDecoder") != -1
        if use_hier:
            output_decoder_ckpt = ckpt['decoder']['kwargs'].pop('output_decoder_ckpt')
            output_decoder = common.load_class_by_name(output_decoder_ckpt["clazz"], **output_decoder_ckpt['kwargs'])
            decoder = common.load_class_by_name(decoder_clazz, output_decoder=output_decoder, **ckpt['decoder']['kwargs'])  # noqa
        else:
            decoder = common.load_class_by_name(decoder_clazz, **ckpt['decoder']['kwargs'])

        decoder.load_state_dict(ckpt["decoder"]['state'])

        return clazz(encoder=encoder, decoder=decoder)


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

    # encoder = RandomEncoder(z_dim)
    # decoder = RandomDecoder(out_size))
    params = {
        "input_size": 1,
        "hidden_size": 1048,
        "z_dim": 1048,
        "num_layers": 2
    }

    encoder = BidirectionalLstmEncoder(**params)
    decoder = LstmDecoder(features, params["z_dim"], teacher_forcing=False)
    model = MusicVAE(encoder=encoder, decoder=decoder)

    #x = torch.randint(0, 127, size=(8, *out_size))
    x = next(iter(dl)).unsqueeze(dim=-1)
    x_hat, mu, var_log = model(x, seq_length)

    print("x.shape = {}".format(x.size()))
    print("x_hat.shape = {}".format(x_hat.size()))
    print("mu.shape = {}".format(mu.size()))
    print("var_log.shape = {}".format(var_log.size()))
