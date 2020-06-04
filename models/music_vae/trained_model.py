import numpy as np
import torch
from torch.distributions.categorical import  Categorical
import torch.nn.functional as F

from smg.datasets.melody_dataset import MelodyEncode
from smg.models.music_vae.music_vae import MusicVAE


class TrainedModel:

    def __init__(self, ckpt_path, device):
        self.device = device

        with open(ckpt_path, 'rb') as file:
            ckpt = torch.load(file, map_location=device)
            model_ckpt = ckpt['model']

        self.model = MusicVAE.load_from_ckpt(model_ckpt).to(device)
        self.model.decoder.set_sampling_probability(1.0)
        self.z_dim = self.model.encoder.z_dim

        self.min_pitch = 21
        self.max_pitch = 108
        self.num_special_events = 2
        self.n_classes = self.max_pitch - self.min_pitch + 1 + self.num_special_events

    def melodies_to_tensors(self, melodies):
        # each event should be either a special event or in allowed pitch range
        is_special_event = np.logical_and(melodies >= -self.num_special_events, melodies < 0)
        is_in_pitch_range = np.logical_and(melodies >= self.min_pitch, melodies <= self.max_pitch)
        assert np.all(np.logical_or(is_special_event, is_in_pitch_range))

        melodies[melodies >= 0] -= self.min_pitch
        melodies += self.num_special_events

        return F.one_hot(torch.from_numpy(melodies.astype(np.int64)), self.n_classes).type(torch.FloatTensor)

    def tensors_to_melodies(self, melodie_tensors):
        melodie_tensors -= self.num_special_events
        melodie_tensors[melodie_tensors >= 0] += self.min_pitch
        return melodie_tensors.detach().cpu().numpy()

    def sample(self, n, length, temperature=1.0, same_z=False):
        if same_z:
            z = torch.randn((1, self.z_dim)).repeat(n, 1)
        else:
            z = torch.randn((n, self.z_dim))
        return self.decode(z, length, temperature=temperature)

    def encode(self, melodies):
        """Encode melodies to latent space

        melodies: np.array of integer type encoded with special events, shape (n_melodies, melody_length)

        Returns torch.tensor, shape (n_melodies, z_dim)"""
        tensor_melodies = self.melodies_to_tensors(melodies)
        return self.encode_tensors(tensor_melodies)

    def encode_tensors(self, tensor_melodies):
        self.model.eval()
        with torch.no_grad():
            tensor_melodies = tensor_melodies.to(self.device)
            mu, sigma = self.model.encode(tensor_melodies)
            z = self.model.reparameterize(mu, sigma)
            return z, mu, sigma

    def decode(self, z, length, temperature=1.0):
        """Decode latent space to melodies

        z: torch.tensor, shape (n_melodies, z_dim)

        Returns np.array of integer, shape (n_melodies, length)"""
        tensor_melodies, ouput_logits = self.decode_to_tensors(z, length, temperature=temperature)
        return self.tensors_to_melodies(tensor_melodies), ouput_logits

    def decode_to_tensors(self, z, length, temperature=1.0):
        self.model.decoder.set_temperature(temperature)
        self.model.eval()
        with torch.no_grad():
            z = z.to(self.device)
            output_logits, _, sampled_output = self.model.decode(z, sequence_length=length)

            # output_distribution = Categorical(logits=output_logits.detach() / temperature)
            # tensor_melodies = output_distribution.sample()

            #tensor_melodies = output_logits.argmax(dim=-1)

            # sampled output is one hot encoded
            tensor_melodies = sampled_output.argmax(dim=-1)
            return tensor_melodies, output_logits.detach()

    def interpolate(self, start_melody, end_melody, num_steps, length, temperature=1.0):
        def _slerp(p0, p1, t):
            """Spherical linear interpolation."""
            omega = np.arccos(np.dot(np.squeeze(p0 / np.linalg.norm(p0)),
                                     np.squeeze(p1 / np.linalg.norm(p1))))
            so = np.sin(omega)
            return np.sin((1.0 - t) * omega) / so * p0 + np.sin(t * omega) / so * p1

        melodies = np.array([start_melody, end_melody])
        z, _, _ = self.encode(melodies)
        z = z.detach().cpu()
        interpolated_z = torch.stack([_slerp(z[0], z[1], t) for t in np.linspace(0, 1, num_steps)])
        return self.decode(interpolated_z, length, temperature=temperature)


class FakeModel:
    def __init__(self, ckpt_path, z_dim):
        self.z_dim = z_dim
        self.model = None  # load from checkpoint path

    def sample(self, n, length, temperature=1.0, same_z=False):
        if same_z:
            z = torch.randn((1, self.z_dim)).repeat(n, 1)
        else:
            z = torch.randn((n, self.z_dim))
        return self.decode(z, length, temperature=temperature)

    def encode(self, melodies):
        """Encode melodies to latent space

        melodies: np.array of integer type encoded with special events, shape (n_melodies, melody_length)

        Returns torch.tensor, shape (n_melodies, z_dim)"""
        return torch.ones((melodies.shape[0], self.z_dim))  # (torch.randn((melodies.shape[0], self.z_dim))

    def decode(self, z, length, temperature=1.0):
        """Decode latent space to melodies

        z: torch.tensor, shape (n_melodies, z_dim)

        Returns np.array of integer, shape (n_melodies, length)"""
        return np.random.randint(-2, 128, size=(z.shape[0], length))

    def interpolate(self, start_melody, end_melody, num_steps, length, temperature=1.0):
        def _slerp(p0, p1, t):
            """Spherical linear interpolation."""
            omega = np.arccos(np.dot(np.squeeze(p0 / np.linalg.norm(p0)),
                                     np.squeeze(p1 / np.linalg.norm(p1))))
            so = np.sin(omega)
            return np.sin((1.0 - t) * omega) / so * p0 + np.sin(t * omega) / so * p1

        melodies = np.array([start_melody, end_melody])
        z = self.encode(melodies).detach().cpu().numpy()
        z = np.array([_slerp(z[0], z[1], t) for t in np.linspace(0, 1, num_steps)])
        return self.decode(z, length, temperature=temperature)

