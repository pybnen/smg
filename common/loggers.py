import librosa
from midi2audio import FluidSynth
import pypianoroll as pp
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from datetime import datetime
from pathlib import Path
from functools import partial
from smg.common.generate_music import generate_multitrack


class Logger():
    def __init__(self):
        pass

    def add_scalar(self, name, value, step):
        pass

    def add_image(self, name, value, step):
        pass

    def add_audio(self, name, value, sample_rate, s):
        pass

    def add_histogram(self, name, value, step):
        pass

    def add_artifact(self, name, filename):
        pass

    def add_pianoroll_img(self, name, pianoroll, step):
        pass

    def add_pianoroll_audio(self, name, pianoroll, step):
        pass

    def close(self):
        pass


class CompositeLogger():
    def __init__(self):
        self.loggers = []

    def add(self, logger):
        self.loggers.append(logger)

    def add_scalar(self, name, value, step):
        for l in self.loggers:
            l.add_scalar(name, value, step)

    def add_image(self, name, value, step):
        for l in self.loggers:
            l.add_image(name, value, step)

    def add_audio(self, name, value, sample_rate, s):
        for l in self.loggers:
            l.add_audio(name, value, sample_rate, s)

    def add_histogram(self, name, value, step):
        for l in self.loggers:
            l.add_histogram(name, value, step)

    def add_artifact(self, name, filename):
        for l in self.loggers:
            l.add_artifact(name, filename)
    
    def add_pianoroll_img(self, name, pianoroll, step):
        for l in self.loggers:
            l.add_pianoroll_img(name, pianoroll, step)
    
    def add_pianoroll_audio(self, name, pianoroll, step):
        for l in self.loggers:
            l.add_pianoroll_audio(name, pianoroll, step)

    def close(self):
        for l in self.loggers:
            l.close()


class TensorBoardLogger(Logger):

    def __init__(self, *args, **kwargs):
        super().__init__()

        path = Path(kwargs['log_dir'])
        path.mkdir(parents=True, exist_ok=True)
        self.log_dir = path
        self.writer = SummaryWriter(path)
        self.track_info = kwargs['track_info']
        self.generate_multitrack = partial(generate_multitrack, **self.track_info)

    def add_scalar(self, name, value, step):
        self.writer.add_scalar(name, value, global_step=step)

    def add_image(self, name, value, step):
        # value is img_tensor with dim (3, H, W) or (1, H, W)
        self.writer.add_image(name, value, global_step=step)

    def add_audio(self, name, value, sample_rate, step):
        self.writer.add_audio(name, value, global_step=step, sample_rate=sample_rate)

    def add_histogram(self, name, value, step):
        self.writer.add_histogram(name, value, global_step=step)

    def add_pianoroll_img(self, name, pianoroll, step):
        for i, instrument in enumerate(self.track_info['instruments']):
            # pianoroll has dim (seq_length, instruments, n_pitches)
            self.add_image("{}_{}".format(name, instrument), np.expand_dims(pianoroll[:, 0, :].T, 0), step)

    def add_pianoroll_audio(self, name, pianoroll, step):
        midi_file = str(self.log_dir / "tmp.mid")
        wav_file = str(self.log_dir / "tmp.wav")

        multitrack = self.generate_multitrack(pianoroll)
        multitrack.write(midi_file)
        FluidSynth().midi_to_audio(midi_file, wav_file)
        y, sr = librosa.load(wav_file)
        self.add_audio(name, y, sr, step)

    def close(self):
        self.writer.close()


class SacredLogger(Logger):

    def __init__(self, experiment, _run, *args, **kwargs):
        super().__init__()
        self._run = _run
        self.experiment = experiment

    def add_scalar(self, name, value, step):
        self._run.log_scalar(name, value, step)

    def add_artifact(self, name, filename):
        self._run.add_artifact(
            filename=filename,
            name=name
        )