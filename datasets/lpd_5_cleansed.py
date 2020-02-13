from torch.utils.data import Dataset
import os
import pypianoroll as pp
import numpy as np
import glob
import torch
from tqdm import tqdm
from enum import Enum


class Instruments(Enum):
    DRUMS = 0
    PIANO = 1
    GUITAR = 2
    BASS = 3
    STRINGS = 4

    def is_drum(self):
        return self == self.__class__.DRUMS

    def midi_program(self):
        midi_program = {
            self.DRUMS: 0,
            self.PIANO: 0,
            self.GUITAR: 25,
            self.BASS: 33,
            self.STRINGS: 48

        }
        return midi_program[self]
    

class LPD5Cleansed(Dataset):

    N_INSTRUMENTS = 5
    DEFAULT_BEAT_RESOLUTION = 24

    def __init__(self, **kwargs):
        '''
        TODO describe kwargs
        '''
        super().__init__()

        self.data_dir = kwargs.pop('data_dir', None)
        data_obj = kwargs.pop('data_obj', None)
        assert(self.data_dir is not None or data_obj is not None)

        self.in_seq_length = kwargs.pop('in_seq_length')
        self.out_seq_length = kwargs.pop('out_seq_length')
        self.seq_length = self.in_seq_length + self.out_seq_length
        self.step_size = kwargs.pop('step_size')

        self.names = []
        self.samples = []
        self.n_sequences = []

        if data_obj is not None:
            self.set_attributes(data_obj)
            self.__load_data_obj__(data_obj)
        else:
            self.set_attributes(kwargs)
            self.__load_data_into_memory__()

    def set_attributes(self, attr_dict):
        self.lowest_pitch = attr_dict['lowest_pitch']
        self.n_pitches = attr_dict['n_pitches']
        self.beat_resolution = attr_dict['beat_resolution']

        self.instruments = attr_dict['instruments']
        self.instruments = [Instruments[inst.upper()] for inst in self.instruments]
        
    def __load_data_obj__(self, data_obj):
        self.names = data_obj['names']
        self.samples = data_obj['samples']

        n_sequences = [self.calc_num_sequences(sample[0][0]) for sample in data_obj['samples']]
        self.cum_n_sequences = np.cumsum(n_sequences)
        self.total_n_sequences = self.cum_n_sequences[-1]

        print('Loaded {} samples with a total of {} sequences!'
            .format(len(self.samples), self.total_n_sequences))

    def __load_data_into_memory__(self):
        # check if path exists
        path = os.path.join(self.data_dir)
        assert(os.path.exists(path))

        print('Loading data into memory...')

        # get all npz files, and check if there is at least one
        files = [x for x in glob.glob(path + '/**/*.npz', recursive=True)]
        assert(len(files) > 0)

        print('{} files found.'.format(len(files)))
        
        downsample_factor = self.__class__.DEFAULT_BEAT_RESOLUTION // self.beat_resolution

        n_sequences = []
        
        # load files
        for f in tqdm(files):
            multitrack_roll = pp.load(str(f))

            # not sure what role downbeats play, but all samples in the lpd5 cleansed dataset
            # contain only one downbeat at the beginning, so I check for that, just in case.
            assert(np.all(multitrack_roll.get_downbeat_steps() == [0]))

            # all samples from the lpd5 cleansed dataset should have a beat resolution of 24
            assert(multitrack_roll.beat_resolution == self.__class__.DEFAULT_BEAT_RESOLUTION)

            multitrack_roll.downsample(downsample_factor)

            assert(multitrack_roll.beat_resolution == self.beat_resolution)

            # transpose to get (timesteps, n_instruments, pitch_range)
            stacked = multitrack_roll.get_stacked_pianoroll().transpose(0, 2, 1)

            # all samples from the lpd5 cleansed dataset should have 5 instruments
            assert(stacked.shape[1] == self.__class__.N_INSTRUMENTS)

            stacked = stacked[:, [inst.value for inst in self.instruments], self.lowest_pitch:self.lowest_pitch + self.n_pitches]

            # only add samples if not empty
            if not np.all(stacked == 0):
                i = np.where(stacked > 0)
                # map values from [0, 127] to [0, 1]
                v = stacked[i] / 127
                        
                self.names.append(multitrack_roll.name)
                self.samples.append((stacked.shape, i, v))

                sample_length = multitrack_roll.get_max_length()
                n_sequences.append(self.calc_num_sequences(sample_length))

        self.cum_n_sequences = np.cumsum(n_sequences)
        self.total_n_sequences = self.cum_n_sequences[-1]

        print('Loaded {} samples with a total of {} sequences!'
            .format(len(self.samples), self.total_n_sequences))

    def __len__(self):
        return self.total_n_sequences

    def __getitem__(self, seq_idx):
        diff = self.cum_n_sequences - seq_idx
        sample_idx = np.asarray(diff > 0).nonzero()[0][0]

        if sample_idx == 0:
            offset = seq_idx * self.step_size
        else:
            offset = abs(diff[sample_idx - 1]) * self.step_size

        # reconstruct sample from triplet
        sample_shape, indices, values = self.samples[sample_idx]
        sample = np.zeros(sample_shape).astype(np.float32)
        sample[indices] = values

        sequence = sample[offset:offset+self.seq_length]
        x, y = sequence[:self.in_seq_length], sequence[self.in_seq_length:]
        return x, y

    def calc_num_sequences(self, sample_len):
        return ((sample_len - self.seq_length) // self.step_size) + 1


if __name__ == "__main__":
    import time
    import sys
    from pathlib import Path
    from smg.experiments.train_lstm import get_data_loader

    samples_file = None
    if len(sys.argv) > 1:
        samples_file = Path(sys.argv[1])

    beat_resolution = 4
    beats_per_measure = 4
    measures_per_sample = 4

    in_seq_length = beat_resolution * beats_per_measure * measures_per_sample
    step_size = beat_resolution * beats_per_measure
    out_seq_length = 1

    if samples_file is None or not samples_file.is_file():
        
        instruments = ['drums', 'piano', 'guitar', 'bass', 'strings']
        
        # test dataset
        kwargs = {
            "data_dir": "../data/examples", # "../data/lpd_5",  
            "lowest_pitch": 24,
            "n_pitches": 72,
            "beat_resolution": beat_resolution,
            "in_seq_length": in_seq_length,
            "out_seq_length": out_seq_length,
            "step_size": step_size,
            "instruments": instruments
        }
        ds = LPD5Cleansed(**kwargs)

    else:
        import pickle
        with open(str(samples_file), 'rb') as f:
            data_obj = pickle.load(f)

        train_obj = dict(data_obj, **{'samples': data_obj['samples']['train'],
                                    'names': data_obj['names']['train']})

        kwargs = {
            "data_dir": "../data/examples", # "../data/lpd_5",  
            "data_obj": train_obj,
            "beat_resolution": beat_resolution,
            "in_seq_length": in_seq_length,
            "out_seq_length": out_seq_length,
            "step_size": step_size,
        }
        ds = LPD5Cleansed(**kwargs)

    #print("n_sequences = {}".format(ds.n_sequences))
    print("Dataset")
    print("-------")
    print(len(ds))

    start_time = time.time()
    x, y = ds[0]
    print("x.shape = {} / type {}".format(x.shape, x.dtype))
    print("y.shape = {} / type {}".format(y.shape, y.dtype))
    print("--- %s seconds ---" % (time.time() - start_time))

    dl = get_data_loader(ds, 64, n_workers=0)
    print("Data Loader")
    print("-----------")
    print(len(dl))


    batch_gen = iter(dl)
    start_time = time.time()
    x, y = next(batch_gen)
    print("x.shape = {} / type {}".format(x.shape, x.type()))
    print("y.shape = {} / type {}".format(y.shape, y.type()))
    print("--- %s seconds ---" % (time.time() - start_time))