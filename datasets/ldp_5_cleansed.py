from torch.utils.data import Dataset
import os
import pypianoroll as pp
import numpy as np
import glob
import torch
from tqdm import tqdm


class LDP5Cleansed(Dataset):

    N_INSTRUMENTS = 5
    DEFAULT_BEAT_RESOLUTION = 24
    BEATS_PER_MEASURE = 4

    def __init__(self, data_dir, lowest_pitch, n_pitches, beat_resolution, measures_per_sample):
        '''
        :param data_dir: root of dataset
        TODO describe other params
        '''
        super().__init__()

        self.data_dir = data_dir
        self.lowest_pitch = lowest_pitch
        self.n_pitches = n_pitches
        self.beat_resolution = beat_resolution
        self.measures_per_sample = measures_per_sample

        self.timesteps_per_measure = self.__class__.BEATS_PER_MEASURE * self.beat_resolution

        self.names = []
        self.samples = []
        self.n_measures = []
        self.__load_data_into_memory__()

    def __load_data_into_memory__(self):
        # check if path exists
        path = os.path.join(self.data_dir)
        assert(os.path.exists(path))

        # get all npz files, and check if there is at least one
        files = [x for x in glob.glob(path + '/**/*.npz', recursive=True)]
        assert(len(files) > 0)

        downsample_factor = self.__class__.DEFAULT_BEAT_RESOLUTION // self.beat_resolution

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

            # transpose to get (n_instruments, timesteps, pitch_range)
            stacked = multitrack_roll.get_stacked_pianoroll().transpose(2, 0, 1)

            # all samples from the lpd5 cleansed dataset should have 5 instruments
            assert(stacked.shape[0] == self.__class__.N_INSTRUMENTS)

            stacked = stacked[:, :, self.lowest_pitch:self.lowest_pitch + self.n_pitches]

            i = np.where(stacked > 0)
            # map values from [0, 127] to [0, 1]
            v = stacked[i] / 127

            self.samples.append((stacked.shape, i, v))
            self.names.append(multitrack_roll.name)
            self.n_measures.append(multitrack_roll.get_max_length() // self.timesteps_per_measure)

        print('Loaded {} samples!'.format(len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        #TODO right now an item is a sample, and a random sequence of that sample is chosen
        # maybe change that to use every possible sequence, thus an item is a sequence
        measure = np.random.choice(self.n_measures[idx] - (self.measures_per_sample + 1))
        x, y = self.__to_matrix__(self.samples[idx], measure)
        return x, y

    def __to_matrix__(self, triplet, measure):
        x = np.zeros(triplet[0]).astype(np.float32)
        x[triplet[1]] = triplet[2]

        start_measure = measure * self.timesteps_per_measure
        stop_measure = (measure + self.measures_per_sample) * self.timesteps_per_measure

        return x[:, start_measure:stop_measure], x[:, stop_measure:stop_measure + self.timesteps_per_measure]