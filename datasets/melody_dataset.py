import torch
from torch.utils.data import Dataset
from glob import glob
import numpy as np
import pretty_midi
from smg.music import melody_lib

STEPS_PER_QUARTER = 4
STEPS_PER_BAR = STEPS_PER_QUARTER * 4
DEFAULT_MEL_LEN = STEPS_PER_BAR * 16

MIN_MIDI_PROGRAM = 0
MAX_MIDI_PROGRAM = 127


class MelodyDataset(Dataset):

    def __init__(self, **kwargs):
        """
        melody_length: length of music sample in bars
        transforms: transform music
        """
        super().__init__()
        self.melody_dir = kwargs.get("melody_dir", None)
        self.melody_length = kwargs.get("melody_length", DEFAULT_MEL_LEN)
        self.transforms = kwargs.get("transforms", None)

        self.stride = STEPS_PER_BAR
        self.melody_files = glob(self.melody_dir + "/**/*.npy", recursive=True)
        self._load_melodies_to_memory()

    def _load_melodies_to_memory(self):
        melodies = []
        for melody_file in self.melody_files:
            melody = list(np.load(melody_file))

            if self.melody_length is not None:
                for i in range(self.melody_length, len(melody) + 1, self.stride):
                    melody_slice = melody[i - self.melody_length: i]
                    for k, event in enumerate(melody_slice):
                        if event not in (-1, -2):
                            break
                        melody_slice[k] = -2

                    melodies.append(melody_slice)
            else:
                melodies.append(melody)

        melodies = list(set(tuple(m) for m in melodies))
        self.melodies = melodies

    def __len__(self):
        return len(self.melodies)

    def __getitem__(self, idx):
        melody = np.array(self.melodies[idx], dtype=np.float32)
        if self.transforms is not None:
            melody = self.transforms(melody)
        return melody


class FixedLengthMelodyDataset(Dataset):
    def __init__(self, **kwargs):
        """
        transforms: transform music
        """
        super().__init__()
        self.melody_dir = kwargs.get("melody_dir", None)
        self.transforms = kwargs.get("transforms", None)
        self.melody_files = glob(self.melody_dir + "/**/*.npy", recursive=True)

    def __len__(self):
        return len(self.melody_files)

    def __getitem__(self, idx):
        melody = np.load(self.melody_files[idx]).astype(np.float32)
        if self.transforms is not None:
            melody = self.transforms(melody)
        return melody


class MelodyEncode:
    def __init__(self, n_classes, num_special_events=2):
        self.n_classes = n_classes
        self.num_special_events = num_special_events

    def __call__(self, sequence):
        # TODO as of now expects integer values from [-2, x],
        #  but the pitch range could also be restricted, e.g. [-2, -1] union [22, 90]
        sequence = sequence + self.num_special_events

        seq_length = sequence.shape[0]
        one_hot = np.zeros((seq_length, self.n_classes), dtype=np.float32)
        one_hot[np.arange(seq_length), sequence.astype(np.int32)] = 1.0
        return one_hot


class MelodyDecode:
    def __call__(self, encoded_melody):
        return encoded_melody - 2


if __name__ == "__main__":
    import time
    from torch.utils.data import DataLoader
    from torchvision import transforms

    #melody_dir = "../data/lmd_full_melody/"
    #ds = MelodyDataset(melody_dir=melody_dir, melody_length=5, transforms=MelodyEncode(n_classes=130))

    melody_dir = "../data/lmd_full_melody_2/"
    ds = FixedLengthMelodyDataset(melody_dir=melody_dir, transforms=MelodyEncode(n_classes=90, num_special_events=0))

    print("Dataset")
    print("-------")
    print(len(ds))

    start_time = time.time()
    for i in range(5):
        seq = ds[i]
        print("seq.shape = {} / type {}".format(seq.shape, seq.dtype))
    # print(seq)
    print("--- %s seconds ---" % (time.time() - start_time))

    dl = DataLoader(ds, batch_size=512, num_workers=0, drop_last=True, shuffle=True)

    print("Data Loader")
    print("-------")
    print(len(dl))

    start_time = time.time()
    seq_batch = next(iter(dl))
    print("seq_batch.shape = {} / type {}".format(seq_batch.shape, seq_batch.dtype))
    print("--- %s seconds ---" % (time.time() - start_time))

    for seq_batch in dl:
        assert np.all(seq_batch.numpy() >= 0)
