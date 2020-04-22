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


class MelodyEncode:
    def __call__(self, melody):
        # TODO remove magic constant, as of now expects integer values from [-2, x],
        #  but the pitch range could also be restricted, e.g. [-2, -1] union [22, 90]
        return melody + 2


class MelodyDecode:
    def __call__(self, encoded_melody):
        return encoded_melody - 2


if __name__ == "__main__":
    import time
    from torch.utils.data import DataLoader

    melody_dir = "../data/lmd_full_melody/"
    ds = MelodyDataset(melody_dir=melody_dir, melody_length=DEFAULT_MEL_LEN, transforms=MelodyEncode())

    print("Dataset")
    print("-------")
    print(len(ds))

    start_time = time.time()
    for i in range(5):
        seq = ds[i]
        print("seq.shape = {} / type {}".format(seq.shape, seq.dtype))
    # print(seq)
    print("--- %s seconds ---" % (time.time() - start_time))

    dl = DataLoader(ds, batch_size=16, num_workers=0)

    print("Data Loader")
    print("-------")
    print(len(dl))

    start_time = time.time()
    seq_batch = next(iter(dl))
    print("seq_batch.shape = {} / type {}".format(seq_batch.shape, seq_batch.dtype))
    print("--- %s seconds ---" % (time.time() - start_time))

    for seq_batch in dl:
        assert np.all(seq_batch.numpy() >= 0)
