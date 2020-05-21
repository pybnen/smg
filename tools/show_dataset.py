import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import os.path
import itertools
import tqdm

from smg.datasets.melody_dataset import FixedLengthMelodyDataset, MelodyEncode
from smg.ingredients.data import dataset_train_valid_split
import pypianoroll as pp
from smg.music import melody_lib

MIN_PITCH = 21
MAX_PITCH = 108
NUM_SPECIAL_EVENTS = 2

N_CLASSES = MAX_PITCH - MIN_PITCH + 1 + NUM_SPECIAL_EVENTS

parser = argparse.ArgumentParser(description="Inspect a given melody dataset.")
parser.add_argument("-stat", action="store_true", default=False, help="Calculate class distribution.")

parser.add_argument("--valid_split", type=float, default=0.2, help="Split ratio between train/eval set, set to 0.0 if not split should be made.")  # noqa
parser.add_argument("--use_train", action="store_true", default=False, help="Use train set instead of eval set, applys only if valid split is given, default False")  # noqa
parser.add_argument("dataset_dirname", type=str, help="Directory name containing dataset")
args = parser.parse_args()


def get_event_decoder(min_pitch, max_pitch, num_special_events):
    n_classes = max_pitch - min_pitch + 1 + num_special_events

    def decode_event(event):
        if event < num_special_events:
            # event is a special event
            return event - num_special_events
        event = event - num_special_events + min_pitch
        assert min_pitch <= event <= max_pitch
        return event

    return decode_event


def calc_stats(generator, melody_cnt):
    generator, gen_copy = itertools.tee(generator)
    first_melody = next(gen_copy)

    melody_length = len(first_melody)
    pitch_cnt = np.zeros(shape=(melody_length, N_CLASSES), dtype=np.int32)

    for i, melody in enumerate(tqdm.tqdm(generator, total=melody_cnt)):
        pitch_cnt[np.arange(melody_length), melody] += 1
        if i == 0:
            assert(np.all(first_melody == melody))
    # melody_cnt = i + 1
    assert(np.all(pitch_cnt.sum(axis=1) == melody_cnt))
    assert(pitch_cnt.sum() == melody_cnt * melody_length)

    return pitch_cnt


def melody_slideshow(generator):
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i, melody in enumerate(generator):
        pp.plot_pianoroll(ax, melody_lib.melody_to_pianoroll(melody))
        fig.canvas.draw()
        fig.canvas.flush_events()


def plot_pitch_cnt(pitch_cnt, sufix):
    pitch_distribution = pitch_cnt.sum(axis=0) / pitch_cnt.sum()

    # plot all
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(np.arange(N_CLASSES), pitch_distribution)
    fig.savefig("pitch_distribution_{}.png".format(sufix))
    plt.title("Pitch distribution")
    plt.xlabel("Pitches")
    plt.close(fig)

    # omit first 2 classes
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(np.arange(N_CLASSES)[2:], pitch_distribution[2:])
    fig.savefig("pitch_distribution_pitches_only_{}.png".format(sufix))
    plt.title("Pitch distribution (Special events omitted)")
    plt.xlabel("Pitches")
    plt.close(fig)

    np.save("pitch_cnt_{}.npy".format(sufix), pitch_cnt)


def main():
    if not os.path.isdir(args.dataset_dirname):
        print("Given datset dirname is not a directory: '{}'".format(args.dataset_dirname))
        return

    decode_event_fn = get_event_decoder(MIN_PITCH, MAX_PITCH, NUM_SPECIAL_EVENTS)
    transforms = MelodyEncode(n_classes=N_CLASSES, num_special_events=0)
    ds = FixedLengthMelodyDataset(melody_dir=args.dataset_dirname, transforms=transforms)
    print("Loaded dataset from '{}' with size {}.".format(args.dataset_dirname, len(ds)))

    ds_train = ds_eval = None
    if args.valid_split > 0.0:
        ds_train, ds_eval = dataset_train_valid_split(ds, valid_split=args.valid_split)
        print("Split the dataset in train/eval set, with lengths {} and {}.".format(len(ds_train), len(ds_eval)))

    def melody_generator(dataset, random=True, decode=True):
        if random:
            indexes = np.random.permutation(len(dataset))
        else:
            indexes = np.arange(len(dataset))

        for i in indexes:
            melody = dataset[i]
            if decode:
                yield np.array([decode_event_fn(event) for event in melody.argmax(axis=-1)])
            else:
                yield melody.argmax(axis=-1)

    if args.stat:
        if args.valid_split > 0.0:
            pitch_cnt_test = calc_stats(melody_generator(ds_train, random=False, decode=False), len(ds_train))
            pitch_cnt_eval = calc_stats(melody_generator(ds_eval, random=False, decode=False), len(ds_eval))
            pitch_cnt_total = pitch_cnt_test + pitch_cnt_eval

            plot_pitch_cnt(pitch_cnt_test, "test")
            plot_pitch_cnt(pitch_cnt_eval, "eval")
            plot_pitch_cnt(pitch_cnt_total, "all")
        else:
            pitch_cnt = calc_stats(melody_generator(ds, random=False, decode=False), len(ds))
            plot_pitch_cnt(pitch_cnt, "all")
    else:
        if args.valid_split > 0.0:
            if args.use_train:
                print("Use train set for slideshow.")
                ds = ds_train
            else:
                print("Use eval set for slideshow.")
                ds = ds_eval

        melody_slideshow(melody_generator(ds, random=True, decode=True))


if __name__ == "__main__":
    main()
