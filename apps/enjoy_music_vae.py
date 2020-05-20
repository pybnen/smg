import argparse


def add_output_args(parser):
    parser.add_argument("--out_dirname", "-o", type=str, help="dirname where output will be saved.")
    parser.add_argument("--delete", "-d", action="store_true", default=False,
                        help="Delete files in out dir if prompt is confirmed.")
    parser.add_argument("--force", "-f", action="store_true", default=False,
                        help="Delete files in out dir without prompt.")


def add_ckpt_args(parser):
    parser.add_argument("--ckpt_path", type=str,
                        help="Path to checkpoint, if not set environment variable CKPT_PATH is used.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Used for sampling next input from current output, lower temperature favours best guess, very low temp would basically result in argmax.")  # noqa
    parser.add_argument("--teacher_forcing", "-t", action="store_true", default=False,
                        help="Use teacher forcing")


parser = argparse.ArgumentParser(description="Whats not to enjoy in an freshly trained MusicVAE!")
# TODO better way to get this value, should be in model and dataset
parser.add_argument("--n_classes", type=int, default=90)
parser.add_argument("--melody_length", type=int, default=32)

subparsers = parser.add_subparsers(title="commands", dest='command', required=True)

list_melodies_parser = subparsers.add_parser("list",  help="List and optionally plots melodies of given midi file.")
add_output_args(list_melodies_parser)
list_melodies_parser.add_argument("midi_path", type=str, help="Path of midi file.")

reconstruct_parser = subparsers.add_parser("reconstruct", help="Reconstruct a given melodie, with a delicious MusicVAE.")  # noqa
add_ckpt_args(reconstruct_parser)
add_output_args(reconstruct_parser)
reconstruct_parser.add_argument("melody_info", type=str, help="Comma separated list, no spaces in between containing: path of midi file, melody index and start bar.")  # noqa

interpolate_parser = subparsers.add_parser("interpolate", help="Interpolate between two melodies")
add_ckpt_args(interpolate_parser)
add_output_args(interpolate_parser)
interpolate_parser.add_argument("--to-midi", "-m", action="store_true", default=False, help="Create midi file in plot dir.")  # noqa
interpolate_parser.add_argument("--num_steps", type=int, default=7, help="Interpolation steps, including start and end")  # noqa
interpolate_parser.add_argument("start_melody_info", type=str,help="Comma separated list, no spaces in between containing: path of midi file, melody index and start bar.")  # noqa
interpolate_parser.add_argument("end_melody_info", type=str, help="Comma separated list, no spaces in between containing: path of midi file, melody index and start bar.")  # noqa

sample_parser = subparsers.add_parser("sample", help="Sampling is what makes the world go round, woudn't want to eat pancakes every day, would you? Well actually...")  # noqa
add_ckpt_args(sample_parser)
add_output_args(sample_parser)
sample_parser.add_argument("--to-midi", "-m", action="store_true", default=False, help="Create midi file in plot dir.")  # noqa
sample_parser.add_argument("--seed", type=str, help="Comma separated list, no spaces in between containing: path of midi file, melody index and start bar.")  # noqa
sample_parser.add_argument("--length", "-l", type=int, default=32, help="Sequence Length.")

args = parser.parse_args()


from pathlib import Path
import os
import shutil

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from smg.datasets.melody_dataset import FixedLengthMelodyDataset, MelodyEncode
from smg.ingredients.data import dataset_train_valid_split
from smg.models.music_vae.music_vae import MusicVAE

from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import pypianoroll as pp

from smg.music import melody_lib
import pretty_midi


STEPS_PER_BAR = 16
MELODY_LENGTH = args.melody_length  # 32

MIN_PITCH = 21
MAX_PITCH = 108
NUM_SPECIAL_EVENTS = 2


class ExceedMelodyError(Exception):
    def __init__(self, melody_length, start_bar, window_length):
        self.melody_length = melody_length
        self.start_bar = start_bar
        self.window_length = window_length


# TODO factor into utils
def load_model_from_ckpt(ckpt_path, device):
    with open(ckpt_path, 'rb') as file:
        ckpt = torch.load(file, map_location=device)
        model_ckpt = ckpt['model']
    return MusicVAE.load_from_ckpt(model_ckpt).to(device)


# TODO factor this out to melody_lib
def extract_melodies(midi_path):
    # TODO maybe use magenta OneHotMelodyConverter
    # data.OneHotMelodyConverter(
    #        valid_programs=data.MEL_PROGRAMS,
    #        skip_polyphony=False,
    #        max_bars=100,  # Truncate long melodies before slicing.
    #        slice_bars=2,
    #        steps_per_quarter=4)
    pm = pretty_midi.PrettyMIDI(midi_file=midi_path)
    melodies = melody_lib.midi_to_melody(midi_path, pm, gap_bars=float("inf"))
    return melodies


def get_melody_from_info(melody_info):
    midi_path, melody_idx, start_bar = [info.strip() for info in melody_info.split(",")]
    melody_idx = int(melody_idx)
    start_bar = int(start_bar)

    melodies = extract_melodies(midi_path)
    melody = melodies[melody_idx]["events"]
    if start_bar * STEPS_PER_BAR + MELODY_LENGTH > len(melody):
        raise ExceedMelodyError(len(melody) // STEPS_PER_BAR, start_bar, MELODY_LENGTH // STEPS_PER_BAR)

    melody = melody[start_bar * STEPS_PER_BAR:start_bar * STEPS_PER_BAR + MELODY_LENGTH]
    # remove any -1 events before the first pitch is played
    for k, event in enumerate(melody):
        if event not in (-1, -2):
            break
        melody[k] = -2

    return melody, (midi_path, melody_idx, start_bar)


def get_event_encoder(min_pitch, max_pitch, num_special_events):
    n_classes = max_pitch - min_pitch + 1 + num_special_events

    def encode_event(event):
        if event < num_special_events:
            # event is a special event
            return event + num_special_events

        assert min_pitch <= event <= max_pitch
        event = event + num_special_events - min_pitch
        return event

    return encode_event


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


def melody_to_sequence(melody, encode_event_fn):
    melody = np.array([encode_event_fn(event) for event in melody])
    n_classes = 90
    sequence_length = len(melody)
    sequence = torch.zeros((sequence_length, n_classes), dtype=torch.float32)
    sequence[np.arange(sequence_length), melody] = 1.0
    return sequence


def sequence_to_melody(sequence, decode_event_fn):
    melody = sequence.argmax(dim=-1).cpu().numpy()
    return np.array([decode_event_fn(event) for event in melody])


def reconstruct_melody(model, device, melody, encode_event_fn, decode_event_fn):
    model.eval()
    with torch.no_grad():
        input_sequence = melody_to_sequence(melody, encode_event_fn).to(device)
        output_sequences, _, _, _ = model.forward(input_sequence.unsqueeze(dim=0))
    return sequence_to_melody(output_sequences[0], decode_event_fn)


def create_or_empty_dir(dirname, delete=True, force=False):
    """Creates directory if already existing deletes files.

    A prompt is shown before deleting the file, use force parameter to directly delete files."""
    dir_path = Path(dirname)

    if not dir_path.is_dir():
        dir_path.mkdir(parents=True)
    elif delete:
        files = list(dir_path.glob("*"))
        delete_files = len(files) > 0 and\
                       (force or input("Delete {} files in {} [y/n]: ".format(len(files), dirname)) == "y")
        if delete_files:
            for f in files:
                try:
                    f.unlink()
                except OSError as e:
                    print("Error: %s : %s" % (f, e.strerror))
    return dir_path


def list_melodies(midi_path, out_dirname=None):
    """Lists extracted melodies of midi file given by path.
    Shows melody idx, length and optionally path to melody plot"""
    melodies = extract_melodies(midi_path)

    pp_multitrack = pp.parse(midi_path)

    for i, melody in enumerate(melodies):
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(211)

        pianoroll = melody_lib.melody_to_pianoroll(melody["events"])
        pp.plot_pianoroll(ax, pianoroll)

        ticks = np.arange(0, pianoroll.shape[0], STEPS_PER_BAR * 2)
        ax.set_xticks(ticks)
        ax.set_xticklabels(np.arange(0, len(ticks)) * 2)
        ax.grid(True, axis='x')
        plt.title('Extracted melody')

        ax = fig.add_subplot(212)
        pp.plot_pianoroll(ax, pp_multitrack.tracks[melody["instrument"]].pianoroll)
        plt.title('Original midi track')

        plot_basename = ""
        if out_dirname is not None:
            plot_basename = "lst_{}_{}.png".format(Path(midi_path).stem, i)
            fig.savefig(str(out_dirname / plot_basename))
        plt.show()
        plt.close()

        print("idx: {:2d},\tlength: {:3d}{}".format(
            i, len(melody["events"]) // 16, "" if out_dirname is None else ",\t" + plot_basename))


def reconstruct(model, device, melody_info, out_dirname=None):
    """"Reconstructs melody by given model.

    melody_info is string, containing a comma separated list of midi_path, melody_idx, start_bar"""
    melody, (midi_path, melody_idx, start_bar) = get_melody_from_info(melody_info)

    encode_event_fn = get_event_encoder(MIN_PITCH, MAX_PITCH, NUM_SPECIAL_EVENTS)
    decode_event_fn = get_event_decoder(MIN_PITCH, MAX_PITCH, NUM_SPECIAL_EVENTS)

    out_melody = reconstruct_melody(model, device, melody, encode_event_fn, decode_event_fn)

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(211)
    pp.plot_pianoroll(ax, melody_lib.melody_to_pianoroll(melody))
    plt.title('Original melody')

    ax = fig.add_subplot(212)
    pp.plot_pianoroll(ax, melody_lib.melody_to_pianoroll(out_melody))
    plt.title('Reconstruction')

    if out_dirname is not None:
        plot_basename = "recon_{}_{}_{}.png".format(Path(midi_path).stem, melody_idx, start_bar)
        fig.savefig(str(out_dirname / plot_basename))
    plt.show()
    plt.close()


def interpolate(model, device, start_melody_info, end_melody_info, num_steps, out_dirname=None, to_midi=False):
    """Interpolates from start melody to end melody using (num_steps - 2) in between.

    x_melody_info contains is the tuple (midi_path, melody_idx, start_bar)
    for start respectively end melody."""
    def _slerp(p0, p1, t):
        """Spherical linear interpolation."""
        omega = np.arccos(np.dot(np.squeeze(p0/np.linalg.norm(p0)),
                                 np.squeeze(p1/np.linalg.norm(p1))))
        so = np.sin(omega)
        return np.sin((1.0-t)*omega) / so * p0 + np.sin(t*omega)/so * p1

    start_melody, (midi_path1, melody_idx1, start_bar1) = get_melody_from_info(start_melody_info)
    end_melody, (midi_path2, melody_idx2, start_bar2) = get_melody_from_info(end_melody_info)

    model.eval()
    with torch.no_grad():
        encode_event_fn = get_event_encoder(MIN_PITCH, MAX_PITCH, NUM_SPECIAL_EVENTS)
        decode_event_fn = get_event_decoder(MIN_PITCH, MAX_PITCH, NUM_SPECIAL_EVENTS)

        start_sequence = melody_to_sequence(start_melody, encode_event_fn)
        end_sequence = melody_to_sequence(end_melody, encode_event_fn)
        input_sequences = torch.stack((start_sequence, end_sequence)).to(device)

        mu, sigma = model.encode(input_sequences)
        z = model.reparameterize(mu, sigma)
        z = z.cpu()  # this needs to be done in order to interpolate the way i do it, maybe there is a better way
        interpolated_z = torch.stack([_slerp(z[0], z[1], t) for t in np.linspace(0, 1, num_steps)]).to(device)

        output_sequences, _ = model.decode(interpolated_z, sequence_length=input_sequences.size(1))
        out_melody = sequence_to_melody(output_sequences.view(-1, output_sequences.size(-1)), decode_event_fn)

        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111)
        pp.plot_pianoroll(ax, melody_lib.melody_to_pianoroll(out_melody))
        ax.set_xticks(np.arange(MELODY_LENGTH, out_melody.shape[0], MELODY_LENGTH))
        plt.title("Interpolate like there is no tomorrow, but be assured there always is.")

        if out_dirname is not None:
            fileroot = "interpolate_{}_{}_{}_to_{}_{}_{}.png".format(
                Path(midi_path1).stem, melody_idx1, start_bar1,
                Path(midi_path2).stem, melody_idx2, start_bar2)

            if to_midi:
                pm = melody_lib.melody_to_midi(out_melody)
                pm.write(str(out_dirname / (fileroot + ".mid")))

            fig.savefig(str(out_dirname / (fileroot + ".png")))
        plt.show()
        plt.close()


def sample_cmd(model, device, sequence_length, seed_melody_info=None, out_dirname=None, to_midi=False):

    model.eval()
    with torch.no_grad():
        encode_event_fn = get_event_encoder(MIN_PITCH, MAX_PITCH, NUM_SPECIAL_EVENTS)
        decode_event_fn = get_event_decoder(MIN_PITCH, MAX_PITCH, NUM_SPECIAL_EVENTS)

        if seed_melody_info is not None:
            seed_melody, (midi_path, melody_idx, start_bar) = get_melody_from_info(seed_melody_info)
            input_sequence = melody_to_sequence(seed_melody, encode_event_fn).to(device)

            mu, sigma = model.encode(input_sequence.unsqueeze(dim=0))
            z = model.reparameterize(mu, sigma)
        else:
            z = torch.randn((1, model.decoder.z_dim)).to(device)

        output_sequences, _ = model.decode(z, sequence_length=sequence_length)
        out_melody = sequence_to_melody(output_sequences.view(-1, output_sequences.size(-1)), decode_event_fn)

        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111)
        pp.plot_pianoroll(ax, melody_lib.melody_to_pianoroll(out_melody))
        ax.set_xticks(np.arange(MELODY_LENGTH, out_melody.shape[0], MELODY_LENGTH))
        plt.title("Sampeling")

        if out_dirname is not None:
            fileroot = "sample_len{}".format(sequence_length)
            if seed_melody_info is not None:
                fileroot += "_{}_{}_{}".format(Path(midi_path).stem, melody_idx, start_bar)

            if to_midi:
                pm = melody_lib.melody_to_midi(out_melody)
                pm.write(str(out_dirname / (fileroot + ".mid")))

            fig.savefig(str(out_dirname / (fileroot + ".png")))

        plt.show()
        plt.close()


def main():
    try:
        out_dirname = None
        if args.out_dirname is not None:
            out_dirname = create_or_empty_dir(args.out_dirname, force=args.force, delete=args.delete)

        if args.command == "list":
            list_melodies(args.midi_path, out_dirname=out_dirname)
        else:
            ckpt_path = args.ckpt_path or os.getenv("CKPT_PATH")
            if not isinstance(ckpt_path, str) or not Path(ckpt_path).is_file():
                print("Path to checkpoint is not a file. '{}'".format(ckpt_path))
                return

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print("Load model from checkpoint '{}'".format(ckpt_path))
            model = load_model_from_ckpt(ckpt_path, device)
            model.decoder.set_temperature(args.temperature)
            model.decoder.allow_teacher_forcing_for_evaluation(True)
            if args.teacher_forcing:
                print("Use teacher forcing")
                model.decoder.set_sampling_probability(0.0)
            else:
                print("Sample input from previous output.")
                model.decoder.set_sampling_probability(1.0)
            print("Temperature for sampling is {}".format(model.decoder.get_temperature()))

            if args.command == "reconstruct":
                reconstruct(model, device, args.melody_info, out_dirname=out_dirname)
            elif args.command == "interpolate":
                interpolate(model, device, args.start_melody_info, args.end_melody_info, args.num_steps,
                            out_dirname=out_dirname, to_midi=args.to_midi)
            elif args.command == "sample":
                sample_cmd(model, device, sequence_length=args.length, seed_melody_info=args.seed,
                           out_dirname=out_dirname, to_midi=args.to_midi)
            else:
                print("Unhandled command '{}'".format(args.command))
    except ExceedMelodyError as e:
        print("Exceeded end of melody, melody has {:d} bars, you want to extract {:d} bars starting from bar {:d}.\nPlease choose another start or melody."  # noqa
              .format(e.melody_length, e.window_length, e.start_bar))


if __name__ == "__main__":
    main()
