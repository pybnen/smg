from pathlib import Path
import argparse

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from smg.datasets.melody_dataset import FixedLengthMelodyDataset, MelodyEncode
from smg.ingredients.data import dataset_train_valid_split
from smg.models.music_vae.music_vae import MusicVAE


parser = argparse.ArgumentParser(description="Evaluate checkpoint with a given dataset")
# TODO better way to get this value, should be in model and dataset
parser.add_argument("--n_classes", type=int, default=90)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--ckpt_path", type=str,
                    help="Path to checkpoint, if not set environment variable CKPT_PATH is used.")
parser.add_argument("--temperature", type=float, default=1.0,
                    help="Used for sampling next input from current output, lower temperature favours best guess, very low temp would basically result in argmax.")  # noqa
parser.add_argument("--teacher_forcing", "-t", action="store_true", default=False,
                    help="Use teacher forcing")
parser.add_argument("--valid_split", type=float, default=0.2, help="Split ratio between train/eval set, set to 0.0 if not split should be made.")  # noqa
parser.add_argument("--use_train", action="store_true", default=False, help="Use train set instead of eval set, applys only if valid split is given, default False")  # noqa
parser.add_argument("dataset_dirname", type=str, help="Directory name containing dataset")

args = parser.parse_args()


def load_model_from_ckpt(ckpt_path, device):
    with open(ckpt_path, 'rb') as file:
        ckpt = torch.load(file, map_location=device)
        model_ckpt = ckpt['model']
    return MusicVAE.load_from_ckpt(model_ckpt).to(device)


def _log_metrics(total_loss, total_acc, sample_acc, cnt, print_header):
    if print_header:
        print("   i | loss     | acc      | sample_acc")
        print("-----+----------+----------+-----------")
    print("{:4d} | {:.6f} | {:.6f} | {:.6f}".format(
        cnt, total_loss / cnt, total_acc / cnt, sample_acc / cnt))


def evaluation_step(model, input_sequences):
    with torch.no_grad():
        batch_size, _, n_classes = input_sequences.size()
        output_sequences, _, _, _, _= model.forward(input_sequences)

        inputs_argmax = input_sequences.argmax(dim=-1)
        loss = F.cross_entropy(output_sequences.view(-1, n_classes), inputs_argmax.view(-1), reduction='mean')
        # multiply with batch_size, because want sum of batch as loss
        cross_entropy_loss = loss.item() * batch_size
        acc_per_steps = output_sequences.argmax(dim=-1) == inputs_argmax
        # noinspection PyUnresolvedReferences
        acc = torch.sum(torch.mean(acc_per_steps.float(), dim=-1)).item()
        # noinspection PyTypeChecker
        acc_per_seq = torch.sum(torch.all(acc_per_steps, dim=-1).float()).item()

    return output_sequences, cross_entropy_loss, acc, acc_per_seq


def evaluate(model, device, dataset=None, data_loader=None, log_interval=None):
    assert data_loader is not None or dataset is not None

    if dataset is not None:
        def dataset_gen(ds):
            for input_sequence in ds:
                yield torch.tensor(input_sequence).unsqueeze(dim=0)
        generator = dataset_gen(dataset)
    else:
        generator = iter(data_loader)

    total_loss = 0.0
    total_acc = 0.0
    total_acc_per_seq = 0.0
    total_sequences = 0
    print_header = True

    model.eval()
    for i, input_sequences in enumerate(generator):
        input_sequences = input_sequences.to(device)
        # NOTE: batch_size can be different for last batch (drop_last=False)
        batch_size = input_sequences.size(0)

        output_sequences, loss, acc, acc_per_seq = evaluation_step(model, input_sequences)

        total_loss += loss
        total_acc += acc
        total_acc_per_seq += acc_per_seq
        total_sequences += batch_size

        if log_interval is not None and (i + 1) % log_interval == 0:
            _log_metrics(total_loss, total_acc, total_acc_per_seq, total_sequences, print_header)
            print_header = False

    _log_metrics(total_loss, total_acc, total_acc_per_seq, total_sequences, print_header)
    return total_loss / total_sequences, total_acc / total_sequences, total_acc_per_seq / total_sequences


def main():
    ckpt_path = args.ckpt_path or os.getenv("CKPT_PATH")
    if not isinstance(ckpt_path, str) or not Path(ckpt_path).is_file():
        print("Path to checkpoint is not a file. '{}'".format(ckpt_path))
        return

    if not Path(args.dataset_dirname).is_dir():
        print("Dataset directory name is no directory. '{}'".format(args.dataset_dirname))
        return

    # TODO find better way to retrieve info about n_classes and num_special_events
    dataset = FixedLengthMelodyDataset(melody_dir=args.dataset_dirname, transforms=MelodyEncode(args.n_classes,
                                                                                                num_special_events=0))
    if args.valid_split > 0.0:
        ds_train, dataset = dataset_train_valid_split(dataset, valid_split=args.valid_split)
        if args.use_train:
            dataset = ds_train
    data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                             shuffle=False, drop_last=False)

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

    log_interval = len(data_loader) // 10
    if log_interval <= 0:
        log_interval = 1

    _ = evaluate(model, device, data_loader=data_loader, log_interval=log_interval)


if __name__ == "__main__":
    main()
