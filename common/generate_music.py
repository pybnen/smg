import torch
import numpy as np
import pypianoroll as pp
from smg.datasets.lpd_5_cleansed import Instruments


def generate_pianoroll(model, stub, total_length, input_length=None, one_pitch_per_sample=False):
    if input_length is None:
        input_length = stub.size(1)
        
    # first generate sequence of total_length
    generated_seq = stub.clone()

    model.eval()
    with torch.no_grad():
        while generated_seq.size(1) < total_length:
            y_hat = model.forward(generated_seq)
            
            # add timestep to sequence
            generated_seq = torch.cat([generated_seq, y_hat], dim=1)
            
            # slide stub
            # stub = generated_seq[:, -input_length:, :, :]
        
    pianoroll = generated_seq.detach().cpu().view(-1, generated_seq.shape[-1])

    if one_pitch_per_sample:
        # generate a pianoroll with zeros except for the max values of the generated sequence
        max_p, argmax_p = pianoroll.max(dim=-1)

        pianoroll = torch.zeros_like(pianoroll)
        pianoroll[np.arange(pianoroll.size(0)), argmax_p.type(torch.LongTensor)] = max_p
        
    pianoroll = (pianoroll.view(generated_seq.size()).squeeze(0) * 127).type(torch.int).numpy()
    
    # pianoroll[np.logical_and(pianoroll > 1, pianoroll < 60)] += 60
    return np.clip(pianoroll, 0, 127)


def generate_multitrack(pianoroll, instruments, lowest_pitch, n_pitches, beat_resolution):
    instruments = [Instruments[inst.upper()] for inst in instruments]
    
    tempo = 120.0
    downbeat=None
    
    pianoroll = pianoroll.astype(np.uint8)
    assert(np.all(np.logical_and(pianoroll >= 0, pianoroll < 128)))
    
    full_pitch_pianoroll = np.zeros((pianoroll.shape[0], pianoroll.shape[1], 128))
    full_pitch_pianoroll[:, :, lowest_pitch:lowest_pitch + n_pitches] = pianoroll
    
    # create tracks
    tracks = []
    for i, instrument in enumerate(instruments):
        tracks.append(
            pp.Track(
                full_pitch_pianoroll[:, i],
                is_drum=instrument.is_drum(),
                program=instrument.midi_program(),
                name=instrument.name
            )
        )
        
    # create multitrack piano rolls
    return pp.Multitrack(
        tracks=tracks,
        tempo=tempo * (np.ones(full_pitch_pianoroll.shape[0])),
        downbeat=downbeat,
        beat_resolution=beat_resolution,
        name='generated'
    )