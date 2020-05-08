from glob import glob
from tqdm import tqdm
import numpy as np
import time
import copy
import sys
import pretty_midi
from pathlib import Path

# definition used by magenta (models\music_vae\data.py)
MEL_PROGRAMS = range(0, 32)  # piano, chromatic percussion, organ, guitar

STEPS_PER_QUARTER = 4
QUANTIZE_CUTOFF = 0.5

MELODY_NOTE_OFF = -1
MELODY_NO_EVENT = -2

MIN_MIDI_PITCH = 0


def create_melody(instrument):
    return {"events": [], "start_step": 0, "end_step": 0, "instrument": instrument}


def melody_set_length(melody, steps):
    old_len = len(melody["events"])

    if steps > old_len:
        melody["events"].extend([MELODY_NO_EVENT] * (steps - old_len))
    else:
        del melody["events"][steps:]

    melody["end_step"] = melody["start_step"] + steps

    if steps > old_len:
        # When extending the music on the right, we end any sustained notes.
        for i in reversed(range(old_len)):
            if melody["events"][i] == MELODY_NOTE_OFF:
                break
            elif melody["events"][i] != MELODY_NO_EVENT:
                melody["events"][old_len] = MELODY_NOTE_OFF
                break


def melody_get_last_on_off_events(melody):
    last_on = sys.maxsize
    length = len(melody["events"])
    last_off = length

    for idx, event in enumerate(reversed(melody["events"])):
        idx = length - 1 - idx

        if event == MELODY_NOTE_OFF:
            last_off = idx
        elif event != MELODY_NOTE_OFF and event != MELODY_NO_EVENT:
            last_on = idx
            break

    assert(last_on < last_off)
    return last_on, last_off


def melody_add_note(melody, pitch, start_step, end_step):
    assert(end_step > start_step)

    melody_set_length(melody, end_step + 1)

    melody["events"][start_step] = pitch
    melody["events"][end_step] = MELODY_NOTE_OFF

    for i in range(start_step + 1, end_step):
        melody["events"][i] = MELODY_NO_EVENT


def populate_melody(instrument,
                    quantized_notes,
                    steps_per_bar,
                    search_start_step,
                    ignore_polyphonic_notes,
                    gap_bars,
                    pad_end=True):
    melody = create_melody(instrument)

    notes = sorted([n for n in quantized_notes
                    if n["instrument"] == instrument and
                    n["quantized_start_step"] >= search_start_step],
                   key=lambda note: (note["quantized_start_step"], -note["pitch"]))
    if not notes:
        return None

    # music start at beginning of a bar
    first_start_step = notes[0]["quantized_start_step"]
    melody_start_step = first_start_step - \
                        (first_start_step - search_start_step) % steps_per_bar

    for note in notes:
        if note["is_drum"] or note["velocity"] == 0:
            continue

        start_index = note["quantized_start_step"] - melody_start_step
        end_index = note["quantized_end_step"] - melody_start_step

        if not melody["events"]:
            melody_add_note(melody, note["pitch"], start_index, end_index)
            continue

        # If `start_index` comes before or lands on an already added note's start
        # step, we cannot add it. In that case either discard the music or keep
        # the highest pitch.

        last_on, last_off = melody_get_last_on_off_events(melody)
        on_distance = start_index - last_on
        off_distance = start_index - last_off
        if on_distance == 0:
            if ignore_polyphonic_notes:
                # Keep highest note.
                # Notes are sorted by pitch descending, so if a note is already at
                # this position its the highest pitch.
                continue
            else:
                raise Exception()
        elif on_distance < 0:
            raise Exception()

        # If a gap of `gap` or more steps is found, end the music.
        if len(melody["events"]) and off_distance >= gap_bars * steps_per_bar:
            # print("gapomat")
            break

        # Add the note-on and off events to the music.
        melody_add_note(melody, note["pitch"], start_index, end_index)

    if not melody["events"]:
        return None
    melody["start_step"] = melody_start_step

    # Strip final MELODY_NOTE_OFF event.
    if melody["events"][-1] == MELODY_NOTE_OFF:
        del melody["events"][-1]

    length = len(melody["events"])
    # Optionally round up `_end_step` to a multiple of `steps_per_bar`.
    if pad_end:
        length += -length % steps_per_bar
    melody_set_length(melody, length)
    return melody


def extract_melodies(pm, quantized_notes,
                     min_bars=1,
                     gap_bars=1.0,
                     ignore_polyphonic_notes=True,
                     pad_end=True):

    steps_per_bar = int(steps_per_bar_in_quantized_notes(pm))
    instruments = set(n["instrument"] for n in quantized_notes)
    melodies = []
    for instrument in instruments:
        instrument_start_step = 0

        while True:
            melody = populate_melody(instrument,
                                     quantized_notes,
                                     steps_per_bar,
                                     instrument_start_step,
                                     ignore_polyphonic_notes,
                                     gap_bars)

            if melody is None:
                break

            # Start search for next music on next bar boundary (inclusive).
            instrument_start_step = melody["end_step"] + melody["end_step"] % steps_per_bar

            # Require a certain music length.
            if len(melody["events"]) < steps_per_bar * min_bars:
                continue
            melodies.append(melody)

    return melodies


def steps_per_bar_in_quantized_notes(pm):
    if len(pm.time_signature_changes) == 1:
        denominator = pm.time_signature_changes[0].denominator
        numerator = pm.time_signature_changes[0].numerator
    elif len(pm.time_signature_changes) == 0:
        denominator = numerator = 4
    else:
        assert(False)

    quarters_per_beat = 4.0 / denominator
    quarters_per_bar = quarters_per_beat * numerator

    steps_per_bar_float = quarters_per_bar * STEPS_PER_QUARTER
    assert(steps_per_bar_float == 16.0)
    return steps_per_bar_float


def steps_per_quarter_to_steps_per_second(steps_per_quarter, qpm):
    return qpm / 60.0 * steps_per_quarter


def quantize_to_step(unquantized_seconds, steps_per_second, quantize_cutoff=QUANTIZE_CUTOFF):
    unquantized_steps = unquantized_seconds * steps_per_second
    return int(unquantized_steps + (1 - quantize_cutoff))


def quantize_notes(quantized_notes, steps_per_second):
    for note in quantized_notes:
        # Quantize the start and end times of the note.
        note["quantized_start_step"] = quantize_to_step(note["start_time"], steps_per_second)
        note["quantized_end_step"] = quantize_to_step(note["end_time"], steps_per_second)

        if note["quantized_end_step"] == note["quantized_start_step"]:
            note["quantized_end_step"] += 1

        # Do not allow notes to start or end in negative time.
        if note["quantized_start_step"] < 0 or note["quantized_end_step"] < 0:
            raise Exception(
                'Got negative note time: start_step = %s, end_step = %s' %
                (note["quantized_start_step"], note["quantized_end_step"]))


def quantize_midi(pm, notes, steps_per_quarter):
    quantized_notes = copy.deepcopy(notes)

    assert(0 <= len(pm.time_signature_changes) <= 1)
    if len(pm.time_signature_changes) == 1:
        time_signature = pm.time_signature_changes[0]
        assert(time_signature.denominator == 4 and
               time_signature.numerator == 4 and
               time_signature.time == 0.0)

    # Compute quantization steps per second
    _, tempo_qpms = pm.get_tempo_changes()
    steps_per_second = steps_per_quarter_to_steps_per_second(steps_per_quarter, tempo_qpms[0])

    quantize_notes(quantized_notes, steps_per_second)
    return quantized_notes


def check_time_signature(filename, pm):
    #TODO allow multiple time signatures if they are all the same

    # From paper "A Hierarchical Latent Vector Model for Learning Long-Term Structure in Music"
    # Appendix A: We removed those that were identified as having a non-4/4 time signature
    n_time_signature_changes = len(pm.time_signature_changes)
    if n_time_signature_changes > 1:
        return False

    if n_time_signature_changes == 1:
        time_signature = pm.time_signature_changes[0]
        if not (time_signature.denominator == 4
                and time_signature.numerator == 4
                and time_signature.time == 0.0):
            return False

    # if no time signature is given it is assumed 4/4
    # see: http://midi.teragonaudio.com/tech/midifile/time.htm
    return True


def process_midi(filename, pm):

    if not check_time_signature(filename, pm):
        raise ValueError("Invalid time signature")

    tempo_times, tempo_qpms = pm.get_tempo_changes()
    #TODO allow multiple tempo times, and split sequence later
    if len(tempo_times) > 1:
        raise ValueError("Multiple tempi")

    notes = []
    total_time = None
    for num_instrument, midi_instrument in enumerate(pm.instruments):
        # Populate instrument name from the midi's instruments
        for midi_note in midi_instrument.notes:
            if not total_time or midi_note.end > total_time:
                total_time = midi_note.end

            note_dict = {"instrument": num_instrument,
                         "program": midi_instrument.program,
                         "start_time": midi_note.start,
                         "end_time": midi_note.end,
                         "pitch": midi_note.pitch,
                         "velocity": midi_note.velocity,
                         "is_drum": midi_instrument.is_drum}
            notes.append(note_dict)

        # ignore midi_instrument.pitch_bends
        # ignore midi_instrument.control_changes

    #TODO currently no need to split on tempo or time signature
    assert(len(tempo_qpms) == 1)
    assert(0 <= len(pm.time_signature_changes) <= 1)

    #TODO filter valid programs (i.e. music programs), and min/max pitch range

    return notes


def melody_to_midi(melody, velocity=100, program=0):
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # TODO add qpm/tempo and time signature, even though if those information are not contained in the midi file
    #   the default (which is what we want anyway) is assumed, never the less for clarity I would argue that
    #   including those information is not the worst thing that one can do
    notes = []
    time_per_step = 1/8.0

    pitch = MELODY_NOTE_OFF
    start_time = cur_time = 0.0

    assert(MIN_MIDI_PITCH > MELODY_NOTE_OFF > MELODY_NO_EVENT)

    # add MELODY_NOTE_OFF event at end of melody
    melody_copy = copy.copy(melody)
    if isinstance(melody_copy, (tuple, list)):
        melody_copy += (MELODY_NOTE_OFF,)
    elif isinstance(melody_copy, np.ndarray):
        melody_copy = np.concatenate((melody_copy, [MELODY_NOTE_OFF]))
    else:
        assert(False)

    for event in melody_copy:
        assert (event >= MELODY_NO_EVENT)

        if event >= MELODY_NOTE_OFF:
            # create previous note
            if pitch >= MIN_MIDI_PITCH:
                note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start_time, end=cur_time)
                notes.append(note)

            # update current note
            pitch = event
            start_time = cur_time

        cur_time += time_per_step

    instrument.notes = notes
    pm.instruments.append(instrument)

    return pm


def melody_to_pianoroll(melody, velocity=100):
    pianoroll = np.zeros((len(melody), 128))
    prev_event = -1
    for i, event in enumerate(melody):
        if event >= -1:
            if event >= 0:
                pianoroll[i, event] = velocity
            prev_event = event

        if event == -2 and prev_event >= 0:
            pianoroll[i, prev_event] = velocity
    return pianoroll


def midi_to_melody(file, pm, gap_bars=1.0):
    notes = process_midi(file, pm)
    quantized_notes = quantize_midi(pm, notes, STEPS_PER_QUARTER)
    melodies = extract_melodies(pm, quantized_notes, gap_bars=gap_bars)
    return melodies


if __name__ == "__main__":
    # test music to midi
    mel_files = glob("../data/lmd_full_melody/*.npy")
    for file in mel_files:
        melody = np.load(file)
        midi = melody_to_midi(melody)

        file_name = Path(file).stem
        midi.write("../data/lmd_full_melody_midi/{}.mid".format(file_name))