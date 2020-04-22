from glob import glob
import numpy as np
import time
import pretty_midi
from pathlib import Path

from smg.music.melody_lib import midi_to_melody

DATASET_DIR = "../data/lmd_full/"
MELODY_DIR = "../data/lmd_full_melody/"


def main():
    start = time.time()
    files = glob(DATASET_DIR + "/**/*.mid", recursive=True)
    print("[GLOB_FILES] {} files ({:.4f}s)".format(len(files), time.time() - start))

    start = time.time()
    faulty_files = []
    total_melody_cnt = 0
    prev_melody_cnt = 0

    for i, file in enumerate(files):
        try:
            pm = pretty_midi.PrettyMIDI(midi_file=file)
        except:
            faulty_files.append(file)
            continue

        try:
            melodies = midi_to_melody(file, pm, gap_bars=1.0)
            melodies = [m["events"] for m in melodies]

            sliced_melodies = []
            slice_steps = None  # 16*16

            if slice_steps:
                for m in melodies:
                    for i in range(slice_steps, len(m) + 1, 16):
                        melody_slice = m[i - slice_steps: i]
                        for k, event in enumerate(melody_slice):
                            if event not in [-1, -2]:
                                break
                            melody_slice[k] = -2

                        sliced_melodies.append(melody_slice)
                melodies = sliced_melodies

            melodies = list(set(tuple(m) for m in melodies))
            # melodies = melodies[:5]

        except Exception as ex:
            print("Error while processing {}: '{}'.".format(file, str(ex)))
            continue

        if len(melodies) > 0:
            file_path = Path(file)
            file_stem = file_path.stem
            subdir_name = file_path.parent.name

            melody_dir = Path(MELODY_DIR) / subdir_name
            melody_dir.mkdir(parents=True, exist_ok=True)

            for num_melody, melody in enumerate(melodies):
                total_melody_cnt += 1
                np.save(melody_dir / "{}_{:02d}.npy".format(file_stem, num_melody), melody)

        if (i + 1) % 100 == 0:
            delta = total_melody_cnt - prev_melody_cnt
            prev_melody_cnt = total_melody_cnt
            print("[{:.4f}s] {} file(s) processed, found {} melodies ({:+d})".format(time.time() - start,
                                                                                    i+1,
                                                                                    total_melody_cnt,
                                                                                    delta))

    print("[ITERATE_FILES] ({:.4f}s)".format(time.time() - start))
    print("Total music count: {}".format(total_melody_cnt))
    print("Could not read {} file(s):".format(len(faulty_files)))
    for faulty_file in faulty_files:
        print("- {}".format(faulty_file))


if __name__ == "__main__":
    main()