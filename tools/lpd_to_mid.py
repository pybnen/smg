import pypianoroll as pp
import argparse
from glob import glob
from pathlib import Path
import os

def main():
    parser = argparse.ArgumentParser(description="Converts the LPD 5 dataset to midi files")
    parser.add_argument("--input_dir", help="dataset directory")
    parser.add_argument("--output_dir", help="output directory")
    args = parser.parse_args()

    assert(args.input_dir and args.output_dir)

    input_dir = Path(args.input_dir)
    mask = str(input_dir / "**/*.npz")
    ouput_dir = Path(args.output_dir)

    filenames = glob(mask, recursive=True)
    for filename in filenames:
        print(f"Converting {filename}...")

        filename = Path(filename)
        file_output_dir = ouput_dir / Path(os.path.relpath(filename, input_dir)).parent

        if not file_output_dir.is_dir():
            file_output_dir.mkdir(parents=True)
    
        sample = pp.load(str(filename))
        pp.write(sample, str(file_output_dir / (filename.stem + ".mid")))


if __name__ == '__main__':
    main()