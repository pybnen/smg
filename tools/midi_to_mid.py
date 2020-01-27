import argparse
from glob import glob
from pathlib import Path
import os

def main():
    parser = argparse.ArgumentParser(description="rename file extension to mid")
    parser.add_argument("--dir", help="directory")
    args = parser.parse_args()

    assert(args.dir)

    input_dir = Path(args.dir)
    mask = str(input_dir / "**/*.midi")
    
    filenames = glob(mask, recursive=True)
    for i, filename in enumerate(filenames):
        if i % 1000 == 0:
            print(f"Rename {filename}...")
        filename = Path(filename)
        os.rename(filename, filename.with_suffix(".mid"))
        

if __name__ == '__main__':
    main()