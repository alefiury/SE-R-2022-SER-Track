import os
import glob
import argparse
from collections import namedtuple

import tqdm
import torch
import numpy as np
from pydub import AudioSegment

device = ('cuda' if torch.cuda.is_available() else 'cpu')
Info = namedtuple("Info", ["length", "sample_rate", "channels"])


def calculate_mean_dbfs(base_dir: str) -> float:
    file_paths = glob.glob(os.path.join(base_dir, '**', '*.wav'), recursive=True)

    dbfs_list = []
    for audio_file in tqdm.tqdm(file_paths):
        dbfs = AudioSegment.from_file(audio_file).dBFS
        dbfs_list.append(dbfs)

    target_dbfs = np.array(dbfs_list).mean()
    return target_dbfs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_base_dir',
        default='../data/train',
        type=str,
        help="Base directory where the audio data is stored"
    )
    args = parser.parse_args()

    print(calculate_mean_dbfs(args.data_base_dir))


if __name__ == '__main__':
    main()