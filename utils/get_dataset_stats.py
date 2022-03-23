import os
import glob
import argparse
from collections import namedtuple

import tqdm
import torchaudio
import numpy as np


Info = namedtuple("Info", ["length", "sample_rate", "channels"])

def get_audio_info(path: str) -> namedtuple:
    """
    Get basic information related to number of frames,
    sample rate and number of channels.
    """

    info = torchaudio.info(path)
    if hasattr(info, 'num_frames'):
        return Info(info.num_frames, info.sample_rate, info.num_channels)
    else:
        siginfo = info[0]
        return Info(siginfo.length // siginfo.channels, siginfo.rate, siginfo.channels)


def get_total_dataset_length(base_dir: str) -> None:
    """
    Gets information related to the length of the audios
    and the amount of data in dataset itself
    """
    length = []
    srs = []
    channels = []

    file_paths = glob.glob(os.path.join(base_dir, '**', '*.wav'), recursive=True)

    for file_path in tqdm.tqdm(file_paths):
        audio_info = get_audio_info(file_path)
        # assert audio_info[1] == 16000
        # assert audio_info[2] == 1
        # assert audio_info[0] > 10
        srs.append(audio_info[1])
        channels.append(audio_info[2])
        length.append(audio_info[0]/audio_info[1])

    print(f"Min audio lenght (in seconds): {min(length)} | Max audio lenght (in seconds): {max(length)}")
    print(f"Mean audio length (in seconds): {np.mean(length)} | Median: {np.median(length)} | Std: {np.std(length)}")
    print(f"Total amount of data (in minutes): {np.sum(length)/60}")
    print('-'*50)
    print(f"Different samplerates in audios in the dataset: {set(srs)}")
    print(f"Different number of channels in the audios in the dataset: {set(channels)}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_base_dir',
        default='../data/train',
        type=str,
        help="Base directory where the audio data is stored"
    )
    args = parser.parse_args()

    print(get_total_dataset_length(args.data_base_dir))


if __name__ == '__main__':
    main()