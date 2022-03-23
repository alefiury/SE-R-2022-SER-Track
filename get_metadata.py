import os
import re
import glob
import argparse
from collections import namedtuple

import shutil
import torchaudio
import pandas as pd
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split

Info = namedtuple("Info", ["length", "sample_rate", "channels"])

def get_audio_info(path: str) -> namedtuple:
    """
    Get basic information related to number of frames,
    sample rate and number of channels.

    Params:
        path: Audio path.

    Returns:
        Tuple
    """

    info = torchaudio.info(path)
    if hasattr(info, 'num_frames'):
        return Info(info.num_frames, info.sample_rate, info.num_channels)
    else:
        siginfo = info[0]
        return Info(siginfo.length // siginfo.channels, siginfo.rate, siginfo.channels)

def coraa_dataset(audio_paths, data_base_dir):
    labels = []

    os.makedirs(os.path.join(data_base_dir, 'dev'), exist_ok=True)

    audio_paths = glob.glob(os.path.join(audio_paths, '**', '*.wav'), recursive=True)

    for audio_path in audio_paths:
        labels.append(audio_path.replace('.wav', '').split('_')[-1])

    df = pd.DataFrame(list(zip(audio_paths, labels)), columns =['wav_file', 'label'])

    train_df, dev_df = train_test_split(
        df,
        stratify=df['label'],
        test_size=0.2,
        random_state=42
    )

    for dev_audio_path in dev_df['wav_file'].values:
        shutil.copy2(dev_audio_path, os.path.join(data_base_dir, 'dev', os.path.basename(dev_audio_path)))

    return train_df, dev_df


def baved_dataset(audio_paths):
    genders = []
    emotions = []
    labels = []

    for idx, audio_path in enumerate(audio_paths):
        gender = os.path.basename(audio_path).split('-')[1]
        emotion = os.path.basename(audio_path).split('-')[4]
        if emotion == '1':
            emotion = 'neutral'
        else:
            emotion = 'non-neutral'

        if gender=='m':
            gender = 'male'
        if gender == 'f':
            gender = 'female'

        if emotion == 'neutral':
            label = 'neutral'
        else:
            label = emotion+'-'+gender

        genders.append(gender)
        emotions.append(emotion)
        labels.append(label)

    df = pd.DataFrame(list(zip(audio_paths, genders, emotions, labels)), columns =['wav_file', 'gender', 'emotion', 'label'])

    return df

def emovo_dataset(audio_paths):
    genders = []
    emotions = []
    labels = []

    for idx, audio_path in enumerate(audio_paths):
        gender = os.path.basename(audio_path).split('-')[1]
        emotion = os.path.basename(audio_path).split('-')[0]
        if emotion == 'neu':
            emotion = 'neutral'
        else:
            emotion = 'non-neutral'

        if gender.startswith('m'):
            gender = 'male'
        if gender.startswith('f'):
            gender = 'female'

        if emotion == 'neutral':
            label = 'neutral'
        else:
            label = emotion+'-'+gender

        genders.append(gender)
        emotions.append(emotion)
        labels.append(label)

    df = pd.DataFrame(list(zip(audio_paths, genders, emotions, labels)), columns =['wav_file', 'gender', 'emotion', 'label'])

    return df

def ravdess_dataset(audio_paths):
    genders = []
    emotions = []
    labels = []

    for idx, audio_path in enumerate(audio_paths):
        gender = os.path.basename(audio_path).split('-')[-1].split('.')[0]
        emotion = os.path.basename(audio_path).split('-')[2]

        if emotion == '01':
            emotion = 'neutral'
        else:
            emotion = 'non-neutral'

        if int(gender)%2 == 0:
            gender = 'female'
        else:
            gender = 'male'

        if emotion == 'neutral':
            label = 'neutral'
        else:
            label = emotion+'-'+gender

        genders.append(gender)
        emotions.append(emotion)
        labels.append(label)

    df = pd.DataFrame(list(zip(audio_paths, genders, emotions, labels)), columns =['wav_file', 'gender', 'emotion', 'label'])

    return df


def iemocap_dataset(audio_paths, ie_base_dir):
    iemocap_datas = []
    genders = []
    emotions = []
    labels = []
    paths = []

    for audio_path in audio_paths:
        audio_info = get_audio_info(audio_path)
        audio_length = audio_info[0]/audio_info[1]
        emotion_dict = {}
        file_name = os.path.basename(audio_path).split('.')[0]
        session_path = "Session"+str(int(file_name.split('_')[0][3:5]))

        gender = file_name.split('_')[-1][0]

        txt_name = os.path.basename(os.path.dirname(audio_path))
        txt_path = os.path.join(ie_base_dir, session_path, 'dialog', 'EmoEvaluation', txt_name+'.txt')

        assert os.path.isfile(txt_path)

        with open(txt_path, 'r') as file:
            string = file.read()
            results = re.findall("\[.+\]\t(.+)\t(.+)\t\[.+\]", string)

        for emo in results:
            emotion_dict[emo[0]] = emo[1]

        # Select good labels and filter by length
        if emotion_dict[file_name]!="xxx" and audio_length<=15.0:
            iemocap_datas.append((audio_path, emotion_dict[file_name], gender))

    for iemocap_data in iemocap_datas:
        gender = iemocap_data[-1]
        emotion = iemocap_data[1]
        path = iemocap_data[0]

        if emotion == 'neu':
            emotion = 'neutral'
        else:
            emotion = 'non-neutral'

        if gender == 'F':
            gender = 'female'
        if gender == 'M':
            gender = 'male'

        if emotion == 'neutral':
            label = 'neutral'
        else:
            label = emotion+'-'+gender

        genders.append(gender)
        emotions.append(emotion)
        labels.append(label)
        paths.append(path)

    df = pd.DataFrame(list(zip(paths, genders, emotions, labels)), columns =['wav_file', 'gender', 'emotion', 'label'])

    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_base_dir',
        default='data',
        help='Path to save metadata'
    )
    parser.add_argument(
        '-mtd', '--metada_train_data_output_data',
        default='metadata/ser_train_multiple_languages.csv',
        help='Path to save metadata'
    )
    parser.add_argument(
        '-mdd', '--metada_dev_data_output_data',
        default='metadata/ser_dev_coraa.csv',
        help='Path to save metadata'
    )

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.metada_train_data_output_data), exist_ok=True)
    os.makedirs(os.path.dirname(args.metada_dev_data_output_data), exist_ok=True)

    coraa_ser_base_dir = os.path.join(args.data_base_dir, 'train', 'CORAA_SER')
    baved_base_dir = os.path.join(args.data_base_dir, 'train', 'BAVED')
    emovo_base_dir = os.path.join(args.data_base_dir, 'train', 'EMOVO')
    ravdess_base_dir = os.path.join(args.data_base_dir, 'train', 'RAVDESS')


    baved_audio_paths = glob.glob(os.path.join(baved_base_dir, '**', '*.wav'), recursive=True)
    emovo_audio_paths = glob.glob(os.path.join(emovo_base_dir, '**', '*.wav'), recursive=True)
    ravdess_audio_paths = glob.glob(os.path.join(ravdess_base_dir, '**', '*.wav'), recursive=True)
    coraa_train_df, coraa_dev_df = coraa_dataset(coraa_ser_base_dir, args.data_base_dir)

    baved_df = baved_dataset(baved_audio_paths)
    emovo_df = emovo_dataset(emovo_audio_paths)
    ravdess_df = ravdess_dataset(ravdess_audio_paths)

    s_train = pd.concat(
        [
            coraa_train_df[['wav_file', 'label']],
            baved_df[['wav_file', 'label']],
            emovo_df[['wav_file', 'label']],
            ravdess_df[['wav_file', 'label']]
        ]
    )

    assert s_train.shape[0] == coraa_train_df.shape[0] + baved_df.shape[0] + emovo_df.shape[0] + ravdess_df.shape[0]

    s_train.to_csv(args.metada_train_data_output_data, index=False)
    coraa_dev_df.to_csv(args.metada_dev_data_output_data, index=False)

if __name__ == '__main__':
    main()