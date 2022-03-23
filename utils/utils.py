import os
import operator
import functools
import traceback
from typing import Dict, List, Optional, Union

import torch
import tqdm
import torch.nn.functional as F
import torchaudio
import numpy as np
import pandas as pd
from pydub import AudioSegment
from omegaconf import DictConfig
from datasets import Dataset, load_metric
from transformers import Wav2Vec2Processor
from audiomentations import AddGaussianNoise, PitchShift

metric = load_metric("f1")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_metrics(eval_pred):
    """Computes metric on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids, average="macro")


def map_path(batch, base_dir, cfg):
    """Maps the real path to the audio files"""
    batch['input_values'] = os.path.join(base_dir, batch[cfg.metadata.audio_path_column])
    return batch


def preprocess_metadata(cfg: DictConfig, df: pd.DataFrame):
    """Maps the real path to the audio files"""
    df.reset_index(drop=True, inplace=True)

    df_dataset = Dataset.from_pandas(df)
    df_dataset = df_dataset.map(
        map_path,
        fn_kwargs={"base_dir": cfg.data.base_dir, "cfg": cfg},
        num_proc=cfg.train.num_workers
    )

    return df_dataset


def get_label_id(dataset: Dataset, label_column: str):
    """Gets the labels IDs"""
    label2id, id2label = dict(), dict()

    labels = dataset.unique(label_column)
    labels.sort()

    num_labels = len(id2label)

    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    return label2id, id2label, num_labels


def predict(test_dataloader, model, cfg):
    model.to(device)
    model.eval()
    preds = []
    paths = []
    with torch.no_grad():
        for batch in tqdm.tqdm(test_dataloader):
            input_values, attention_mask = batch['input_values'].to(device), batch['attention_mask'].to(device)

            logits = model(input_values, attention_mask=attention_mask).logits

            scores = F.softmax(logits, dim=-1)
            pred = torch.argmax(scores, dim=1).cpu().detach().numpy()

            preds.append(list(pred))
            paths.append(list(batch[cfg.metadata.audio_path_column]))

    preds = functools.reduce(operator.iconcat, preds, [])
    paths = functools.reduce(operator.iconcat, paths, [])

    return preds, paths


def map_data_augmentation(aug_config):
    # Adapted from: https://github.com/Edresson/Wav2Vec-Wrapper
    aug_name = aug_config.name
    del aug_config.name
    if aug_name == 'gaussian':
        return AddGaussianNoise(**aug_config)
    elif aug_name == 'pitch_shift':
        return PitchShift(**aug_config)
    else:
        raise ValueError("The data augmentation '" + aug_name + "' doesn't exist !!")

class DataColletorTrain:
    # Adapted from https://github.com/Edresson/Wav2Vec-Wrapper
    def __init__(
        self,
        processor: Wav2Vec2Processor,
        apply_augmentation: bool = False,
        audio_augmentator: List[Dict] =  None,
        sampling_rate: int = 16000,
        padding: Union[bool, str] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        apply_dbfs_norm: Union[bool, str] = False,
        target_dbfs: int = 0.0,
        label2id: Dict = None
    ):

        self.processor = processor
        self.sampling_rate = sampling_rate

        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of

        self.apply_dbfs_norm = apply_dbfs_norm
        self.target_dbfs = target_dbfs

        self.apply_augmentation = apply_augmentation
        self.audio_augmentator = audio_augmentator

        self.label2id = label2id

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = []
        label_features = []
        for feature in features:
            try:
                # Gain Normalization
                if self.apply_dbfs_norm:
                    # Audio is loaded in a byte array
                    sound = AudioSegment.from_file(feature["input_values"], format="wav")
                    sound = sound.set_channels(1)
                    change_in_dBFS = self.target_dbfs - sound.dBFS
                    # Apply normalization
                    normalized_sound = sound.apply_gain(change_in_dBFS)
                    # Convert array of bytes back to array of samples in the range [-1, 1]
                    # This enables to work wih the audio without saving on disk
                    norm_audio_samples = np.array(normalized_sound.get_array_of_samples()).astype(np.float32, order='C') / 32768.0

                    if sound.channels < 2:
                        norm_audio_samples = np.expand_dims(norm_audio_samples, axis=0)

                    # Expand one dimension and convert to torch tensor to have the save output shape and type as torchaudio.load
                    speech_array = torch.from_numpy(norm_audio_samples)
                    sampling_rate = sound.frame_rate

                # Load wav
                else:
                    speech_array, sampling_rate = torchaudio.load(feature["input_values"])
                # Transform to Mono
                speech_array = torch.mean(speech_array, dim=0, keepdim=True)

                if sampling_rate != self.sampling_rate:
                    transform = torchaudio.transforms.Resample(sampling_rate, self.sampling_rate)
                    speech_array = transform(speech_array)
                    sampling_rate = self.sampling_rate

                speech_array = speech_array.squeeze().numpy()
                input_tensor = self.processor(speech_array, sampling_rate=sampling_rate).input_values
                input_tensor = np.squeeze(input_tensor)

                if self.audio_augmentator is not None and self.apply_augmentation:
                    input_tensor = self.audio_augmentator(input_tensor, sample_rate=self.sampling_rate).tolist()

                input_features.append({"input_values": input_tensor})
                label_features.append(int(self.label2id[feature["label"]]))
            except Exception:
                print("Error during load of audio:", feature["input_values"])
                continue

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch["labels"] = torch.tensor(label_features)

        return batch


class DataColletorTest:
    def __init__(
        self,
        processor: Wav2Vec2Processor,
        sampling_rate: int,
        padding: Union[bool, str],
        apply_dbfs_norm: Union[bool, str],
        target_dbfs: int,
        cfg: DictConfig,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ):

        self.processor = processor
        self.sampling_rate = sampling_rate

        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of

        self.apply_dbfs_norm = apply_dbfs_norm
        self.target_dbfs = target_dbfs
        self.cfg = cfg

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = []
        audio_paths = []
        for feature in features:
            try:
                # Gain Normalization
                if self.apply_dbfs_norm:
                    # Audio is loaded in a byte array
                    sound = AudioSegment.from_file(feature["input_values"])
                    sound = sound.set_channels(1)
                    change_in_dBFS = self.target_dbfs - sound.dBFS
                    # Apply normalization
                    normalized_sound = sound.apply_gain(change_in_dBFS)
                    # Convert array of bytes back to array of samples in the range [-1, 1]
                    # This enables to work wih the audio without saving on disk
                    norm_audio_samples = np.array(normalized_sound.get_array_of_samples()).astype(np.float32, order='C') / 32768.0

                    if sound.channels < 2:
                        norm_audio_samples = np.expand_dims(norm_audio_samples, axis=0)

                    # Expand one dimension and convert to torch tensor to have the save output shape and type as torchaudio.load
                    speech_array = torch.from_numpy(norm_audio_samples)
                    sampling_rate = sound.frame_rate

                # load wav
                else:
                    speech_array, sampling_rate = torchaudio.load(feature["input_values"])
                # Transform to Mono
                speech_array = torch.mean(speech_array, dim=0, keepdim=True)

                if sampling_rate != self.sampling_rate:
                    transform = torchaudio.transforms.Resample(sampling_rate, self.sampling_rate)
                    speech_array = transform(speech_array)
                    sampling_rate = self.sampling_rate

                speech_array = speech_array.squeeze().numpy()
                input_tensor = self.processor(speech_array, sampling_rate=sampling_rate).input_values
                input_tensor = np.squeeze(input_tensor)

                input_features.append({"input_values": input_tensor})
                audio_paths.append(feature[self.cfg.metadata.audio_path_column])
            except Exception:
                print("Error during load of audio:", feature["input_values"])
                continue

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch[self.cfg.metadata.audio_path_column] = audio_paths

        return batch