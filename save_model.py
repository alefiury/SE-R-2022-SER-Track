import os
import glob
import argparse

import tqdm
import pandas as pd
from sklearn import metrics
from omegaconf import OmegaConf
from transformers import (
    AutoModelForAudioClassification,
    AutoFeatureExtractor,
)
from torch.utils.data import DataLoader

from utils.utils import (
    DataColletorTest,
    preprocess_metadata,
    get_label_id, predict,
    save_conf_matrix
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        '--config_path',
        default='config/default.yaml',
        type=str,
        help="YAML file with configurations"
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_path)

    label2id = {
        "neutral": 0,
        "happy": 1,
        "sad": 2,
        "angry": 3,
        "fear": 4,
        "disgust": 5,
        "surprise": 6,
    }

    id2label = {
        0: "neutral",
        1: "happy",
        2: "sad",
        3: "angry",
        4: "fear",
        5: "disgust",
        6: "surprise",
    }

    feature_extractor = AutoFeatureExtractor.from_pretrained(cfg.test.model_checkpoint)
    model = AutoModelForAudioClassification.from_pretrained(
        pretrained_model_name_or_path="checkpoints/wav2vec2-xls-r-300m-pt-br-spontaneous-speech-emotion-recognition-multidaset-3.0-20_epochs/train",
        num_labels=7,
        label2id=label2id,
        id2label=id2label,
    )

    model.save_model("saved_models/wav2vec2-xls-r-300m-pt-br-spontaneous-speech-emotion-recognition-multidaset-3.0-20_epochs.pth")


if __name__ == '__main__':
    main()