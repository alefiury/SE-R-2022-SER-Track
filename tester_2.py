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

    metadata_paths = [
            # "../new_data_2.4/Multidataset-ser/metadata/metadata_aesdd_test.csv",
            # "../new_data_2.4/Multidataset-ser/metadata_normalized/metadata_emodb_test.csv",
            # "../new_data_2.4/Multidataset-ser/metadata/metadata_emovo_test.csv",
            # "../new_data_2.4/Multidataset-ser/metadata/metadata_iemocap_test.csv",
            # "../new_data_2.4/Multidataset-ser/metadata/metadata_urdu_test.csv",
            # "../new_data_2.4/Multidataset-ser/metadata/metadata_test.csv",
            # "../new_data_2.4/Multidataset-ser/metadata_normalized/coraa_ser.csv",
            # "../new_data_2.4/Multidataset-ser/metadata_normalized/ravdess_test.csv",
            # "../new_data_2.4/Multidataset-ser/metadata_normalized/ravdess_complete.csv",
            "../new_data_3.0/Metadata/metadata_test_audios_zap.csv",
            "../new_data_3.0/Metadata/metadata_test_coraa-ser.csv",
            "../new_data_3.0/Metadata/metadata_test_emoUERJ.csv",
            "../new_data_3.0/Metadata/metadata_test_ravdess.csv"
        ]

    dataset_names = [
        # "aesdd",
        # "emodb",
        # "emovo",
        # "iemocap",
        # "urdu",
        # "test",
        # "coraa_ser",
        # "ravdess_test",
        # "ravdess_complete",
        "audios_zap_3.0",
        "coraa_ser_3.0",
        "emoUERJ_3.0",
        "ravdess_test_3.0"
    ]

    base_dirs = [
        # "../new_data_2.4/Multidataset-ser",
        # "../new_data_2.4/Multidataset-ser",
        # "../new_data_2.4/Multidataset-ser",
        # "../new_data_2.4/Multidataset-ser",
        # "../new_data_2.4/Multidataset-ser",
        # "../new_data_2.4/Multidataset-ser",
        # "../data/coraa_ser/audios",
        # "../new_data_2.4/Multidataset-ser",
        # "../new_data_2.4/Multidataset-ser",
        "../new_data_3.0",
        "../new_data_3.0",
        "../new_data_3.0",
        "../new_data_3.0"
    ]

    label_column = cfg.metadata.label_column

    for metadata_path, base_dir, dataset_name in tqdm.tqdm(zip(metadata_paths, base_dirs, dataset_names)):
        print(f"Testing {dataset_name} ... \n")
        df = pd.read_csv(metadata_path)

        df = df[df[label_column]!="frustrated"]
        df = df[df[label_column]!="excited"]

        test_dataset = preprocess_metadata(base_dir=base_dir, cfg=cfg, df=df)

        data_collator = DataColletorTest(
            processor=feature_extractor,
            sampling_rate=cfg.data.target_sampling_rate,
            padding=cfg.data.pad_audios,
            label2id=label2id,
            max_audio_len=cfg.data.max_audio_len
        )

        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=cfg.test.batch_size,
            collate_fn=data_collator,
            shuffle=False,
            num_workers=cfg.test.num_workers
        )

        pred_list = predict(
            test_dataloader=test_dataloader,
            model=model,
            cfg=cfg
        )

        df = df.replace({label_column: label2id})

        labels = df[label_column].values

        save_conf_matrix(
            targets=labels,
            preds=pred_list,
            classes=[
                "neutral",
                "happy",
                "sad",
                "angry",
                "fear",
                "disgust",
                "surprise",
            ],
            output_path=f"../conf_m/wav2vec2-dataset_3_0-10_epochs_{dataset_name}.png"
        )

        # acc = metrics.accuracy_score(y_true=labels, y_pred=pred_list)
        # f1 = metrics.f1_score(y_true=labels, y_pred=pred_list, average='macro')
        # precision = metrics.precision_score(y_true=labels, y_pred=pred_list, average='macro')
        # recall = metrics.recall_score(y_true=labels, y_pred=pred_list, average='macro')

        # with open(os.path.join("../scores", 'scores_wav2vec2-dataset_3_0-20_epochs.txt'), 'a+') as file:
        #         file.write(f'{dataset_name} | '\
        #                         f'Accuracy: {acc} | '\
        #                         f'Precision: {precision} | '\
        #                         f'Recall: {recall} | '\
        #                         f'F1 Score: {f1}\n')


if __name__ == '__main__':
    main()