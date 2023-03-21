import os
import glob
import argparse

import pandas as pd
from omegaconf import OmegaConf
from transformers import (
    AutoModelForAudioClassification,
    AutoFeatureExtractor,
)
from torch.utils.data import DataLoader

from utils.utils import (
    DataColletorTest,
    preprocess_metadata,
    get_label_id, predict
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
    pred_labels = []
    new_paths = []

    os.makedirs(os.path.dirname(cfg.test.output_path), exist_ok=True)

    test_audio_paths = glob.glob(
        os.path.join(
            cfg.test.test_base_dir,
            "**",
            "*.wav"
        ),
        recursive=True
    )

    train_df = pd.read_csv(cfg.metadata.train_path)
    test_df = pd.DataFrame(test_audio_paths, columns =[cfg.metadata.audio_path_column])

    train_dataset = preprocess_metadata(cfg=cfg, df=train_df)
    test_dataset = preprocess_metadata(cfg=cfg, df=test_df)

    label2id, id2label, num_labels = get_label_id(
        dataset=train_dataset,
        label_column=cfg.metadata.label_column
    )

    feature_extractor = AutoFeatureExtractor.from_pretrained(cfg.test.model_checkpoint)
    model = AutoModelForAudioClassification.from_pretrained(
        pretrained_model_name_or_path=cfg.test.model_checkpoint,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )

    data_collator = DataColletorTest(
        processor=feature_extractor,
        sampling_rate=cfg.data.target_sampling_rate,
        padding=cfg.data.pad_audios,
        apply_dbfs_norm=cfg.data.apply_dbfs_norm,
        target_dbfs=cfg.data.target_dbfs,
        cfg=cfg
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=cfg.test.batch_size,
        collate_fn=data_collator,
        shuffle=False,
        num_workers=cfg.test.num_workers
    )

    preds, paths = predict(
        test_dataloader=test_dataloader,
        model=model,
        cfg=cfg
    )

    for pred in preds:
        pred_labels.append(id2label[str(pred)])

    for path in paths:
        new_paths.append(os.path.basename(path))

    df = pd.DataFrame(
        list(zip(new_paths, pred_labels)),
            columns = [
                cfg.metadata.audio_path_column,
                cfg.metadata.label_column
            ]
        )

    df.to_csv(cfg.test.output_path, index=False)

if __name__ == '__main__':
    main()