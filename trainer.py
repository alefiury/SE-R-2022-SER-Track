import os
import argparse

import pandas as pd
from omegaconf import OmegaConf
from audiomentations import Compose
from sklearn.model_selection import train_test_split
from transformers.trainer_utils import get_last_checkpoint
from transformers import (
                            AutoModelForAudioClassification,
                            EarlyStoppingCallback,
                            AutoFeatureExtractor,
                            TrainingArguments,
                            Trainer
)

from utils.utils import (
    DataColletorTrain, compute_metrics, preprocess_metadata, get_label_id, map_data_augmentation
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
    parser.add_argument(
        '--continue_train',
        default=False,
        action='store_true',
        help='If True, continues training using the checkpoint_path parameter'
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_path)

    if os.path.isdir(cfg.train.model_checkpoint):
        last_checkpoint = get_last_checkpoint(cfg.train.model_checkpoint)
        print("> Resuming Train with checkpoint: ", last_checkpoint)
    else:
        last_checkpoint = None

    if cfg.data.audio_augmentator and cfg.data.apply_augmentation:
        audio_augmentator = Compose([map_data_augmentation(aug_config) for aug_config in cfg.data.audio_augmentator])
    else:
        audio_augmentator = None

    train_df = pd.read_csv(cfg.metadata.train_path)

    if cfg.metadata.dev_path is None:
        train_df, val_df = train_test_split(
            train_df,
            test_size=0.2,
            random_state=42,
            stratify=train_df[[cfg.metadata.label_column]]
        )
    else:
        val_df = pd.read_csv(cfg.metadata.dev_path)

    train_df = train_df[train_df[cfg.metadata.label_column]!="frustrated"]
    train_df = train_df[train_df[cfg.metadata.label_column]!="excited"]

    val_df = val_df[val_df[cfg.metadata.label_column]!="frustrated"]
    val_df = val_df[val_df[cfg.metadata.label_column]!="excited"]

    train_dataset = preprocess_metadata(cfg=cfg, df=train_df)
    val_dataset = preprocess_metadata(cfg=cfg, df=val_df)

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

    feature_extractor = AutoFeatureExtractor.from_pretrained(cfg.train.model_checkpoint)
    model = AutoModelForAudioClassification.from_pretrained(
        pretrained_model_name_or_path=last_checkpoint if last_checkpoint else cfg.train.model_checkpoint,
        num_labels=7,
        label2id=label2id,
        id2label=id2label,
    )

    data_collator = DataColletorTrain(
        feature_extractor,
        apply_augmentation=cfg.data.apply_augmentation,
        audio_augmentator=audio_augmentator,
        sampling_rate=cfg.data.target_sampling_rate,
        padding=cfg.data.pad_audios,
        apply_dbfs_norm=cfg.data.apply_dbfs_norm,
        target_dbfs=cfg.data.target_dbfs,
        label2id=label2id,
        max_audio_len=cfg.data.max_audio_len
    )

    train_args = TrainingArguments(
        output_dir=cfg.train.weights_output_path,
        run_name=cfg.logging.run_name,
        report_to="all",
        save_strategy="epoch",
        evaluation_strategy="epoch",
        learning_rate=cfg.train.learning_rate,
        dataloader_num_workers=cfg.train.num_workers,
        per_device_train_batch_size=cfg.train.batch_size,
        per_device_eval_batch_size=cfg.train.batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        save_total_limit=cfg.train.save_total_limit,
        metric_for_best_model=cfg.train.metric,
        logging_steps=cfg.train.logging_steps,
        warmup_ratio=cfg.train.warmup_ratio,
        num_train_epochs=cfg.train.epochs,
        load_best_model_at_end=True,
        logging_first_step=True,
        greater_is_better=True,
        seed=cfg.train.seed,
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics
    )

    if cfg.train.use_early_stop:
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=cfg.train.early_stop_epochs))

    print("> Starting Training")
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint if args.continue_train else None)
    # Save best model
    trainer.save_model()

    # Save train results
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Save eval results
    print("--- Evaluate ---")
    metrics = trainer.evaluate()
    metrics["eval_samples"] = len(val_dataset)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

if __name__ == '__main__':
    main()