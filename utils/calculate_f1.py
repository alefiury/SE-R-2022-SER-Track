import argparse

import pandas as pd
from omegaconf import OmegaConf
from sklearn.metrics import f1_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        '--config_path',
        default='../config/default.yaml',
        type=str,
        help="YAML file with configurations"
    )
    parser.add_argument(
        '--pred_csv',
        required=True,
        help='Path to the csv where are the predictions'
    )
    parser.add_argument(
        '--ground_truth_csv',
        required=True,
        help='Path to the csv where are the ground truth'
    )

    args = parser.parse_args()
    cfg = OmegaConf.load(args.config_path)

    df_pred = pd.read_csv(args.pred_csv).sort_values(by=[cfg.metadata.audio_path_column])
    df_truth = pd.read_csv(args.ground_truth_csv).sort_values(by=[cfg.metadata.audio_path_column])

    print(f"F1-Score: {f1_score(y_true=df_truth[cfg.metadata.label_column].values, y_pred=df_pred[cfg.metadata.label_column].values, average='macro')}")

if __name__ == "__main__":
    main()