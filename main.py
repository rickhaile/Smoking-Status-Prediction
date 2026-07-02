import argparse

from data_preprocessing import SmokingDataProcessor
from model_training import SmokingModelTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Smoking status prediction pipeline: EDA, statistical tests, "
        "model training and evaluation."
    )
    parser.add_argument(
        "--data", default="smoking.csv",
        help="Path to the health screening dataset (default: smoking.csv)",
    )
    args = parser.parse_args()

    # 1. Preprocessing Phase
    processor = SmokingDataProcessor(args.data)
    processor.load_data()
    processor.perform_eda_and_stats()
    df_clean = processor.transform_data()

    # 2. Modeling Phase
    trainer = SmokingModelTrainer(df_clean)
    trainer.split_data()
    trainer.train_models()
    trainer.evaluate_models()
    trainer.plot_feature_importance()


if __name__ == "__main__":
    main()
