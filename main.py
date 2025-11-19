from data_preprocessing import SmokingDataProcessor
from model_training import SmokingModelTrainer

# 1. Preprocessing Phase
# NOTE: Ensure 'smoking.csv' is in the same folder
processor = SmokingDataProcessor("smoking.csv")
processor.load_data()
processor.perform_eda_and_stats()
df_clean = processor.transform_data()

# 2. Modeling Phase
trainer = SmokingModelTrainer(df_clean)
trainer.split_data()
trainer.train_models()
trainer.evaluate_models()
trainer.plot_feature_importance()