import pandas as pd
import numpy as np

import xiaoya
from xiaoya import pyehr  # training pipeline, models, etc
from xiaoya import data  # data handler, data loaders, etc
from xiaoya import pipeline # training and predict pipeline

print(xiaoya.__version__)

labtest_data = pd.read_csv('datasets/labtest_data.csv')
events_data = pd.read_csv('datasets/events_data.csv')
target_data = pd.read_csv('datasets/target_data.csv')

data_handler = data.DataHandler(labtest_data=labtest_data, events_data=events_data, target_data=target_data, save_processed_data=True)

features = data_handler.extract_features()
for key, feats in data_handler.extract_features().items():
    print(key, feats)

for info in data_handler.analyze_dataset():
    print(info)

# execute the preprocessing pipeline.
data_handler.execute()

demo_dim = len(features['events_features']) + 1
lab_dim =  len(features['labtest_features']) - 1
pipeline = pipeline.Pipeline(dataset=data_handler.merged_df, model='MLP', demographic_dim=demo_dim, labtest_dim=lab_dim)

# execute the training pipeline, returns the performance of the model.
performance = pipeline.execute()
print(performance)
