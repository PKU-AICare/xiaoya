import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

import xiaoya
from xiaoya import data  # data handler, data loaders, etc
from xiaoya import pipeline # training and predict pipeline
from xiaoya import analysis # data analyzer

print(xiaoya.__version__)

labtest_data = pd.read_csv('datasets/labtest_data.csv')
events_data = pd.read_csv('datasets/events_data.csv')
target_data = pd.read_csv('datasets/target_data.csv')

data_handler = data.DataHandler(labtest_data=labtest_data, events_data=events_data, target_data=target_data)
data_handler.format_dataframe('labtest')
data_handler.format_dataframe('events')
data_handler.format_dataframe('target')
data_handler.merge_dataframes()

labtest_feats = data_handler.extract_features('labtest')
events_feats = data_handler.extract_features('events')
target_feats = data_handler.extract_features('target')

detail = data_handler.analyze_dataset()['detail']
pd.to_pickle(detail, 'detail.pkl')
for info in detail:
    print(info)

# execute the preprocessing pipeline.
# data_handler.execute()

# demo_dim = 2  # len(features['events_features']) + 1     # +1 for the age feature
# lab_dim =  73 # len(features['labtest_features']) - 1    # -1 for the age feature
# pl = pipeline.Pipeline(model='MHAGRU', demographic_dim=demo_dim, labtest_dim=lab_dim)

# execute the training pipeline, returns the performance of the model.
# pl.train()
# model_path = '/home/akai/xiaoya/checkpoints/multitask/MHAGRU-seed42/best-v3.ckpt'
# performance = pl.predict(model_path)
# print(performance)

# analyzer = analysis.DataAnalyzer(pipeline=pl, model_path=model_path)
# train_raw = pd.read_csv('datasets/train_raw.csv')
# x = pd.read_pickle('datasets/train_x.pkl')
# patients = analyzer.data_dimension_reduction("PCA", 2, "multitask")
# print(patients)

# mask = pd.read_pickle('datasets/train_missing_mask.pkl')
# result = analyzer.risk_curve(train_raw, x, mask, 3)
# print(result)

# scores = analyzer.get_importance_scores(x)
# res=analyzer.data_dimension_reduction("PCA",2,"Outcome")
# print(res)
# for key, value in scores.items():
#     print(key, value)
