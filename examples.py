import pandas as pd
import torch

import xiaoya
from xiaoya import data  # data handler, data loaders, etc
from xiaoya import pipeline # training and predict pipeline
from xiaoya import analysis # data analyzer

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

demo_dim = len(features['events_features']) + 1     # +1 for the age feature
lab_dim =  len(features['labtest_features']) - 1    # -1 for the age feature
pl = pipeline.Pipeline(model='MHAGRU', demographic_dim=demo_dim, labtest_dim=lab_dim)

# execute the training pipeline, returns the performance of the model.
pl.train()
performance = pl.predict(pl.model_path)
print(performance)

analyzer = analysis.DataAnalyzer(pipeline=pl, model_path=pl.model_path)
x = pd.read_pickle('datasets/train_x.pkl')
x = torch.tensor(x[0]).unsqueeze(0)
print(analyzer.feature_advice(x, -1))

scores = analyzer.get_importance_scores(x)
res=analyzer.data_dimension_reduction(x,"PCA",2,"Outcome")
print(res)
for key, value in scores.items():
    print(key, value)
