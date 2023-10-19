# Xiaoya-Core

xiaoya 2.0 core

## Project Structure

```bash
xiaoya/ # root
    pyehr/ # yhzhu99/pyehr project
    data/ # import user uploaded data, merge data tables, stats...
    pipeline/ # model training and evaluation, ...
    analysis/ # analysis modules
    plot/ # plot modules
```

## Sample Usages

### Pipeline of Training and Predicting

```python
from xiaoya.data import DataHandler
from xiaoya.pipeline import Pipeline

labtest_data = pd.read_csv('datasets/labtest_data.csv')
events_data = pd.read_csv('datasets/events_data.csv')
target_data = pd.read_csv('datasets/target_data.csv')
data_handler = DataHandler(labtest_data=labtest_data, events_data=events_data, target_data=target_data)
data_handler.execute()

pl = Pipeline()
result = pl.execute()
```

### Analysis and Plot

* Dataset Visualization

```python
from xiaoya.data import DataHandler
from xiaoya.plot import plot_vis_dataset

labtest_data = pd.read_csv('datasets/labtest_data.csv')
events_data = pd.read_csv('datasets/events_data.csv')
target_data = pd.read_csv('datasets/target_data.csv')
data_handler = DataHandler(labtest_data=labtest_data, events_data=events_data, target_data=target_data)
result = data_handler.analyze_dataset()
plot_vis_dataset(result['detail'], save_path='./output/')
```

* Plot Feature Importance histogram

```python
from xiaoya.pipeline import Pipeline
from xiaoya.analysis import DataAnalyzer
from xiaoya.plot import plot_feature_importance

pl = Pipeline(model='MHAGRU')
pl.execute()

data_analyzer = DataAnalyzer(pl.config, pl.model_path)
train_raw = pd.read_csv('datasets/train_raw.csv')
train_x = pd.read_pickle('datasets/train_x.pkl')
result = data_analyzer.feature_importance(
    df=train_raw,
    x=train_x,
    patient_index=0
)
plot_feature_importance(result['detail'], save_path='./output/')
```

* Plot Patient Risk curve

```python
from xiaoya.pipeline import Pipeline
from xiaoya.analysis import DataAnalyzer
from xiaoya.plot import plot_risk_curve

pl = Pipeline(model='MHAGRU')
pl.execute()

data_analyzer = DataAnalyzer(pl.config, pl.model_path)
train_raw = pd.read_csv('datasets/train_raw.csv')
train_x = pd.read_pickle('datasets/train_x.pkl')
train_mask = pd.read_pickle('datasets/train_missing_mask.pkl')
result = data_analyzer.risk_curve(
    df=train_raw,
    x=train_x,
    mask=train_mask,
    patient_index=0
)
plot_risk_curve(result, save_path='./output/')
```

* Plot Patient Embedding and Trajectory

```python
from xiaoya.pipeline import Pipeline
from xiaoya.analysis import DataAnalyzer
from xiaoya.plot import plot_patient_embedding

pl = Pipeline(model='MHAGRU')
pl.execute()

data_analyzer = DataAnalyzer(pl.config, pl.model_path)
train_x = pd.read_pickle('datasets/train_x.pkl')
train_pid = pd.read_pickle('datasets/train_pid.pkl')
train_record_time = pd.read_pickle('datasets/train_record_time.pkl')
train_mean_age = pd.read_pickle('datasets/train_mean.pkl')['Age']
train_std_age = pd.read_pickle('datasets/train_std.pkl')['Age']
result = data_analyzer.data_dimension_reduction(
    x = train_x,
    pid = train_pid,
    record_time = train_record_time,
    mean_age = train_mean_age,
    std_age = train_std_age
)
plot_patient_embedding(result['detail'], save_path='./output/')
```

* AI Advice

```python

```
