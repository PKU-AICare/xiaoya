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
