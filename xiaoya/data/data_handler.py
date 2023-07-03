import os
from typing import List, Dict
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from .raw_files_utils import (
    get_features,
    to_dataframe,
    merge_dfs
)
from .processed_datasets_utils import (
    normalize_dataframe,
    forward_fill_pipeline,
)


class DataHandler:
    """
    Import user uploaded data, merge data tables, stats...

    Args:
        labtest_data: DataFrame.
        events_data: DataFrame.
        target_data: DataFrame.
        data_path: str.
            path to save processed data, default: './datasets'.

    """

    def __init__(
            self, 
            labtest_data: pd.DataFrame,
            events_data: pd.DataFrame,
            target_data: pd.DataFrame,
            data_path: str = Path('./datasets'),
        ) -> None:

        self.labtest_df = pd.DataFrame(labtest_data)
        self.events_df = pd.DataFrame(events_data)
        self.target_df = pd.DataFrame(target_data)
        self.labtest_standard_df = to_dataframe(self.labtest_df, 1)
        self.events_standard_df = to_dataframe(self.events_df, 2)
        self.target_standard_df = to_dataframe(self.target_df, 3)
        self.merged_df = merge_dfs(
            self.labtest_standard_df,
            self.events_standard_df,
            self.target_standard_df
        )
        self.data_path = data_path

    def extract_features(self) -> Dict:
        feats = {}
        feats['labtest_features'] = get_features(self.labtest_df, 1)
        self.labtest_features = feats['labtest_features']
        feats['events_features'] = get_features(self.events_df, 2)
        self.events_features = feats['events_features']
        feats['target_features'] = get_features(self.target_df, 3)
        self.target_features = feats['target_features']
        return feats

    def analyze_dataset(self) -> List:
        len_df = len(self.merged_df.index)
        header = [
            {"key": "name", "value": "name"},
            {"key": "count", "value": "count"},
            {"key": "missing", "value": "missing"},
            {"key": "min", "value": "min"},
            {"key": "max", "value": "max"},
            {"key": "mean", "value": "mean"},
            {"key": "std", "value": "std"},
            {"key": "median", "value": "median"},
        ]
        statistic_info = []
        for idx, e in enumerate(self.merged_df.columns):
            if idx == 1:
                continue
            h = {}
            h["id"] = idx
            h["name"] = e
            h["count"] = int(self.merged_df[e].count())
            h["missing"] = str(round(float((100 - self.merged_df[e].count() * 100 / len_df)), 2)) + "%"
            h["mean"] = round(float(self.merged_df[e].mean()), 2)
            h["max"] = round(float(self.merged_df[e].max()), 2)
            h["min"] = round(float(self.merged_df[e].min()), 2)
            h["median"] = round(float(self.merged_df[e].median()), 2)
            h["std"] = round(float(self.merged_df[e].std()), 2)
            statistic_info.append(h)
        return statistic_info
    
    def execute(self,
        train: int = 70,
        val: int = 10,
        test: int = 20,
        seed: int = 42,
    ) -> None:
        """
        
        """
        
        data_path = self.data_path
        demographic_features = self.events_features
        labtest_features = self.labtest_features

        # Group the dataframe by patient ID
        grouped = self.merged_df.groupby('PatientID')
        patients = np.array(list(grouped.groups.keys()))
        
        # Get the train_val/test patient IDs
        patients_outcome = np.array([grouped.get_group(patient_id)['Outcome'].iloc[0] for patient_id in patients])
        train_val_patients, test_patients = train_test_split(patients, test_size=test/(train+val+test), random_state=seed, stratify=patients_outcome)

        # Get the train/val patient IDs
        train_val_patients_outcome = np.array([grouped.get_group(patient_id)['Outcome'].iloc[0] for patient_id in train_val_patients])
        train_patients, val_patients = train_test_split(train_val_patients, test_size=val/(train+val), random_state=seed, stratify=train_val_patients_outcome)

        #  Create train, val, test, [traincal, calib] dataframes for the current fold
        train_raw_df = self.merged_df[self.merged_df['PatientID'].isin(train_patients)]
        val_raw_df = self.merged_df[self.merged_df['PatientID'].isin(val_patients)]
        test_raw_df = self.merged_df[self.merged_df['PatientID'].isin(test_patients)]

        # Calculate the mean and std of the train set (include age, lab test features, and LOS) on the data in 5% to 95% quantile range
        normalize_features = ['Age'] + labtest_features + ['LOS']
        train_after_zscore, val_after_zscore, test_after_zscore, default_fill, los_info, _, _ = \
            normalize_dataframe(train_raw_df, val_raw_df, test_raw_df, normalize_features)

        # Drop rows if all features are recorded NaN
        train_after_zscore = train_after_zscore.dropna(axis=0, how='all', subset=normalize_features)
        val_after_zscore = val_after_zscore.dropna(axis=0, how='all', subset=normalize_features)
        test_after_zscore = test_after_zscore.dropna(axis=0, how='all', subset=normalize_features)

        # Forward Imputation after grouped by PatientID
        train_x, train_y, train_pid = forward_fill_pipeline(train_after_zscore, default_fill, demographic_features, labtest_features)
        val_x, val_y, val_pid = forward_fill_pipeline(val_after_zscore, default_fill, demographic_features, labtest_features)
        test_x, test_y, test_pid = forward_fill_pipeline(test_after_zscore, default_fill, demographic_features, labtest_features)

        # Save the dataframes
        train_raw_df.to_csv(os.path.join(data_path, "train_raw.csv"), index=False)
        val_raw_df.to_csv(os.path.join(data_path, "val_raw.csv"), index=False)
        test_raw_df.to_csv(os.path.join(data_path, "test_raw.csv"), index=False)

        train_after_zscore.to_csv(os.path.join(data_path, "train_after_zscore.csv"), index=False)
        val_after_zscore.to_csv(os.path.join(data_path, "val_after_zscore.csv"), index=False)
        test_after_zscore.to_csv(os.path.join(data_path, "test_after_zscore.csv"), index=False)

        pd.to_pickle(train_x, os.path.join(data_path, "train_x.pkl"))
        pd.to_pickle(train_y, os.path.join(data_path, "train_y.pkl"))
        pd.to_pickle(train_pid, os.path.join(data_path, "train_pid.pkl"))
        pd.to_pickle(val_x, os.path.join(data_path, "val_x.pkl"))
        pd.to_pickle(val_y, os.path.join(data_path, "val_y.pkl"))
        pd.to_pickle(val_pid, os.path.join(data_path, "val_pid.pkl"))
        pd.to_pickle(test_x, os.path.join(data_path, "test_x.pkl"))
        pd.to_pickle(test_y, os.path.join(data_path, "test_y.pkl"))
        pd.to_pickle(test_pid, os.path.join(data_path, "test_pid.pkl"))
        pd.to_pickle(los_info, os.path.join(data_path, "los_info.pkl"))
