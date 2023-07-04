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
        data_path: Path.
            path to save processed data, default: Path('./datasets').

    """

    def __init__(
            self, 
            labtest_data: pd.DataFrame,
            events_data: pd.DataFrame,
            target_data: pd.DataFrame,
            data_path: Path = Path('./datasets'),
            save_processed_data: bool = False
        ) -> None:

        self.labtest_df = pd.DataFrame(labtest_data)
        self.events_df = pd.DataFrame(events_data)
        self.target_df = pd.DataFrame(target_data)


        self.labtest_standard_df = to_dataframe(self.labtest_df, 'labtest')
        self.events_standard_df = to_dataframe(self.events_df, 'events')
        self.target_standard_df = to_dataframe(self.target_df, 'target')
        self.merged_df = merge_dfs(
            self.labtest_standard_df,
            self.events_standard_df,
            self.target_standard_df
        )
        self.data_path = data_path
        if save_processed_data:
            data_path.mkdir(parents=True, exist_ok=True)
            self.labtest_standard_df.to_csv(os.path.join(data_path, 'labtest_standard_data.csv'), index=False)
            self.events_standard_df.to_csv(os.path.join(data_path, 'events_standard_data.csv'), index=False)
            self.target_standard_df.to_csv(os.path.join(data_path, 'target_standard_data.csv'), index=False)
            self.merged_df.to_csv(os.path.join(data_path, 'merged_standard_data.csv'), index=False)

    def extract_features(self) -> Dict:
        """
        Extract features from the merged dataframe.

        Returns:
            feats: Dict.
                features.
        """

        feats = {}
        feats['labtest_features'] = get_features(self.labtest_df, 'labtest')
        feats['events_features'] = get_features(self.events_df, 'events')
        feats['target_features'] = get_features(self.target_df, 'target')
        return feats

    def analyze_dataset(self) -> List:
        """
        Analyze the dataset.

        Returns:
            statistic_info: List.
                statistic information of the dataset.
        """
        
        len_df = len(self.merged_df.index)
        features = self.extract_features()
        statistic_info = []
        for idx, e in enumerate(['PatientID'] + features['target_features'] + features['events_features'] + features['labtest_features']):
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
    
    def split_dataset(self, 
                    train: int = 70, 
                    val: int = 10, 
                    test: int = 20, 
                    seed: int = 42
                ) -> None:
        """
        Split the dataset into train/val/test sets.

        Args:
            train: int.
                train set percentage.
            val: int.
                val set percentage.
            test: int.
                test set percentage.
            seed: int.
                random seed.
        """
        assert train + val + test == 100, "train + val + test must equal to 100"

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
        self.train_raw_df = self.merged_df[self.merged_df['PatientID'].isin(train_patients)]
        self.val_raw_df = self.merged_df[self.merged_df['PatientID'].isin(val_patients)]
        self.test_raw_df = self.merged_df[self.merged_df['PatientID'].isin(test_patients)]
    
    def normalize_dataset(self,
                        normalize_features: List[str]
                    ) -> None:
        """
        Normalize the dataset.

        Args:
            normalize_features: List[str].
                features to be normalized.
        """

        # Calculate the mean and std of the train set (include age, lab test features, and LOS) on the data in 5% to 95% quantile range
        train_after_zscore, val_after_zscore, test_after_zscore, self.default_fill, self.los_info, _, _ = \
            normalize_dataframe(self.train_raw_df, self.val_raw_df, self.test_raw_df, normalize_features)
        
        # Drop rows if all features are recorded NaN
        self.train_after_zscore = train_after_zscore.dropna(axis=0, how='all', subset=normalize_features)
        self.val_after_zscore = val_after_zscore.dropna(axis=0, how='all', subset=normalize_features)
        self.test_after_zscore = test_after_zscore.dropna(axis=0, how='all', subset=normalize_features)

    def forward_fill_dataset(self,
                            demographic_features: List[str],
                            labtest_features: List[str]
                        ) -> None:
        """
        Forward fill the dataset.

        Args:
            demographic_features: List[str].
                demographic features.
            labtest_features: List[str].
                lab test features.
        """
        
        # Forward Imputation after grouped by PatientID
        self.train_x, self.train_y, self.train_pid = forward_fill_pipeline(self.train_after_zscore, self.default_fill, demographic_features, labtest_features)
        self.val_x, self.val_y, self.val_pid = forward_fill_pipeline(self.val_after_zscore, self.default_fill, demographic_features, labtest_features)
        self.test_x, self.test_y, self.test_pid = forward_fill_pipeline(self.test_after_zscore, self.default_fill, demographic_features, labtest_features)

    def execute(self,
        train: int = 70,
        val: int = 10,
        test: int = 20,
        seed: int = 42,
    ) -> None:
        """
        Execute the preprocessing pipeline, including split the dataset, normalize the dataset, and forward fill the dataset.

        Args:
            train: int.
                train set percentage.
            val: int.
                val set percentage.
            test: int.
                test set percentage.
            seed: int.
                random seed.
        """
        
        data_path = self.data_path
        features = self.extract_features()
        demographic_features = features['events_features']
        labtest_features = features['labtest_features']
        if 'Age' in labtest_features:
            demographic_features.append('Age')
            labtest_features.remove('Age') 

        # Split the dataset
        self.split_dataset(train, val, test, seed)

        # Normalize the dataset
        self.normalize_dataset(['Age'] + labtest_features + ['LOS'])

        # Forward fill the dataset
        self.forward_fill_dataset(demographic_features, labtest_features)

        # Save the dataframes
        data_path.mkdir(parents=True, exist_ok=True)
        self.train_raw_df.to_csv(os.path.join(data_path, "train_raw.csv"), index=False)
        self.val_raw_df.to_csv(os.path.join(data_path, "val_raw.csv"), index=False)
        self.test_raw_df.to_csv(os.path.join(data_path, "test_raw.csv"), index=False)

        self.train_after_zscore.to_csv(os.path.join(data_path, "train_after_zscore.csv"), index=False)
        self.val_after_zscore.to_csv(os.path.join(data_path, "val_after_zscore.csv"), index=False)
        self.test_after_zscore.to_csv(os.path.join(data_path, "test_after_zscore.csv"), index=False)

        pd.to_pickle(self.train_x, os.path.join(data_path, "train_x.pkl"))
        pd.to_pickle(self.train_y, os.path.join(data_path, "train_y.pkl"))
        pd.to_pickle(self.train_pid, os.path.join(data_path, "train_pid.pkl"))
        pd.to_pickle(self.val_x, os.path.join(data_path, "val_x.pkl"))
        pd.to_pickle(self.val_y, os.path.join(data_path, "val_y.pkl"))
        pd.to_pickle(self.val_pid, os.path.join(data_path, "val_pid.pkl"))
        pd.to_pickle(self.test_x, os.path.join(data_path, "test_x.pkl"))
        pd.to_pickle(self.test_y, os.path.join(data_path, "test_y.pkl"))
        pd.to_pickle(self.test_pid, os.path.join(data_path, "test_pid.pkl"))
        pd.to_pickle(self.los_info, os.path.join(data_path, "los_info.pkl"))
