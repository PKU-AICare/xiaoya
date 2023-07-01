from typing import List, Dict

import pandas as pd

from raw_files_utils import (
    get_save_name,
    get_features,
    to_dataframe,
    merge_dfs
)


class FileHandler:
    """
    Import user uploaded data, merge data tables, stats...

    Args:

    """

    def __init__(
            self, 
            labtest_data: pd.DataFrame,
            events_data: pd.DataFrame,
            target_data: pd.DataFrame,
        ) -> None:

        self.labtest_df = pd.DataFrame(labtest_data)
        self.events_df = pd.DataFrame(events_data)
        self.target_df = pd.DataFrame(target_data)

    def preview_features(self) -> Dict:
        feats = {}
        for i, df in enumerate([self.labtest_df, self.events_df, self.target_df]): 
            try:
                # Get features from file
                feat = get_features(df, i)
            except:
                feat = None
            save_name = get_save_name(i)
            feats[save_name] = feat
        return feats
    
    def merge_tables(self) -> None:
        self.labtest_standard_df = to_dataframe(self.labtest_df, 1)
        self.events_standard_df = to_dataframe(self.events_df, 2)
        self.target_standard_df = to_dataframe(self.target_df, 3)
        self.merged_df = merge_dfs(
            self.labtest_standard_df,
            self.events_standard_df,
            self.target_standard_df
        )

    def analyse_data(self) -> List:
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
    