import os

import pandas as pd


def get_features(df, table: int):
    """
    df: DataFrame.
    table: 1, 2, 3.
    """

    if table == 1 or table == 2:
        feats = df['Name'].dropna().unique().tolist()
    else:
        feats = df.columns.tolist()
        feats.remove('PatientID')
        feats.remove('RecordTime')
    return feats

def to_dataframe(df: pd.DataFrame, table: int):
    """
    将读入的文件转换为标准格式，方便后续合并
    """
    if table == 3:
        df = df.drop_duplicates(subset=['PatientID', 'RecordTime'], keep='last')
    else:
        df = df.drop_duplicates(subset=['PatientID', 'RecordTime', 'Name'], keep='last')
        columns = ['PatientID', 'RecordTime'] + list(df['Name'].dropna().unique())
        df_new = pd.DataFrame(data=None, columns=columns)
        grouped = df.groupby(['PatientID', 'RecordTime'])
        for i, group in enumerate(grouped):
            patient_id, record_time = group[0]
            df_group = group[1]
            df_new.loc[i, 'PatientID'] = patient_id
            df_new.loc[i, 'RecordTime'] = record_time
            for _, row in df_group.iterrows():
                df_new.loc[i, row['Name']] = row['Value']
        df = df_new

    df['RecordTime'] = pd.to_datetime(df['RecordTime'], format='%Y-%m-%d')
    df.sort_values(by=['PatientID', 'RecordTime'], inplace=True)
    return df

def merge_dfs(df_labtest, df_events=None, df_target=None):
    """
    将多个DataFrame合并为一个
    """
    df = df_labtest 

    if df_events is not None:
        df = pd.merge(df, df_events, left_on=['PatientID', 'RecordTime'], right_on=['PatientID', 'RecordTime'], how='outer')

    if df_target is not None:
        df = pd.merge(df, df_target, left_on=['PatientID', 'RecordTime'], right_on=['PatientID', 'RecordTime'], how='outer')
    
    # 前向填充events
    for col in df_events.columns.tolist():
        df[col] = df[col].fillna(method='ffill')

    # 调整列的顺序
    cols = ['PatientID', 'RecordTime', 'Outcome', 'LOS', 'Sex', 'Age']
    all_cols = df.columns.tolist()
    for col in cols:
        all_cols.remove(col) if col in all_cols else None
    all_cols = cols + all_cols
    df = df[all_cols]
    return df
