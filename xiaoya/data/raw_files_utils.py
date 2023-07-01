import os

import pandas as pd

PROCESSED_ROOT = '' 
DATASETS_ROOT = '' 
CHECKPOINTS_ROOT = ''

def delete_folder(root: str):
    """
    root: Path to delete.
    """
    if os.path.exists(root):
        for root, dirs, files in os.walk(root, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(root)

def read_file(file, filename: str):
    """
    file: Uploaded file in request.FILES.
    """
    suffix = filename.split('.')[-1]
    if suffix == 'csv':
        return pd.read_csv(file)
    elif suffix == 'xlsx' or suffix == 'xls':
        return pd.read_excel(file)
    else:
        return None
    
def save_csv(df: pd.DataFrame, root: str, filename: str, **kwargs):
    """
    root: Path to save file.
    df: DataFrame to save.
    filename: Name of file.
    """
    if not os.path.exists(root):
        os.makedirs(root)
    
    df.to_csv(os.path.join(root, filename), **kwargs)

def get_save_name(type: int):
    """
    type: 1, 2, 3.
    """
    names = {
        1: 'labtest.csv',
        2: 'events.csv',
        3: 'target.csv'
    }
    return names[type]

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
        if str(df['Outcome'][0]).lower() in ['true', 'false']:
            df['Outcome'].replace({False: 0, True: 1}, inplace=True)
        return df

    new_dict = {}
    new_dict['PatientID'] = df['PatientID']
    new_dict['RecordTime'] = df['RecordTime']

    for name in df['Name'].dropna().unique():
        new_dict[name] = df[df['Name'] == name]['Value']
    return pd.DataFrame(new_dict)

def merge_dfs(df_labtest, df_events=None, df_target=None):
    """
    将多个DataFrame合并为一个
    """
    df = df_labtest
    cols = df.columns.tolist()
    cols.remove('Age')
    df = df[['Age'] + cols]   

    if df_events is not None:
        df = pd.merge(df_events, df, how='outer', on=['PatientID', 'RecordTime'])

    if df_target is not None:
        df = pd.merge(df_target, df, how='outer', on=['PatientID', 'RecordTime'])
    
    return df

def processed_root(username: str, job_name: str):
    return os.path.join(PROCESSED_ROOT, username, job_name)

def datasets_root(username: str, job_name: str):
    return os.path.join(DATASETS_ROOT, username, job_name)

def checkpoints_root(username: str, job_name: str):
    return os.path.join(CHECKPOINTS_ROOT, username, job_name)
