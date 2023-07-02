import os
import math
import copy
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def df_column_switch(df: pd.DataFrame, column1, column2):
    i = list(df.columns)
    a, b = i.index(column1), i.index(column2)
    i[b], i[a] = i[a], i[b]
    df = df[i]
    return df


def calculate_data_existing_length(data):
    res = 0
    for i in data:
        if not pd.isna(i):
            res += 1
    return res


# elements in data are sorted in time ascending order
def fill_missing_value(data, to_fill_value=0):
    data_len = len(data)
    data_exist_len = calculate_data_existing_length(data)
    if data_len == data_exist_len:
        return data
    elif data_exist_len == 0:
        # data = [to_fill_value for _ in range(data_len)]
        for i in range(data_len):
            data[i] = to_fill_value
        return data
    if pd.isna(data[0]):
        # find the first non-nan value's position
        not_na_pos = 0
        for i in range(data_len):
            if not pd.isna(data[i]):
                not_na_pos = i
                break
        # fill element before the first non-nan value with median
        for i in range(not_na_pos):
            data[i] = to_fill_value
    # fill element after the first non-nan value
    for i in range(1, data_len):
        if pd.isna(data[i]):
            data[i] = data[i - 1]
    return data


def forward_fill_pipeline(
    df: pd.DataFrame,
    default_fill: pd.DataFrame,
    demographic_features: list[str],
    labtest_features: list[str],
):
    grouped = df.groupby("PatientID")

    all_x = []
    all_y = []
    all_pid = []

    for name, group in grouped:
        sorted_group = group.sort_values(by=["RecordTime"], ascending=True)
        patient_x = []
        patient_y = []

        for f in ["Age"] + labtest_features:
            to_fill_value = default_fill[f]
            # take median patient as the default to-fill missing value
            fill_missing_value(sorted_group[f].values, to_fill_value)

        for _, v in sorted_group.iterrows():
            patient_y.append([v["Outcome"], v["LOS"]])
            x = [v["PatientID"]]
            for f in demographic_features + labtest_features:
                x.append(v[f])
            patient_x.append(x)
        all_x.append(patient_x)
        all_y.append(patient_y)
        all_pid.append(name)
    return all_x, all_y, all_pid

# elements in data are sorted in time ascending order


def export_missing_mask_string(data, time):
    missing_mask_string = []

    # calculate the length of a certain patient
    data_len = len(data)

    # calculate the non-empty length of a certain patient
    data_exist_len = calculate_data_existing_length(data)

    # if all existed
    if data_len == data_exist_len:
        # missing_array will be all filled with E to represent the existance of data
        missing_mask_string = ['E' for _ in range(data_len)]
        return missing_mask_string
    # if all not existed
    elif data_exist_len == 0:
        missing_mask_string = ['D' for _ in range(data_len)]
        return missing_mask_string

    not_na_pos = 0
    if pd.isna(data[0]):
        # find the first non-nan value's position
        for i in range(data_len):
            if not pd.isna(data[i]):
                not_na_pos = i
                break
        missing_mask_string = ['D' for _ in range(not_na_pos)]

    # fill element after the first non-nan value
    not_na_time_pos = not_na_pos
    for i in range(not_na_pos, data_len):
        if pd.isna(data[i]):
            # count the days of filling
            missing_mask_string.append(
                float((pd.to_datetime(time[i]) - pd.to_datetime(time[not_na_time_pos])).days))
        else:
            missing_mask_string.append('E')
            not_na_time_pos = i
    return missing_mask_string


def missing_rate_mapping(x, method, b=0.3):

    if method == 'e':
        return (1 - b) * math.exp(-x/2) + b

    elif method == 'd':
        return -b * x + b


def export_missing_mask_pipeline(
    df: pd.DataFrame,
    feature_missing_array: list[float],
    demographic_features: list[str],
    labtest_features: list[str]
):
    grouped = df.groupby("PatientID")
    all_missing_mask_string = []

    # get all_missing_mask_string about D E Delta_time
    for name, group in grouped:
        # sort every group by time in ascending order
        sorted_group = group.sort_values(by=["RecordTime"], ascending=True)
        group_missing_mask_string = []

        for f in demographic_features + labtest_features:
            missing_mask_string = export_missing_mask_string(
                sorted_group[f].values, sorted_group['RecordTime'].values)
            group_missing_mask_string.append(missing_mask_string)
        all_missing_mask_string.append(group_missing_mask_string)

    # all_missing_mask_string.shape-> group_num, feature_num, visit
    local_missing_mask_value = copy.deepcopy(all_missing_mask_string)
    for pid in range(len(all_missing_mask_string)):
        for feature_id in range(len(all_missing_mask_string[pid])):
            for visit_id in range(len(all_missing_mask_string[pid][feature_id])):
                if all_missing_mask_string[pid][feature_id][visit_id] == 'E':
                    local_missing_mask_value[pid][feature_id][visit_id] = 1
                elif all_missing_mask_string[pid][feature_id][visit_id] == 'D':
                    # determined by the missing rate among all patients
                    local_missing_mask_value[pid][feature_id][visit_id] = missing_rate_mapping(
                        feature_missing_array[feature_id], 'd')
                else:
                    # e decay
                    local_missing_mask_value[pid][feature_id][visit_id] = missing_rate_mapping(
                        all_missing_mask_string[pid][feature_id][visit_id], 'e')

    return (all_missing_mask_string, local_missing_mask_value)


# outlier processing
def filter_outlier(element):
    if pd.isna(element):
        return 0
    elif np.abs(float(element)) > 1e4:
        return 0
    else:
        return element


def normalize_dataframe(train_df, val_df, test_df, normalize_features):
    # Calculate the quantiles
    q_low = train_df[normalize_features].quantile(0.05)
    q_high = train_df[normalize_features].quantile(0.95)

    # Filter the DataFrame based on the quantiles
    filtered_df = train_df[(train_df[normalize_features] > q_low) & 
                           (train_df[normalize_features] < q_high)]

    # Calculate the mean and standard deviation and median of the filtered data, also the default fill value
    train_mean = filtered_df[normalize_features].mean()
    train_std = filtered_df[normalize_features].std()
    train_median = filtered_df[normalize_features].median()
    default_fill: pd.DataFrame = (train_median-train_mean)/(train_std+1e-12)

    # LOS info
    los_info = {"los_mean": train_mean["LOS"].item(
    ), "los_std": train_std["LOS"].item(), "los_median": train_median["LOS"].item()}

    # Z-score normalize the train, val, and test sets with train_mean and train_std
    train_df[normalize_features] = (train_df[normalize_features] - train_mean) / (train_std+1e-12)
    val_df[normalize_features] = (val_df[normalize_features] - train_mean) / (train_std+1e-12)
    test_df[normalize_features] = (test_df[normalize_features] - train_mean) / (train_std+1e-12)
        
    train_df.loc[:, normalize_features] = train_df.loc[:, normalize_features].applymap(filter_outlier)
    val_df.loc[:, normalize_features] = val_df.loc[:, normalize_features].applymap(filter_outlier)
    test_df.loc[:, normalize_features] = test_df.loc[:, normalize_features].applymap(filter_outlier)

    return train_df, val_df, test_df, default_fill, los_info, train_mean, train_std


def normalize_df_with_statatistics(df, normalize_features, train_mean, train_std):
    df[normalize_features] = (df[normalize_features] - train_mean) / (train_std+1e-12)
    df.loc[:, normalize_features] = df.loc[:, normalize_features].applymap(filter_outlier)
    return df


def save_tjh_datasets(root: str, src: str='merged.csv', dst: str='dataset_formatted.csv'):
    """
    root: Path to read and save file.
    src: Name of file to read.
    dst: Name of file to save.
    """
    # Read data from file
    df = pd.read_excel(os.path.join(root, src))

    # Rename columns
    df = df.rename(columns={"PATIENT_ID": "PatientID", "outcome": "Outcome", "gender": "Sex", "age": "Age", "RE_DATE": "RecordTime", "Admission time": "AdmissionTime", "Discharge time": "DischargeTime"})
    
    # Fill PatientID column
    df['PatientID'].fillna(method='ffill', inplace=True)
    
    # Format data values
    # gender transformation: 1--male, 0--female
    df['Sex'].replace(2, 0, inplace=True)

    # only reserve y-m-d precision for `RE_DATE` and `Discharge time` columns
    df['RecordTime'] = df['RecordTime'].dt.strftime('%Y-%m-%d')
    df['DischargeTime'] = df['DischargeTime'].dt.strftime('%Y-%m-%d')
    df['AdmissionTime'] = df['AdmissionTime'].dt.strftime('%Y-%m-%d')

    # Exclude patients with missing labels
    df = df.dropna(subset = ['PatientID', 'RecordTime', 'DischargeTime'], how='any')
    
    # Calculate the Length-of-Stay (LOS) label
    df['LOS'] = (pd.to_datetime(df['DischargeTime']) - pd.to_datetime(df['RecordTime'])).dt.days

    # Notice: Set negative LOS values to 0
    df['LOS'] = df['LOS'].apply(lambda x: 0 if x < 0 else x)

    # Drop '2019-nCoV nucleic acid detection' column 
    df = df.drop(columns=['2019-nCoV nucleic acid detection'])

    # Record feature names
    basic_records = ['PatientID', 'RecordTime', 'AdmissionTime', 'DischargeTime']
    target_features = ['Outcome', 'LOS']
    demographic_features = ['Sex', 'Age']
    labtest_features = df.columns.tolist()
    for feat in basic_records + target_features + demographic_features:
        labtest_features.remove(feat) if feat in labtest_features else None
    # labtest_features = ['Hypersensitive cardiac troponinI', 'hemoglobin', 'Serum chloride', 'Prothrombin time', 'procalcitonin', 'eosinophils(%)', 'Interleukin 2 receptor', 'Alkaline phosphatase', 'albumin', 'basophil(%)', 'Interleukin 10', 'Total bilirubin', 'Platelet count', 'monocytes(%)', 'antithrombin', 'Interleukin 8', 'indirect bilirubin', 'Red blood cell distribution width ', 'neutrophils(%)', 'total protein', 'Quantification of Treponema pallidum antibodies', 'Prothrombin activity', 'HBsAg', 'mean corpuscular volume', 'hematocrit', 'White blood cell count', 'Tumor necrosis factorα', 'mean corpuscular hemoglobin concentration', 'fibrinogen', 'Interleukin 1β', 'Urea', 'lymphocyte count', 'PH value', 'Red blood cell count', 'Eosinophil count', 'Corrected calcium', 'Serum potassium', 'glucose', 'neutrophils count', 'Direct bilirubin', 'Mean platelet volume', 'ferritin', 'RBC distribution width SD', 'Thrombin time', '(%)lymphocyte', 'HCV antibody quantification', 'D-D dimer', 'Total cholesterol', 'aspartate aminotransferase', 'Uric acid', 'HCO3-', 'calcium', 'Amino-terminal brain natriuretic peptide precursor(NT-proBNP)', 'Lactate dehydrogenase', 'platelet large cell ratio ', 'Interleukin 6', 'Fibrin degradation products', 'monocytes count', 'PLT distribution width', 'globulin', 'γ-glutamyl transpeptidase', 'International standard ratio', 'basophil count(#)', 'mean corpuscular hemoglobin ', 'Activation of partial thromboplastin time', 'Hypersensitive c-reactive protein', 'HIV antibody quantification', 'serum sodium', 'thrombocytocrit', 'ESR', 'glutamic-pyruvic transaminase', 'eGFR', 'creatinine']

    # Set negative values to NaN
    df[df[demographic_features + labtest_features] < 0] = np.nan

    # Merge by PatientID and RecordTime
    df = df.groupby(['PatientID', 'RecordTime', 'AdmissionTime', 'DischargeTime'], dropna=True, as_index = False).mean()
    
    # Change the order of columns
    df = df[ basic_records + target_features + demographic_features + labtest_features ]

    # Export data to files
    df.to_csv(os.path.join(root, dst), index=False)


def split_datasets(src_root: str, 
                   dst_root: str, 
                   src_filename: str='merged.csv',
                   train: int=70,
                   val: int=10,
                   test: int=20,):
    # calibration
    calib = 5
    seed = 42
    # labtest_features = ['Hypersensitive cardiac troponinI', 'hemoglobin', 'Serum chloride', 'Prothrombin time', 'procalcitonin', 'eosinophils(%)', 'Interleukin 2 receptor', 'Alkaline phosphatase', 'albumin', 'basophil(%)', 'Interleukin 10', 'Total bilirubin', 'Platelet count', 'monocytes(%)', 'antithrombin', 'Interleukin 8', 'indirect bilirubin', 'Red blood cell distribution width ', 'neutrophils(%)', 'total protein', 'Quantification of Treponema pallidum antibodies', 'Prothrombin activity', 'HBsAg', 'mean corpuscular volume', 'hematocrit', 'White blood cell count', 'Tumor necrosis factorα', 'mean corpuscular hemoglobin concentration', 'fibrinogen', 'Interleukin 1β', 'Urea', 'lymphocyte count', 'PH value', 'Red blood cell count', 'Eosinophil count', 'Corrected calcium', 'Serum potassium', 'glucose', 'neutrophils count', 'Direct bilirubin', 'Mean platelet volume', 'ferritin', 'RBC distribution width SD', 'Thrombin time', '(%)lymphocyte', 'HCV antibody quantification', 'D-D dimer', 'Total cholesterol', 'aspartate aminotransferase', 'Uric acid', 'HCO3-', 'calcium', 'Amino-terminal brain natriuretic peptide precursor(NT-proBNP)', 'Lactate dehydrogenase', 'platelet large cell ratio ', 'Interleukin 6', 'Fibrin degradation products', 'monocytes count', 'PLT distribution width', 'globulin', 'γ-glutamyl transpeptidase', 'International standard ratio', 'basophil count(#)', 'mean corpuscular hemoglobin ', 'Activation of partial thromboplastin time', 'Hypersensitive c-reactive protein', 'HIV antibody quantification', 'serum sodium', 'thrombocytocrit', 'ESR', 'glutamic-pyruvic transaminase', 'eGFR', 'creatinine']

    # Read data from file
    df = pd.read_csv(os.path.join(src_root, src_filename))
    demographic_features = ['Sex', 'Age']
    labtest_features = df.columns.tolist()[6:]

    # Group the dataframe by patient ID
    grouped = df.groupby('PatientID')
    patients = np.array(list(grouped.groups.keys()))
    
    # Get the train_val/test patient IDs
    patients_outcome = np.array([grouped.get_group(patient_id)['Outcome'].iloc[0] for patient_id in patients])
    train_val_patients, test_patients = train_test_split(patients, test_size=test/(train+val+test), random_state=seed, stratify=patients_outcome)

    # Get the train/val patient IDs
    train_val_patients_outcome = np.array([grouped.get_group(patient_id)['Outcome'].iloc[0] for patient_id in train_val_patients])
    train_patients, val_patients = train_test_split(train_val_patients, test_size=val/(train+val), random_state=seed, stratify=train_val_patients_outcome)

    # Get the traincal and calib patient IDs (for calibration required methods)
    # train_patients_outcome = np.array([grouped.get_group(patient_id)['Outcome'].iloc[0] for patient_id in train_patients])
    # traincal_patients, calib_patients = train_test_split(train_patients, test_size=calib/train, random_state=seed, stratify=train_patients_outcome)

    # assert that traincal_patients and calib_patients are disjoint
    # assert len(set(traincal_patients).intersection(set(calib_patients))) == 0

    # assert that traincal_patients + cal patients = train_patients. Both lengths should be equal and the union should be equal to the train_patients
    # assert sorted(set(traincal_patients).union(set(calib_patients))) == sorted(train_patients)
    # assert len(traincal_patients) + len(calib_patients) == len(train_patients)

    #  Create train, val, test, [traincal, calib] dataframes for the current fold
    train_df = df[df['PatientID'].isin(train_patients)]
    val_df = df[df['PatientID'].isin(val_patients)]
    test_df = df[df['PatientID'].isin(test_patients)]
    # traincal_df = df[df['PatientID'].isin(traincal_patients)]
    # calib_df = df[df['PatientID'].isin(calib_patients)]

    # Save the train, val, and test dataframes for the current fold to csv files
    
    Path(dst_root).mkdir(parents=True, exist_ok=True)
    train_df.to_csv(os.path.join(dst_root, "train_raw.csv"), index=False)
    val_df.to_csv(os.path.join(dst_root, "val_raw.csv"), index=False)
    test_df.to_csv(os.path.join(dst_root, "test_raw.csv"), index=False)
    # traincal_df.to_csv(os.path.join(dst_root, "traincal_raw.csv"), index=False)
    # calib_df.to_csv(os.path.join(dst_root, "calib_raw.csv"), index=False)

    # Calculate the mean and std of the train set (include age, lab test features, and LOS) on the data in 5% to 95% quantile range
    normalize_features = ['Age'] + labtest_features + ['LOS']
    train_df, val_df, test_df, default_fill, los_info, train_mean, train_std = normalize_dataframe(train_df, val_df, test_df, normalize_features)
    # traincal_df =  normalize_df_with_statatistics(traincal_df, normalize_features, train_mean, train_std)
    # calib_df =  normalize_df_with_statatistics(calib_df, normalize_features, train_mean, train_std)
    
    """
    Notice: we do not need the following code to filter outliers since some of the `outliers` are actually the real values.
    """
    
    # Drop rows if all features are recorded NaN
    train_df = train_df.dropna(axis=0, how='all', subset=normalize_features)
    val_df = val_df.dropna(axis=0, how='all', subset=normalize_features)
    test_df = test_df.dropna(axis=0, how='all', subset=normalize_features)

    train_df.to_csv(os.path.join(dst_root, "train_after_zscore.csv"), index=False)
    val_df.to_csv(os.path.join(dst_root, "val_after_zscore.csv"), index=False)
    test_df.to_csv(os.path.join(dst_root, "test_after_zscore.csv"), index=False)

    # Forward Imputation after grouped by PatientID
    # Notice: if a patient has never done certain lab test, the imputed value will be the median value calculated from train set
    train_x, train_y, train_pid = forward_fill_pipeline(train_df, default_fill, demographic_features, labtest_features)
    val_x, val_y, val_pid = forward_fill_pipeline(val_df, default_fill, demographic_features, labtest_features)
    test_x, test_y, test_pid = forward_fill_pipeline(test_df, default_fill, demographic_features, labtest_features)

    # Save the imputed dataset to pickle file
    pd.to_pickle(train_x, os.path.join(dst_root, "train_x.pkl"))
    pd.to_pickle(train_y, os.path.join(dst_root, "train_y.pkl"))
    pd.to_pickle(train_pid, os.path.join(dst_root, "train_pid.pkl"))
    pd.to_pickle(val_x, os.path.join(dst_root, "val_x.pkl"))
    pd.to_pickle(val_y, os.path.join(dst_root, "val_y.pkl"))
    pd.to_pickle(val_pid, os.path.join(dst_root, "val_pid.pkl"))
    pd.to_pickle(test_x, os.path.join(dst_root, "test_x.pkl"))
    pd.to_pickle(test_y, os.path.join(dst_root, "test_y.pkl"))
    pd.to_pickle(test_pid, os.path.join(dst_root, "test_pid.pkl"))
    pd.to_pickle(los_info, os.path.join(dst_root, "los_info.pkl"))


def save_cdsl_datasets(root: str, src: str='merged.csv', dst: str='dataset_formatted.csv'):
    """
    root: Path to read and save file.
    src: Name of file to read.
    dst: Name of file to save.
    """
    # Read data from file
    df = pd.read_excel(os.path.join(root, src))

    # Exclude patients with missing labels
    demographic = df.dropna(axis=0, how='any', subset=['IDINGRESO', 'F_INGRESO_ING', 'F_ALTA_ING', 'MOTIVO_ALTA_ING'])

    # Select necessary columns from demographic
    demographic = demographic[
        [
            'IDINGRESO', 
            'EDAD',
            'SEX',
            'F_INGRESO_ING', 
            'F_ALTA_ING', 
            'MOTIVO_ALTA_ING', 
            'ESPECIALIDAD_URGENCIA', 
            'DIAG_URG'
        ]
    ]

    # Rename column
    demographic = demographic.rename(columns={
        'IDINGRESO': 'PATIENT_ID',
        'EDAD': 'AGE',
        'SEX': 'SEX',
        'F_INGRESO_ING': 'ADMISSION_DATE',
        'F_ALTA_ING': 'DEPARTURE_DATE',
        'MOTIVO_ALTA_ING': 'OUTCOME',
        'ESPECIALIDAD_URGENCIA': 'DEPARTMENT_OF_EMERGENCY',
        'DIAG_URG': 'DIAGNOSIS_AT_EMERGENCY_VISIT'
    })

    # SEX: Male: 1; Female: 0
    demographic['SEX'].replace('MALE', 1, inplace=True)
    demographic['SEX'].replace('FEMALE', 0, inplace=True)

    # Outcome: Fallecimiento(dead): 1; others: 0
    demographic['OUTCOME'] = demographic['OUTCOME'].map(lambda x: 1 if x == 'Fallecimiento' else 0)

    # Only reserve useful columns in demographic table
    demographic = demographic[
        [
            'PATIENT_ID',
            'AGE',
            'SEX',
            'ADMISSION_DATE',
            'DEPARTURE_DATE',
            'OUTCOME',
            # 'DIFFICULTY_BREATHING',
            # 'SUSPECT_COVID',
            # 'FEVER',
            # 'EMERGENCY'
        ]
    ]

    # TODO: change the path of saving the formatted dataset
    demographic.describe().to_csv(os.path.join(root, 'statistics', 'demographic_overview.csv'), index=False)
    demographic.to_csv(os.path.join(root, 'processed', 'demographic.csv'), index=False)
