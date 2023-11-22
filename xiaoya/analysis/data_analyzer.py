from typing import List, Dict, Optional

import torch
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import optimize
from sklearn.cluster import KMeans

from xiaoya.pyehr.pipelines import DlPipeline
from xiaoya.pipeline import Pipeline


class DataAnalyzer:
    """
    DataAnalyzer.

    Args:
        config: Dict.
            the config of the pipeline.
        model_path: str.
            the saved path of the model.
    """

    def __init__(self, 
        config: Dict,
        model_path: str,
    ) -> None:
        self.config = config
        self.model_path = model_path

    def adaptive_feature_importance(
            self, 
            df: pd.DataFrame,
            x: List,
            patient_index: Optional[int] = None,
            patient_id: Optional[int] = None,
        ) -> Dict:
        """
        Return the adaptive feature importance of a patient.

        Args:
            df: pd.DataFrame.
                A dataframe representing the patients' raw data.
            x: List.
                A List of shape [batch_size, time_step, feature_dim],
                representing the input of the patients.
            patient_index: Optional[int].
                The index of the patient in dataframe.
            patient_id: Optional[int].
                The patient ID recorded in dataframe.
                patient_index and patient_id can only choose one.

        Returns: Dict.
            detail: List.
                a List of shape [time_step, feature_dim], representing the adaptive feature importance of the patient.
        """
        pipeline = DlPipeline(self.config)
        pipeline = pipeline.load_from_checkpoint(self.model_path)
        xid = patient_index if patient_index is not None else list(df['PatientID'].drop_duplicates()).index(patient_id)        
        x = torch.Tensor(x[xid]).unsqueeze(0)   # [1, ts, f]
        if pipeline.on_gpu:
            x = x.to('cuda:0')
        _, _, input_attn = pipeline.predict_step(x)
        input_attn = input_attn[0]

        rerult = {
            'detail': input_attn.detach().cpu().numpy().tolist() # [ts, f]
        }
        return rerult
    
    def feature_importance(
            self,
            df: pd.DataFrame,
            x: List,
            patient_index: Optional[int] = None,
            patient_id: Optional[int] = None,
        ) -> Dict:
        """
        Return feature importance of a patient.

        Args:
            df: pd.DataFrame.
                A dataframe representing the patients' raw data.
            x: List.
                A List of shape [batch_size, time_step, feature_dim],
                representing the input of the patients.
            patient_index: Optional[int].
                The index of the patient in dataframe.
            patient_id: Optional[int].
                The patient ID recorded in dataframe.
                patient_index and patient_id can only choose one.

        Returns: Dict.
            detail: List.
                a List of dicts with shape [lab_dim]:
                name: the name of the feature.
                value: the feature importance value.
                adaptive: the adaptive feature importance value.
        """
        pipeline = DlPipeline(self.config)
        pipeline = pipeline.load_from_checkpoint(self.model_path)
        xid = patient_index if patient_index is not None else list(df['PatientID'].drop_duplicates()).index(patient_id)        
        x = torch.Tensor(x[xid]).unsqueeze(0)   # [1, ts, f]
        x = torch.cat((x, x), dim=0)
        if pipeline.on_gpu:
            x = x.to('cuda:0')
        _, _, feat_attn = pipeline.predict_step(x)
        feat_attn = feat_attn[0].detach().cpu().numpy()  # [ts, f] feature importance value
        column_names = list(df.columns[6:])
        return {
            'detail': [{
                'name': column_names[i],
                'value': feat_attn[-1, i].tolist(),
                'adaptive': feat_attn[:, i].tolist(),
            } for i in range(len(column_names))]
        }

    def risk_curve(
            self, 
            df: pd.DataFrame,
            x: List,
            mean: List,
            std: List,
            mask: Optional[List] = None,
            patient_index: Optional[int] = None,
            patient_id: Optional[int] = None,
        ) -> Dict:
        """
        Return risk curve of a patient.

        Args:
            df: pd.DataFrame.
                A dataframe representing the patients' raw data.
            x: List.
                A List of shape [batch_size, time_step, feature_dim],
                representing the input of the patients.
            mask: Optional[List].
                A List of shape [batch_size, time_step, feature_dim],
                representing the missing status of the patients's raw data.
            patient_index: Optional[int].
                The index of the patient in dataframe.
            patient_id: Optional[int].
                The patient ID recorded in dataframe.
                patient_index and patient_id can only choose one.

        Returns: Dict.
            detail: A List of dicts with shape [lab_dim].
                name: the name of the feature.
                value: the value of the feature in all visits.
                importance: the feature importance value.
                adaptive: the adaptive feature importance value.
                missing: the missing status of the feature in all visits.
                unit: the unit of the feature.
            time: A List of shape [time_step],
                representing the date of the patient's visits.
            time_risk: A List of shape [time_step],
                representing the risk of the patient at each visit.
        """
        pipeline = DlPipeline(self.config)
        pipeline = pipeline.load_from_checkpoint(self.model_path)
        if patient_index is not None:
            xid = patient_index
            patient_id = list(df['PatientID'].drop_duplicates())[patient_index]
        else:
            xid = list(df['PatientID'].drop_duplicates()).index(patient_id)
        x = torch.Tensor(x[xid]).unsqueeze(0)   # [1, ts, f]
        x = torch.cat((x, x), dim=0)
        if pipeline.on_gpu:
            x = x.to('cuda:0')
        y_hat, _, feat_attn = pipeline.predict_step(x)
        x = x[0, :, 2:].detach().cpu().numpy()  # [ts, lab]
        y_hat = y_hat[0].detach().cpu().numpy()  # [ts, 2]
        feat_attn = feat_attn[0].detach().cpu().numpy()  # [ts, lab]
        mask = np.array(mask[xid]) if mask is not None else np.zeros_like(feat_attn)  # [ts, f]
        column_names = list(df.columns[6:])
        record_times = list(df[df['PatientID'] == patient_id]['RecordTime'].values) # [ts]
        return {
            'detail': [{
                'name': column_names[i],
                'value': (x[:, i] * std[column_names[i]] + mean[column_names[i]]).tolist(),
                'importance': feat_attn[-1, i].tolist(),
                'adaptive': feat_attn[:, i].tolist(),
                'missing': mask[:, i].tolist(),
                'unit': ''
            } for i in range(len(column_names))],
            'time': record_times,   # ts
            'time_risk': y_hat[:, 0],  # ts
        }
    
    def ai_advice(
            self,
            df: pd.DataFrame,
            x: List,
            mean: List,
            std: List,
            time_index: int,
            patient_index: Optional[int] = None,
            patient_id: Optional[int] = None,
        ) -> List:
        """
        Return the advice of the AI system.

        Args:
            df: pd.DataFrame.
                the dataframe of the patient.
            x: List.
                the input of the patient.
            mask: Optional[List].
                the missing mask of the patient.
            patient_index: Optional[int].
                the index of the patient in dataframe.
            patient_id: Optional[int].
                the patient ID.
            time_index: int.
                the time index of the patient.

        Returns: Dict.
            detail: A List of dicts with shape [num_advice], default is 3.
                name: the name of the feature.
                old_value: the old value of the feature.
                new_value: the new value of the feature.
        """
        pipeline = DlPipeline(self.config)
        pipeline = pipeline.load_from_checkpoint(self.model_path)
        if patient_index is not None:
            xid = patient_index
            patient_id = list(df['PatientID'].drop_duplicates())[patient_index]
        else:
            xid = list(df['PatientID'].drop_duplicates()).index(patient_id)
        x = torch.Tensor(x[xid]).unsqueeze(0)   # [1, ts, f]
        x = torch.cat((x, x), dim=0)
        device = torch.device('cuda:0' if pipeline.on_gpu else 'cpu')
        _, _, feat_attn = pipeline.predict_step(x.to(device))
        feat_attn = feat_attn[0].detach().cpu().numpy()  # [ts, f]

        demo_dim = 2
        column_names = list(df.columns[4 + demo_dim:])
        feature_last_step: List = feat_attn[time_index].tolist()
        index_dict = {index: value for index, value in enumerate(feature_last_step)}
        max_indices = sorted(index_dict, key=index_dict.get, reverse=True)
        if len(max_indices) > 3:
            max_indices = max_indices[:3]

        def f(x, args):
            input, i = args
            input[-1][-1][i] = torch.from_numpy(x).float()
            input = torch.cat((input, input), dim=0)
            y_hat, _, _ = pipeline.predict_step(input.to(device))      # y_hat: [bs, seq_len, 2]
            return y_hat[0][time_index][0].cpu().detach().numpy().item()

        result = []
        for i in max_indices:
            x0 = float(x[-1][-1][i])
            bounds = (max(-3, x0 - 1), min(3, x0 + 1))
            args = (x, i)
            res = optimize.minimize(f, x0=x0, bounds=(bounds,), args=(args,), method='nelder-mead', options={'disp': True})
            result.append({
                'name': column_names[i],
                'old_value': x0  * std[column_names[i]] + mean[column_names[i]],
                'new_value': res.x[0] * std[column_names[i]] + mean[column_names[i]]
            })
        return {'detail': result}

    def data_dimension_reduction(
            self,
            df: pd.DataFrame,
            x: List,
            mean_age: Optional[float],
            std_age: Optional[float],
            method: str = "PCA",
            dimension: int = 2,
            target: str = "outcome",
        )-> List:
        """
        Return dimension reduced data of the patients.

        Args:
            df: pd.DataFrame.
                A dataframe representing the patients' raw data.
            x: List.
                A List of shape [batch_size, time_step, feature_dim],
                representing the input of the patients.
            mean_age: Optional[float].
                The mean age of the patients.
            std_age: Optional[float].
                The std age of the patients.
            method: str.
                The method of dimension reduction, one of "PCA" and "TSNE", default is "PCA".
            dimension: int.
                The dimension of dimension reduction, one of 2 and 3, default is 2.
            target: str.
                The target of the model, one of "outcome", "los" and "multitask", default is "outcome".

        Returns: Dict.
            detail: A List of dicts with shape [lab_dim],
                data: the dimension reduced data of the patient.
                patient_id: the patient ID of the patient.
                record_time: the visit datetime of the patient.
                age: the age of the patient.      
        """
        num = len(x)
        patients = []
        pid = df['PatientID'].drop_duplicates().tolist()  # [b]
        record_time = df.groupby('PatientID')['RecordTime'].apply(list).tolist()  # [b, ts]
        for i in range(num):
            xi = torch.tensor(x[i]).unsqueeze(0)
            pidi = torch.tensor(pid[i]).unsqueeze(0)
            timei = record_time[i]
            pipeline = DlPipeline(self.config)
            pipeline = pipeline.load_from_checkpoint(self.model_path)
            xi = torch.cat((xi, xi), dim=0)
            if pipeline.on_gpu:
                xi = xi.to('cuda:0')   # cuda
            y_hat, embedding, _ = pipeline.predict_step(xi)
            embedding = embedding[0].cpu().detach().numpy().squeeze()  # cpu
            y_hat = y_hat[0].cpu().detach().numpy().squeeze()      # cpu
            
            df = pd.DataFrame(embedding)
            if method == "PCA":  # 判断降维类别
                reduction_model = PCA().fit_transform(df)
            elif method == "TSNE":
                reduction_model = TSNE(n_components=dimension, learning_rate='auto', init='random').fit_transform(df)
            if(reduction_model.shape[0] != reduction_model.shape[1]):
                continue
            if target == "outcome":
                y_hat = y_hat[:, 0].flatten().tolist()
            else:
                y_hat = y_hat[:, 1].flatten().tolist()   
            
            patient = {}
            if dimension == 2:  # 判断降维维度
                patient['data'] = [list(x) for x in zip(reduction_model[:, 0], reduction_model[:, 1], y_hat)]
            elif dimension == 3:
                patient['data'] = [list(x) for x in zip(reduction_model[:, 0], reduction_model[:, 1], reduction_model[:, 2], y_hat)]
            patient['patient_id'] = pidi.item()
            patient['record_time'] = [str(x) for x in timei]
            if std_age is not None and mean_age is not None:
                patient['age'] = int(xi[0][0][1].item() * std_age + mean_age)
            patients.append(patient)
        return {'detail': patients}
    
    def similar_patients(
            self,
            x_df: pd.DataFrame,
            x: List,
            p_df: pd.DataFrame,
            patients: List,
            patient_index: Optional[int] = None,
            patient_id: Optional[int] = None,
            n_clu: int = 10,
            topk: int = 6,
        ):
        """
        Return similar patients information.
        """
        
        pipeline = DlPipeline(self.config)
        pipeline = pipeline.load_from_checkpoint(self.model_path)
        xid = patient_index if patient_index is not None else list(x_df['PatientID'].drop_duplicates()).index(patient_id)        
        x = torch.Tensor(x[xid]).unsqueeze(0)   # [1, ts, f]
        patients = torch.Tensor(patients)       # [b, ts, f]
        if pipeline.on_gpu:
            x = x.to('cuda:0')
            patients = patients.to('cuda:0')
        _, x_context, _ = pipeline.predict_step(x)
        _, patients_context, _ = pipeline.predict_step(patients)
        
        x_context, patients_context = np.array(x_context[:, -1, :]), np.array(patients_context[:, -1, :])
        cluster = KMeans(n_clusters=n_clu).fit(patients_context)
        center_id = cluster.predict(x_context)
        similar_patients_id = cluster.labels_ == center_id
        similar_patients_context = patients_context[similar_patients_id]
        similar_patients_info = p_df[similar_patients_id]
        
        dist = np.sqrt(np.sum(np.square(x_context - similar_patients_context), axis=1))
        dist_dict = {index: value for index, value in enumerate(dist)}
        dist_sorted = list(sorted(dist_dict.items(), key=lambda x: x[1]))[:topk]
        index = [item[0] for item in dist_sorted]
        
        topDist = dist[index]
        maxDist, minDist = np.max(topDist), np.min(topDist)
        topSimilarity = ((topDist - minDist) / (maxDist - minDist)).tolist()