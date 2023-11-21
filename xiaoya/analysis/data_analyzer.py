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
                A list of shape [batch_size, time_step, feature_dim],
                representing the input of the patients.
            patient_index: Optional[int].
                The index of the patient in dataframe.
            patient_id: Optional[int].
                The patient ID recorded in dataframe.
                patient_index and patient_id can only choose one.

        Returns:
            Dict.
                detail: a numpy array of shape [time_step, feature_dim],
                representing the adaptive feature importance of the patient.
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
                the dataframe of the patient.
            x: List.
                the input of the patient.
            patient_index: Optional[int].
                the index of the patient in dataframe.
            patient_id: Optional[int].
                the patient ID.

        Returns:
            Dict.
                the feature importance.
        """
        pipeline = DlPipeline(self.config)
        pipeline = pipeline.load_from_checkpoint(self.model_path)
        xid = patient_index if patient_index is not None else list(df['PatientID'].drop_duplicates()).index(patient_id)        
        x = torch.Tensor(x[xid]).unsqueeze(0)   # [1, ts, f]
        if pipeline.on_gpu:
            x = x.to('cuda:0')
        _, _, feat_attn = pipeline.predict_step(x)
        feat_attn = feat_attn[0][-1]    # [f, f]
        column_names = list(df.columns[6:])
        return {
            'detail': {
                'name': column_names,
                'value': feat_attn.detach().cpu().numpy().tolist()  # feature importance value
            }
        }

    def risk_curve(
            self, 
            df: pd.DataFrame,
            x: List,
            mask: Optional[List],
            patient_index: Optional[int] = None,
            patient_id: Optional[int] = None,
        ) -> Dict:
        """
        Return data to draw risk curve of a patient.

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

        Returns:
            Dict.
                the data to draw risk curve.
        """
        pipeline = DlPipeline(self.config)
        pipeline = pipeline.load_from_checkpoint(self.model_path)
        if patient_index is not None:
            xid = patient_index
            patient_id = list(df['PatientID'].drop_duplicates())[patient_index]
        else:
            xid = list(df['PatientID'].drop_duplicates()).index(patient_id)
        x = torch.Tensor(x[xid]).unsqueeze(0)   # [1, ts, f]
        if pipeline.on_gpu:
            x = x.to('cuda:0')
        y_hat, _, scores = pipeline.predict_step(x)
        x = x[0].detach().cpu().numpy()  # [ts, f]
        y_hat = y_hat[0].detach().cpu().numpy()  # [ts, 2]
        scores = scores[0].detach().cpu().numpy()  # [ts, f]
        
        mask = np.array(mask[xid]) if mask is not None else None  # [ts, f]
        column_names = list(df.columns[6:])
        record_times = list(item[1] for item in df[df['PatientID'] == patient_id]['RecordTime'].items()) 

        return {
            'detail': [{
                'name': column_names[i],
                'value': x[:, i],
                'time_step_feature_importance': scores[:, i],
                'missing': mask[:, i],
                'unit': ''
            } for i in range(len(column_names))],
            'time': record_times,   # ts
            'time_risk': y_hat[:, 0],  # ts
        }
    
    def ai_advice(
            self,
            df: pd.DataFrame,
            x: List,
            mask: List,
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

        Returns:
            List.
                the advice of the AI system.
        """
        pipeline = DlPipeline(self.config)
        pipeline = pipeline.load_from_checkpoint(self.model_path)
        if patient_index is not None:
            xid = patient_index
            patient_id = list(df['PatientID'].drop_duplicates())[patient_index]
        else:
            xid = list(df['PatientID'].drop_duplicates()).index(patient_id)
        x = torch.Tensor(x[xid]).unsqueeze(0)   # [1, ts, f]
        device = torch.device('cuda:0' if pipeline.on_gpu else 'cpu')
        _, _, feat_attn = pipeline.predict_step(x.to(device))
        feat_attn = feat_attn[0].detach().cpu().numpy()  # [ts, f, f]
        mask = mask[xid][time_index]

        demo_dim = 2
        column_names = list(df.columns[4 + demo_dim:])
        feature_last_step: List = feat_attn[time_index].sum(dim=0).tolist()[demo_dim:]
        index_dict = {index: value for index, value in enumerate(feature_last_step) if mask[index] != 0}
        max_indices = sorted(index_dict, key=index_dict.get, reverse=True)
        if len(max_indices) > 3:
            max_indices = max_indices[:3]

        def f(x, args):
            input, i = args
            input[-1][-1][i] = torch.from_numpy(x).float()
            y_hat, _, _ = pipeline.predict_step(input.to(device))      # y_hat: [bs, seq_len, 2]
            return y_hat[0][time_index][0].cpu().detach().numpy()

        result = []
        for i in max_indices:
            x0 = float(x[-1][-1][i])
            bounds = (max(-3, x0 - 1), min(3, x0 + 1))
            args = (x, i)
            res = optimize.minimize(f, x0=x0, bounds=(bounds,), args=(args,), method='nelder-mead', options={'disp': True})
            result.append({
                'name': column_names[i],
                'old_value': x0,
                'new_value': float(res.x[0])
            })
        return {'detail': result}

    def data_dimension_reduction(
            self,
            x: List,
            pid: List,
            record_time: List,
            mean_age: Optional[float],
            std_age: Optional[float],
            method: str = "PCA",
            dimension: int = 2,
            target: str = "outcome",
        )-> List:
        """
        Return data to draw dimension reduction.

        Args:
            x: List.
                the input of the patient.
            pid: List.
                the patient ID.
            record_time: List.
                the record time of the patient.
            mean_age: Optional[float].
                the mean age of the patient.
            std_age: Optional[float].
                the std age of the patient.
            method: str.
                the method of dimension reduction, one of "PCA" and "TSNE".
            dimension: int.
                the dimension of dimension reduction, one of 2 and 3.
            target: str.
                the target of the model, one of "outcome", "los" and "multitask".

        Returns:
            List.
                the data to draw dimension reduction.        
        """
        num = len(x)
        patients = []
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
            
            patient = []
            if dimension == 2:  # 判断降维维度
                patient.append({'name': 'data', 'value': [list(x) for x in zip(reduction_model[:, 0], reduction_model[:, 1], y_hat)]})
            elif dimension == 3:
                patient.append({'name': 'data', 'value': [list(x) for x in zip(reduction_model[:, 0], reduction_model[:, 1], reduction_model[:, 2], y_hat)]})
            patient.append({'name': 'patient_id', 'value': pidi.item()})
            patient.append({'name': 'record_time', 'value': [str(x) for x in timei]})
            if std_age is not None and mean_age is not None:
                patient.append({'name': 'age', 'value': int(xi[0][0][1].item() * std_age + mean_age)})
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