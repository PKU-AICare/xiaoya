from typing import List, Dict, Optional

import torch
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import optimize

from xiaoya.pyehr.pipelines import DlPipeline
from xiaoya.pipeline import Pipeline


class DataAnalyzer:
    """
    DataAnalyzer

    Args:
        config: Dict.
            the config of the pipeline.
        model_path: str.
            the path of the model.
    """

    def __init__(self, 
        config: Dict,
        model_path: str,
    ) -> None:
        self.config = config
        self.model_path = model_path

    def importance_scores(
            self, 
            x: torch.Tensor
        ) -> Dict:
        """
        Return the importance scores of a patient.

        Args:
            x: torch.Tensor.
                the input of the patient.

        Returns:
            Dict.
                the importance scores.
        """
        config = self.config
        config['model'] = 'MHAGRU'
    
        pipeline = DlPipeline(config)
        pipeline = pipeline.load_from_checkpoint(self.model_path)
        if pipeline.on_gpu():
            x = x.to('cuda:0')
        _, _, scores = pipeline.predict_step(x)

        for key in scores:
            scores[key] = scores[key].cpu().detach().numpy() if isinstance(scores[key], torch.Tensor) else scores[key]
        return scores
    
    def feature_importance(
            self,
            df: pd.DataFrame,
            x: List,
            patient_index: Optional[int],
            patient_id: Optional[int],
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
        xid = patient_index if patient_index is not None else list(df['PatientID'].drop_duplicates()).index(patient_id)        
        x = torch.Tensor(x[xid]).unsqueeze(0)   # [1, ts, f]
        scores = self.importance_scores(x)
        column_names = list(df.columns[4:])
        return {
            'detail': {
                'name': column_names,
                'value': scores['feature_importance'][0],   # feature importance value
            }
        }

    def risk_curve(
            self, 
            df: pd.DataFrame,
            x: List,
            mask: Optional[torch.Tensor],
            patient_index: Optional[int],
            patient_id: Optional[int],
        ) -> Dict:
        """
        Return data to draw risk curve of a patient.

        Args:
            df: pd.DataFrame.
                the dataframe of the patient.
            x: List.
                the input of the patient.
            mask: torch.Tensor.
                the missing mask of the patient.
            patient_index: Optional[int].
                the index of the patient in dataframe.
            patient_id: Optional[int].
                the patient ID.

        Returns:
            Dict.
                the data to draw risk curve.
        """
        if patient_index is not None:
            xid = patient_index
            patient_id = list(df['PatientID'].drop_duplicates())[patient_index]
        else:
            xid = list(df['PatientID'].drop_duplicates()).index(patient_id)   
        x = torch.Tensor(x[xid]).unsqueeze(0)   # [1, ts, f]
        mask = mask[xid] if mask is not None else None  # [ts, f]
        scores = self.importance_scores(x)
        
        column_names = list(df.columns[4:])
        record_times = list(item[1] for item in df[df['PatientID'] == patient_id]['RecordTime'].items()) 

        return {
            'detail': [{
                'name': column_names[i],
                'value': x[0][:, i],
                'time_step_feature_importance': scores['time_step_feature_importance'][0][:, i],
                'missing': mask[:, i] if mask is not None else None,
                'unit': ''
            } for i in range(len(column_names))],
            'time': record_times,   # ts
            'time_step_importance': scores['time_step_importance'][0],  # ts
            'feature_importance': scores['feature_importance'][0],
        }
    
    def ai_advice(self,
            input: torch.Tensor,
            time_index: int,
        ) -> List:
        """
        Return the advice of the AI system.

        Args:
            input: torch.Tensor.
                the input of the patient.
            time_index: int.
                the time index of the patient.

        Returns:
            List.
                the advice of the AI system.
        """
        # x: [batch_size, seq_len, feature_dim]
        config = self.config
        config['model'] = 'MHAGRU'

        pipeline = DlPipeline(config)
        pipeline = pipeline.load_from_checkpoint(self.model_path)

        # three most important labtest features in the last time step
        _, _, scores = pipeline.predict_step(input)
        # TODO: demo dim
        demo_dim = 2
        input_last_step = input[0][time_index].tolist()[demo_dim:]
        feature_last_step: List = scores['time_step_feature_importance'][0][time_index].tolist()[demo_dim:]

        index_dict = {index: value for index, value in enumerate(feature_last_step[demo_dim:]) if input_last_step[index] != 0}
        max_indices = sorted(index_dict, key=index_dict.get, reverse=True)
        if len(max_indices) > 3:
            max_indices = max_indices[:3]

        def f(x, args):
            input, i = args
            input[-1][-1][i] = torch.from_numpy(x).float()
            y_hat, _, _ = pipeline.predict_step(input)      # y_hat: [bs, seq_len, 2]
            return y_hat[0][time_index][0].detach().numpy()

        result = []
        for i in max_indices:
            print('index: ', i)
            x0 = float(input[-1][-1][i])
            bounds = (max(-3, x0 - 1), min(3, x0 + 1))
            args = (input, i)
            res = optimize.minimize(f, x0=x0, bounds=(bounds,), args=(args,), method='nelder-mead', options={'disp': True})
            result.append(res.x[0])
        return result

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
            config = self.config
            pipeline = DlPipeline(config)
            pipeline = pipeline.load_from_checkpoint(self.model_path)
            if pipeline.on_gpu():
                xi = xi.to('cuda:0')   # cuda
                y_hat, embedding, _ = pipeline.predict_step(xi)
                embedding = embedding.cpu().detach().numpy().squeeze()  # cpu
                y_hat = y_hat.cpu().detach().numpy().squeeze()      # cpu
            else:
                y_hat, embedding, _ = pipeline.predict_step(xi)
                embedding = embedding.detach().numpy().squeeze()
                y_hat = y_hat.detach().numpy().squeeze()
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
    