from typing import List, Dict 

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
        pipeline: Pipeline.
            the pipeline.
        model_path: str.
            the path of the model.
    """

    def __init__(self, 
        pipeline: Pipeline,
        model_path: str,
    ) -> None:
        self.pipeline = pipeline
        self.model_path = model_path

    def importance_scores(
            self, 
            x: torch.Tensor
        ) -> Dict:
        config = self.pipeline.config
        config['model'] = 'MHAGRU'
    
        pipeline = DlPipeline(config)
        pipeline = pipeline.load_from_checkpoint(self.model_path)
        _, _, scores = pipeline.predict_step(x)

        for key in scores:
            scores[key] = scores[key].cpu().detach().numpy() if isinstance(scores[key], torch.Tensor) else scores[key]
        return scores
    
    def feature_importance(
            self,
            df: pd.DataFrame,
            x: List,
            patientID: int,
        ) -> Dict:
        """
        Return feature importance of a patient.

        Args:
            df: pd.DataFrame.
                the dataframe of the patient.
            x: List.
                the input of the patient.
            patientID: int.
                the patient ID.

        Returns:
            Dict.
                the feature importance.
        """

        xid = list(df['PatientID'].drop_duplicates()).index(patientID)        
        x = torch.Tensor(x[xid]).unsqueeze(0)   # [1, ts, f]
        scores = self.importance_scores(x.to('cuda:0'))
        column_names = list(df.columns[6:])
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
            mask: torch.Tensor,
            patientID: int,
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
            patientID: int.
                the patient ID.

        Returns:
            Dict.
                the data to draw risk curve.
        """

        xid = list(df['PatientID'].drop_duplicates()).index(patientID)        
        x = torch.Tensor(x[xid]).unsqueeze(0)   # [1, ts, f]
        mask = mask[xid]                        # [ts, f]
        scores = self.importance_scores(x.to('cuda:0'))
        
        record_times = list(item[1] for item in df[df['PatientID'] == patientID]['RecordTime'].items())
        column_names = list(df.columns[6:])

        return {
            'detail': [{
                'name': column_names[i],
                'value': x[0][:, i],
                'time_step_feature_importance': scores['time_step_feature_importance'][0][:, i],
                'missing': mask[:, i],
                'unit': ''
            } for i in range(len(column_names))],
            'time': record_times,   # ts
            'time_step_importance': scores['time_step_importance'][0],  # ts
        }
    
    def ai_advice(self,
            input: torch.Tensor,
            time_index: int,
        ) -> List:
        # x: [batch_size, seq_len, feature_dim]
        config = self.pipeline.config
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
            method: str = "PCA",
            dimension: int = 2,
            target: str = "multitask",
        )-> List:
        """
        Return data to draw dimension reduction.

        Args:
            method: one of {"PCA", "TSNE"}.
            dimension: one of {2, 3}.
            target: one of {"multitask", "los", "outcome"}.

        Returns:
            List.
                the data to draw dimension reduction.        
        """
        x = pd.read_pickle('datasets/test_x.pkl')
        y = pd.read_pickle('datasets/test_pid.pkl')
        z = pd.read_pickle('datasets/test_record_time.pkl')
        # mean = pd.read_pickle('datasets/test_mean.pkl')['Age']
        # std = pd.read_pickle('datasets/test_std.pkl')['Age']
        num = len(x)
        patients = []
        for i in range(num):
            xi = torch.tensor(x[i]).unsqueeze(0).to('cuda:0')   # cuda
            yi = torch.tensor(y[i]).unsqueeze(0)
            zi = z[i]
            config = self.pipeline.config
            pipeline = DlPipeline(config)
            pipeline = pipeline.load_from_checkpoint(self.model_path)
            y_hat, embedding, _ = pipeline.predict_step(xi)
            embedding = embedding.cpu().detach().numpy().squeeze()  # cpu
            df = pd.DataFrame(embedding)
            if method == "PCA":  # 判断降维类别
                reduction_model = PCA().fit_transform(df)
            elif method == "TSNE":
                reduction_model = TSNE(n_components=dimension, learning_rate='auto', init='random').fit_transform(df)
            if(reduction_model.shape[0] != reduction_model.shape[1]):
                continue
            y_hat = y_hat.cpu().detach().numpy().squeeze()      # cpu
            if target == "outcome":
                y_hat = y_hat[:, 0].flatten().tolist()
            else:
                y_hat = y_hat[:, 1].flatten().tolist()   
            
            patient = []
            if dimension == 2:  # 判断降维维度
                patient.append({'name': 'data', 'value': [list(x) for x in zip(reduction_model[:, 0], reduction_model[:, 1], y_hat)]})
            elif dimension == 3:
                patient.append({'name': 'data', 'value': [list(x) for x in zip(reduction_model[:, 0], reduction_model[:, 1], reduction_model[:, 2], y_hat)]})
            patient.append({'name': 'patient_id', 'value': yi.item()})
            patient.append({'name': 'record_time', 'value': [str(x) for x in zi]})
            # result['Age'] = xi[0][0][1].item() * std + mean
            # patient.append({'name': 'age', 'value': xi[0][0][1].item() * std + mean})
            patients.append(patient)
        return {'detail': patients}
    