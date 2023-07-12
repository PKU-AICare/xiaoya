import torch
import lightning as L
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import List

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

    def get_importance_scores(self, x) -> torch.Tensor:
        # config
        config = self.pipeline.config
        config['model'] = 'MHAGRU'
    
        # train/val/test
        pipeline = DlPipeline(config)
        pipeline = pipeline.load_from_checkpoint(self.model_path)
        y_hat, embedding, scores = pipeline.predict_step(x)
        return scores

    def data_dimension_reduction(self,x,method,dimension,target)-> List:
        config = self.pipeline.config
        pipeline = DlPipeline(config)
        pipeline = pipeline.load_from_checkpoint(self.model_path)
        y_hat, embedding, scores = pipeline.predict_step(x)
        array = embedding.detach().numpy()
        array_flat=array[0]
        df = pd.DataFrame(array_flat)
        if method == "PCA":  # 判断降维类别
            reduction_model = PCA().fit_transform(df)
        elif method == "TSNE":
            reduction_model = TSNE(n_components=dimension).fit_transform(df)
        if target == "Outcome":
            y_hat = y_hat.detach().numpy()[0][:,0].flatten().tolist()
        else:
            y_hat = y_hat.detach().numpy()[0][:,1].flatten().tolist()   
        if dimension == 2:  # 判断降维维度
            df_subset = pd.DataFrame(
                {
                    "2d-one": reduction_model[:, 0],
                    "2d-two": reduction_model[:, 1],
                    "target": y_hat,
                }
            )
            return list(zip([df_subset["2d-one"].tolist(), df_subset["2d-two"].tolist(), df_subset["target"].tolist()]))
        elif dimension == 3:
            df_subset = pd.DataFrame(
                {
                    "3d-one": reduction_model[:, 0],
                    "3d-two": reduction_model[:, 1],
                    "3d-three": reduction_model[:, 2],
                    "target": y_hat,
                }
            )
            return list(zip([df_subset["3d-one"].tolist(), df_subset["3d-two"].tolist(), df_subset["3d-three"].tolist(), df_subset["target"].tolist()]))
        res = {"message": "reduction fail"}
        return res