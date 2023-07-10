import torch
import lightning as L

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
