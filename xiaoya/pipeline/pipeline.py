import os
from pathlib import Path
from typing import List, Tuple

import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd

from xiaoya.pyehr.dataloaders import EhrDataModule
from xiaoya.pyehr.pipelines import DlPipeline
from xiaoya.pyehr.dataloaders.utils import get_los_info

class Pipeline:
    """
    Pipeline

    Args:
        model: str.
            the model to use, available models:
                - LSTM
                - GRU
                - AdaCare
                - RNN
                - MLP
        task: str. 
            the task, default is multitask, available tasks:
                - multitask
                - outcome
                - los
        batch_size: int.
            the batch size, default is 64.
        learning_rate: float.
            the learning rate, default is 0.001.
        hidden_dim: int.
            the hidden dimension, default is 32.
        epochs: int.
            the number of epochs, default is 100.
        patience: int.
            the patience for early stopping, default is 10.
        seed: int.
            the random seed, default is 42.
        pretrained_model_path: Path.
            the path of the pretrained model, default is None.
        data_path: Path.
            the path of the data, default is Path('./datasets').
        ckpt_path: Path.
            the path to save the checkpoints, default is Path('./checkpoints').
        demographic_dim: int.
            the dimension of the demographic features.
        labtest_dim: int.
            the dimension of the labtest features.
    """

    def __init__(self,
            model: str = 'GRU',
            batch_size: int = 64,
            learning_rate: float = 0.001,
            hidden_dim: int = 32,
            epochs: int = 10,
            patience: int = 10,
            task: str = 'multitask',
            seed: int = 42,
            data_path: Path = Path('./datasets'),
            ckpt_path: Path = Path('./checkpoints'),
            metric_path: Path = Path('./metrics'),
            demographic_dim: int = 2,
            labtest_dim: int = 73
        ) -> None:
        
        self.config = {
            'model': model,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'hidden_dim': hidden_dim,
            'output_dim': 1,
            'epochs': epochs,
            'patience': patience,
            'task': task,
            'seed': seed,

            'demo_dim': demographic_dim,
            'lab_dim': labtest_dim,
        }
        self.data_path = data_path
        self.ckpt_path = ckpt_path
        self.metric_path = metric_path
        self.los_info = get_los_info(data_path)
        self.model_path = None

    def train(self) -> None:
        """
        Train the model.
        """

        main_metric = 'auprc' if self.config['task'] in ['outcome', 'multitask'] else 'mae'
        mode = 'max' if self.config['task'] in ['outcome', 'multitask'] else 'min'

        self.config.update({'los_info': self.los_info, 'main_metric': main_metric, 'mode': mode})

        # datamodule
        dm = EhrDataModule(data_path=self.data_path, batch_size=self.config['batch_size'])

        # checkpoint
        ckpt_url = os.path.join(self.ckpt_path, self.config['task'], f'{self.config["model"]}-seed{self.config["seed"]}')

        # EarlyStop and checkpoint callback
        early_stopping_callback = EarlyStopping(monitor=main_metric, patience=self.config['patience'], mode=mode)
        checkpoint_callback = ModelCheckpoint(monitor=main_metric, mode=mode, dirpath=ckpt_url, 
                                        filename='best')
        # seed
        L.seed_everything(self.config['seed'])

        # device
        accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

        pipeline = DlPipeline(self.config)
        trainer = L.Trainer(accelerator=accelerator, max_epochs=self.config['epochs'], callbacks=[early_stopping_callback, checkpoint_callback], logger=False, enable_progress_bar=True)
        trainer.fit(pipeline, datamodule=dm)
        self.model_path = checkpoint_callback.best_model_path

    def predict(self, model_path: str):
        """
        Use the best model to predict.

        Args:
            model_path: str.
                the path of the best model.

        Returns:
            dict: the performance of the model.
        """

        self.config.update({'los_info': self.los_info})

        # data
        dm = EhrDataModule(self.data_path, batch_size=self.config['batch_size'])

        # device
        accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

        # train/val/test
        pipeline = DlPipeline(self.config)
        trainer = L.Trainer(accelerator=accelerator, max_epochs=1, logger=False, num_sanity_val_steps=0)
        trainer.test(pipeline, datamodule=dm, ckpt_path=model_path)

        performance = {k: v.item() for k, v in pipeline.test_performance.items()}
        ckpt_name = Path(model_path).name.replace('ckpt', 'csv')
        metric_url = os.path.join(self.metric_path, self.config['task'], f'{self.config["model"]}-seed{self.config["seed"]}')
        Path(metric_url).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(performance, index=[0]).to_csv(os.path.join(metric_url, ckpt_name), index=False)

        output = pipeline.test_outputs
        return {'detail': {
            'preds': output['preds'],
            'labels': output['labels'],
            'config': self.config,
        }}

    def execute(self, model_path: str = None):
        """
        Execute the pipeline.

        Returns:
            dict: the performance of the model.
        """

        if model_path is None:
            self.train()
            model_path = self.model_path
        return self.predict(model_path)

    def get_history(self) -> List[Tuple[str, str, float]]:
        """
        Get the history of the model.

        Returns:
            dict: the history of the model.
        """

        ckpt_dir = os.listdir(self.ckpt_path)
        metric_dir = os.listdir(self.metric_path)

        result = []
        for item in zip(ckpt_dir, metric_dir, os.path.getctime(os.path.join(self.ckpt_path, item[0]))):
            result.append(item)
        return result
    