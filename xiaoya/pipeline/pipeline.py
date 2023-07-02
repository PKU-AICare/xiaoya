import os
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from pyehr.dataloaders import EhrDataModule
from pyehr.pipelines import DlPipeline
from pyehr.dataloaders.utils import get_los_info

class Pipeline:
    """
    Pipeline

    Args:
        model: str.
            the model to use, Available model:
                - LSTM
                - GRU
                - AdaCare
                - RNN
                - MLP
                - MHAGRU, for feature importance
        task: str. 
            the task.
                - multitask
                - outcome
                - los

    """
    def __init__(self,
                dataset: pd.DataFrame,
                model: str = 'GRU',
                batch_size: int = 64,
                learning_rate: float = 0.001,
                hidden_dim: int = 32,
                epochs: int = 100,
                patience: int = 10,
                task: str = 'multitask',
                seed: int = 42,
                use_pretrain_model: bool = False,
                data_path: str = Path('./datasets'),
                ckpt_path: str = Path('./checkpoints'),
            ) -> None:
        
        self.dataset = pd.DataFrame(dataset)
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
            'use_pretrain_model': use_pretrain_model,

            'demo_dim': 2,
            'lab_dim': 0,
        }
        self.data_path = data_path
        self.ckpt_path = ckpt_path
        self.los_info = get_los_info(data_path)

    def train(self):
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
        trainer = L.Trainer(accelerator=accelerator, max_epochs=self.config['epochs'],
                            callbacks=[early_stopping_callback, checkpoint_callback], logger=False,
                            enable_progress_bar=True)
        trainer.fit(pipeline, datamodule=dm)
        self.ckpt_best_url = checkpoint_callback.best_model_path

    def predict(self):
        self.config.update({'los_info': self.los_info})

        # data
        dm = EhrDataModule(self.data_path, batch_size=self.config['batch_size'])

        # device
        accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

        # train/val/test
        pipeline = DlPipeline(self.config)
        trainer = L.Trainer(accelerator=accelerator, max_epochs=1, logger=False, num_sanity_val_steps=0)
        trainer.test(pipeline, datamodule=dm, ckpt_path=self.ckpt_best_url)

        # perf = pipeline.test_performance
        self.test_performance = pipeline.test_performance

    def feature_importance(self):
        self.config.update({'los_info': self.los_info, 'model': 'MHAGRU'})

        # data
        dm = EhrDataModule(self.data_path, batch_size=self.config['batch_size'])

        # device
        accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    
        # train/val/test
        pipeline = DlPipeline(self.config)
        trainer = L.Trainer(accelerator=accelerator, max_epochs=1, logger=False, num_sanity_val_steps=0)
        trainer.test(pipeline, datamodule=dm, ckpt_path=self.ckpt_best_url)