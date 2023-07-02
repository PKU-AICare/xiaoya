import os
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from .processed_datasets_utils import (
    normalize_dataframe,
    forward_fill_pipeline,
)
from pyehr.dataloaders import EhrDataModule
from pyehr.pipelines import DlPipeline

class Pipeline:
    """
    Pipeline

    Args:
    model: str ['LSTM', 'GRU', 'AdaCare', 'RNN', 'MLP']
    task: str ['multitask', 'outcome', 'los']

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
                train: int = 70,
                val: int = 10,
                test: int = 20,
                use_pretrain_model: bool = False,
                data_path: str = os.path.join(Path(__file__).resolve().parent.parent, 'data'),
                ckpt_path: str = os.path.join(Path(__file__).resolve().parent.parent, 'checkpoints'),
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
        Path(data_path).mkdir(parents=True, exist_ok=True)
        self.ckpt_path = ckpt_path

        demographic_features = ['Sex', 'Age']
        labtest_features = []

        # Group the dataframe by patient ID
        grouped = self.dataset.groupby('PatientID')
        patients = np.array(list(grouped.groups.keys()))
        
        # Get the train_val/test patient IDs
        patients_outcome = np.array([grouped.get_group(patient_id)['Outcome'].iloc[0] for patient_id in patients])
        train_val_patients, test_patients = train_test_split(patients, test_size=test/(train+val+test), random_state=seed, stratify=patients_outcome)

        # Get the train/val patient IDs
        train_val_patients_outcome = np.array([grouped.get_group(patient_id)['Outcome'].iloc[0] for patient_id in train_val_patients])
        train_patients, val_patients = train_test_split(train_val_patients, test_size=val/(train+val), random_state=seed, stratify=train_val_patients_outcome)

        #  Create train, val, test, [traincal, calib] dataframes for the current fold
        self.train_raw_df = self.dataset[self.dataset['PatientID'].isin(train_patients)]
        self.val_raw_df = self.dataset[self.dataset['PatientID'].isin(val_patients)]
        self.test_raw_df = self.dataset[self.dataset['PatientID'].isin(test_patients)]

        # Calculate the mean and std of the train set (include age, lab test features, and LOS) on the data in 5% to 95% quantile range
        normalize_features = ['Age'] + labtest_features + ['LOS']
        self.train_after_zscore, self.val_after_zscore, self.test_after_zscore, default_fill, self.los_info, _, _ = \
            normalize_dataframe(self.train_raw_df, self.val_raw_df, self.test_raw_df, normalize_features)

        # Drop rows if all features are recorded NaN
        self.train_after_zscore = self.train_after_zscore.dropna(axis=0, how='all', subset=normalize_features)
        self.val_after_zscore = self.val_after_zscore.dropna(axis=0, how='all', subset=normalize_features)
        self.test_after_zscore = self.test_after_zscore.dropna(axis=0, how='all', subset=normalize_features)

        # Forward Imputation after grouped by PatientID
        self.train_x, self.train_y, self.train_pid = forward_fill_pipeline(self.train_after_zscore, default_fill, demographic_features, labtest_features)
        self.val_x, self.val_y, self.val_pid = forward_fill_pipeline(self.val_after_zscore, default_fill, demographic_features, labtest_features)
        self.test_x, self.test_y, self.test_pid = forward_fill_pipeline(self.test_after_zscore, default_fill, demographic_features, labtest_features)

        # Save the dataframes
        self.train_raw_df.to_csv(os.path.join(data_path, "train_raw.csv"), index=False)
        self.val_raw_df.to_csv(os.path.join(data_path, "val_raw.csv"), index=False)
        self.test_raw_df.to_csv(os.path.join(data_path, "test_raw.csv"), index=False)

        self.train_after_zscore.to_csv(os.path.join(data_path, "train_after_zscore.csv"), index=False)
        self.val_after_zscore.to_csv(os.path.join(data_path, "val_after_zscore.csv"), index=False)
        self.test_after_zscore.to_csv(os.path.join(data_path, "test_after_zscore.csv"), index=False)

        pd.to_pickle(self.train_x, os.path.join(data_path, "train_x.pkl"))
        pd.to_pickle(self.train_y, os.path.join(data_path, "train_y.pkl"))
        pd.to_pickle(self.train_pid, os.path.join(data_path, "train_pid.pkl"))
        pd.to_pickle(self.val_x, os.path.join(data_path, "val_x.pkl"))
        pd.to_pickle(self.val_y, os.path.join(data_path, "val_y.pkl"))
        pd.to_pickle(self.val_pid, os.path.join(data_path, "val_pid.pkl"))
        pd.to_pickle(self.test_x, os.path.join(data_path, "test_x.pkl"))
        pd.to_pickle(self.test_y, os.path.join(data_path, "test_y.pkl"))
        pd.to_pickle(self.test_pid, os.path.join(data_path, "test_pid.pkl"))

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
        self.config.update({'los_info': self.los_info, 'model': 'TGRU'})

        # data
        dm = EhrDataModule(self.data_path, batch_size=self.config['batch_size'])

        # device
        accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    
        # train/val/test
        pipeline = DlPipeline(self.config)
        trainer = L.Trainer(accelerator=accelerator, max_epochs=1, logger=False, num_sanity_val_steps=0)
        trainer.test(pipeline, datamodule=dm, ckpt_path=self.ckpt_best_url)
