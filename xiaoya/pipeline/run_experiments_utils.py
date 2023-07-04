import os

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import torch

from pyehr.dataloaders.utils import get_los_info
from pyehr.dataloaders import EhrDataModule
from pyehr.pipelines import DlPipeline


def model_train(config: dict, data_url: str, ckpts_url: str) -> str:
    """
    Train the model.

    Args:
        config: dict.
            the config of the model.
        data_url: str.
            the url of the data.
        ckpts_url: str.
            the url to save the checkpoints.

    Returns:
        best_model_path: str.
            the path of the best model.
    """

    los_config = get_los_info(data_url)

    main_metric = 'auprc' if config['task'] in ['outcome', 'multitask'] else 'mae'
    mode = 'max' if config['task'] in ['outcome', 'multitask'] else 'min'

    config.update({'los_info': los_config, 'main_metric': main_metric})
    
    # data
    dm = EhrDataModule(data_url, batch_size=config['batch_size'])

    # checkpoint 
    ckpts_url = os.path.join(ckpts_url, config['task'], f"{config['model']}-seed{config['seed']}")

    # EarlyStop and checkpoint callback
    early_stopping_callback = EarlyStopping(monitor=main_metric, patience=config['patience'], mode=mode)
    checkpoint_callback = ModelCheckpoint(monitor=main_metric, mode=mode, dirpath=ckpts_url, 
                                        filename="best")


    # seed for reproducibility
    L.seed_everything(config['seed'])

    # device
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

    # train/val/test
    pipeline = DlPipeline(config)
    trainer = L.Trainer(accelerator=accelerator, max_epochs=config['epochs'],
                        callbacks=[early_stopping_callback, checkpoint_callback], logger=False,
                        enable_progress_bar=True)
    trainer.fit(pipeline, datamodule=dm)
    return checkpoint_callback.best_model_path


def model_predict(config: dict, data_url: str, ckpts_url: str):
    """
    Use the best model to predict.

    Args:
        config: dict.
            the config of the model.
        data_url: str.
            the url of the data.
        ckpts_url: str.
            the url to save the checkpoints.
    """

    los_config = get_los_info(data_url)
    config.update({"los_info": los_config})

    # data
    dm = EhrDataModule(data_url, batch_size=config['batch_size'])

    # device
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

    # train/val/test
    pipeline = DlPipeline(config)
    trainer = L.Trainer(accelerator=accelerator, max_epochs=1, logger=False, num_sanity_val_steps=0)
    trainer.test(pipeline, datamodule=dm, ckpt_path=ckpts_url)
    # perf = pipeline.test_performance
    return pipeline.test_performance
