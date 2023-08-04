from typing import Any, Optional

import hydra
from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule
from omegaconf import DictConfig

class PBRDataModule(LightningDataModule):
    """A universal `LightningDataModule` for PBR dataset.
    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```
    """
    
    def __init__(
        self,
        batch_size: int, 
        num_workers: int = 0,
        pin_memory: bool = False,
        cfg_train: DictConfig = None, 
        cfg_val: DictConfig = None,
        cfg_test: DictConfig = None,
    ) -> None:
        """Initialize a `PBRDataModule`.
        
        :param batch_size: The batch size.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        :param cfg_train: The config of train dataset. Defaults to `None`.
        :param cfg_val: The config of validation dataset. Defaults to `None`.
        :param cfg_test: The config of of test dataset. Defaults to `None`.
        """
        super().__init__()
        
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
    
    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        # already processed
        pass
        
    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # load dataset
        if stage == 'fit' or stage == None:
            self.data_train = hydra.utils.instantiate(self.hparams.cfg_train)
            self.data_val = hydra.utils.instantiate(self.hparams.cfg_val)
        
        if stage == 'test':
            self.data_test = hydra.utils.instantiate(self.hparams.cfg_test)
        
    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.
        
        :return: The train dataloader
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )
        
    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.
        
        :return: The validation dataloader
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
        
    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.
        
        :return: The test dataloader
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
    
    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass