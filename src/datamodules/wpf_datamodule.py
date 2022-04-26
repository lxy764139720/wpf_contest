from typing import Optional
import os

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

from .components.wpf_dataset import WPFDataset


class WPFDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
            self,
            data_dir: str = "data/",
            feature_file: str = "sdwpf_baidukddcup2022_full.CSV",
            position_file: str = "sdwpf_baidukddcup2022_turb_location.CSV",
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
            day_len: int = 144,
            train_days: int = 153,
            val_days: int = 16,
            test_days: int = 15,
            total_days: int = 184,
            turbine_id: int = 0,
            task: str = "MS",
            target: str = "Patv",
            scale: bool = True,
            start_col: int = 3,
            input_len: int = 144,
            output_len: int = 288,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[WPFDataset] = None
        self.data_val: Optional[WPFDataset] = None
        self.data_test: Optional[WPFDataset] = None
        self.raw_data_train: Optional[pd.DataFrame] = None
        self.raw_data_val: Optional[pd.DataFrame] = None
        self.raw_data_test: Optional[pd.DataFrame] = None
        self.scaler = None

        self.tid: int = turbine_id
        self.total_size: int = total_days * day_len
        self.train_size: int = train_days * day_len
        self.val_size: int = val_days * day_len
        self.test_size: int = test_days * day_len

    def prepare_data(self):
        """Download data if needed.

        This method is called only from a single GPU.
        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """

        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.scaler = StandardScaler()
            df_raw = pd.read_csv(os.path.join(self.hparams.data_dir, self.hparams.feature_file))
            # [border1s[0]: border2s[0]]: train_data (153*24*6, feature)
            # [border1s[1]: border2s[1]]: val_data (16*24*6, feature)
            # [border1s[2]: border2s[2]]: test_data (15*24*6, feature)
            border1s = [self.tid * self.total_size,
                        self.tid * self.total_size + self.train_size - self.hparams.input_len,
                        self.tid * self.total_size + self.train_size + self.val_size - self.hparams.input_len]
            border2s = [self.tid * self.total_size + self.train_size,
                        self.tid * self.total_size + self.train_size + self.val_size,
                        self.tid * self.total_size + self.train_size + self.val_size + self.test_size]

            df_data = df_raw
            if self.hparams.task == 'M':
                cols_data = df_raw.columns[self.hparams.start_col:]
                df_data = df_raw[cols_data]  # feature=10
            elif self.hparams.task == 'MS':
                cols_data = df_raw.columns[self.hparams.start_col:]
                df_data = df_raw[cols_data]  # feature=10
            elif self.hparams.task == 'S':
                df_data = df_raw[[self.tid, self.hparams.target]]  # feature=2

            # Turn off the SettingWithCopyWarning
            pd.set_option('mode.chained_assignment', None)
            df_data.replace(to_replace=np.nan, value=0, inplace=True)

            if self.hparams.scale:
                train_data = df_data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data.values)
                data = self.scaler.transform(df_data.values)
            else:
                data = df_data.values

            type_map = {'train': 0, 'val': 1, 'test': 2}
            if stage == 'fit' or stage is None:
                border1, border2 = border1s[type_map['train']], border2s[type_map['train']]
                self.data_train = WPFDataset(data_type='train',
                                             data=data[border1:border2],
                                             input_len=self.hparams.input_len,
                                             output_len=self.hparams.output_len)
                self.raw_data_train = df_data[border1 + self.hparams.input_len:border2]
                border1, border2 = border1s[type_map['val']], border2s[type_map['val']]
                self.data_val = WPFDataset(data_type='val',
                                           data=data[border1:border2],
                                           input_len=self.hparams.input_len,
                                           output_len=self.hparams.output_len)
                self.raw_data_val = df_data[border1 + self.hparams.input_len:border2]
            if stage == 'test' or stage is None:
                border1, border2 = border1s[type_map['test']], border2s[type_map['test']]
                self.data_test = WPFDataset(data_type='test',
                                            data=data[border1:border2],
                                            input_len=self.hparams.input_len,
                                            output_len=self.hparams.output_len)
                self.raw_data_test = df_data[border1 + self.hparams.input_len:border2]

    def get_raw_data_test(self):
        return self.raw_data_test

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
