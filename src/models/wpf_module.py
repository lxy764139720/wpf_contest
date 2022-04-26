from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MinMetric
from torchmetrics.regression.mse import MeanSquaredError
from torchmetrics.regression.mae import MeanAbsoluteError

from src.models.components.gru import GRUNet
from src.utils.loss import wpf_loss


class WPFModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        lr: float = 0.0001,
        weight_decay: float = 0,
        task: str = 'MS',
        train_bar: bool = True,
        val_bar: bool = True,
        test_bar: bool = True,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function, used for backpropagation
        self.criterion = wpf_loss
        # metrix, used for calculate metrix
        self.train_rmse = MeanSquaredError(squared=False)
        self.train_mae = MeanAbsoluteError()
        self.val_rmse = MeanSquaredError(squared=False)
        self.val_mae = MeanAbsoluteError()
        self.test_rmse = MeanSquaredError(squared=False)
        self.test_mae = MeanAbsoluteError()

        # for logging best so far validation loss
        self.val_loss_best = MinMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def step(self, batch: Any):
        x, y = batch
        y_hat = self.forward(x)
        # If the task is the multivariate-to-univariate forecasting task,
        # the last column is the target variable to be predicted
        f_dim = -1 if self.hparams.task == 'MS' else 0
        y = y[:, -self.net.get_output_len():, f_dim:]
        y_hat = y_hat[..., :, f_dim:]
        loss = self.criterion(y_hat, y)
        return loss, y, y_hat

    def training_step(self, batch: Any, batch_idx: int):
        loss, targets, predicts = self.step(batch)

        # log train metrics
        rmse = self.train_rmse(targets, predicts)
        mae = self.train_mae(targets, predicts)
        avg_loss = (rmse + mae) / 2
        self.log("train/loss", avg_loss, on_step=False, on_epoch=True, prog_bar=self.hparams.train_bar)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        # here return 'loss' not 'avg_loss' for backpropagation
        return {"loss": loss, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, targets, predicts = self.step(batch)

        # log val metrics
        rmse = self.val_rmse(targets, predicts)
        mae = self.val_mae(targets, predicts)
        avg_loss = (rmse + mae) / 2
        self.log("val/loss", avg_loss, on_step=False, on_epoch=True, prog_bar=self.hparams.val_bar)

        return {"loss": avg_loss, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        rmse = self.val_rmse.compute()  # get val loss from current epoch
        mae = self.val_mae.compute()
        self.val_loss_best.update((rmse + mae) / 2)
        self.log("val/loss_best", self.val_loss_best.compute(), on_epoch=True, prog_bar=self.hparams.val_bar)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        acc = self.test_acc(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=self.hparams.test_bar)
        self.log("test/acc", acc, on_step=False, on_epoch=self.hparams.test_bar)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        self.train_rmse.reset()
        self.test_rmse.reset()
        self.val_rmse.reset()
        self.train_mae.reset()
        self.test_mae.reset()
        self.val_mae.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
