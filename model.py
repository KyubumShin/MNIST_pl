import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch
from torchmetrics import Accuracy

from loss import create_criterion


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), (1, 1)),
            nn.BatchNorm2d(32),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3), (1, 1)),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),
            nn.ReLU(),
        )
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        x = self.feature(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class MNISTModel(pl.LightningModule):
    def __init__(self, loss, lr):
        super(MNISTModel, self).__init__()
        self.net = Backbone()
        self._criterion = create_criterion(loss)
        self.acc = Accuracy()
        self.learning_rate = lr
        self.save_hyperparameters(ignore="model")

    def forward(self, x):
        x = self.net(x)
        x = F.softmax(x, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        preds, loss, acc, labels = self.__share_step(batch, 'train')
        self.log("train_loss", loss)
        self.log("train_accuracy", acc)
        return {"loss": loss, "pred": preds.detach(), 'labels': labels.detach()}

    def validation_step(self, batch, batch_idx):
        preds, loss, acc, labels = self.__share_step(batch, 'val')
        self.log("val_loss", loss)
        self.log("val_accuracy", acc)
        return {"loss": loss, "pred": preds, 'labels': labels}

    def __share_step(self, batch, mode):
        x, y = batch
        y_hat = self.net(x)
        loss = self._criterion(y_hat, y)
        acc = self.acc(y_hat, y)
        return y_hat, loss, acc, y

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), lr=self.learning_rate)
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=10, gamma=0.5
        )
        return [optimizer], [scheduler]
