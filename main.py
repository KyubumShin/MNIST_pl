import argparse

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from dataset import MNISTDataloader
from model import MNISTModel


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", type=str, default="cross_entropy")
    parser.add_argument("--lr", type=float, default=1e-2)
    args = parser.parse_args()
    return args


def main():
    args = vars(get_args())
    dataloader = MNISTDataloader("./data")
    model = MNISTModel(**args)
    wandb_logger = WandbLogger(project="mnist_test")
    trainer = Trainer(
        logger=wandb_logger,
        max_epochs=10,
        gpus=1,
        accumulate_grad_batches=1,
        fast_dev_run=False,
        precision=16
    )
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    main()
