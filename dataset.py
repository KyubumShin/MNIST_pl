import pytorch_lightning as pl
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST


class MNISTDataloader(pl.LightningDataModule):
    def __init__(self, data_dir, num_worker=8, batch_size=32):
        super(MNISTDataloader, self).__init__()
        self.data_dir = data_dir
        self.num_worker = num_worker
        self.batch_size = batch_size

    @property
    def transform(self):
        return T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self) -> None:
        MNIST(self.data_dir, download=True)

    def train_dataloader(self):
        train_dataset = MNIST(self.data_dir, train=True, download=False, transform=self.transform)
        return DataLoader(train_dataset, num_workers=self.num_worker, batch_size=self.batch_size)

    def val_dataloader(self):
        test_dataset = MNIST(self.data_dir, train=False, download=False, transform=self.transform)
        return DataLoader(test_dataset, num_workers=self.num_worker, batch_size=self.batch_size)
