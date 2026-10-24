
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from .config import Config
from .preprocessing import MEAN, STD

TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))])


@dataclass
class DataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    sizes: dict[str, int]


def build_dataloaders(cfg: Config) -> DataLoaders:
    d, t = cfg.data, cfg.training
    full_train = datasets.FashionMNIST(d.data_dir, train=True, download=True, transform=TRANSFORM)
    test = datasets.FashionMNIST(d.data_dir, train=False, download=True, transform=TRANSFORM)

    split_gen = torch.Generator().manual_seed(t.seed)
    n_val = int(round(len(full_train) * d.val_split))
    train, val = random_split(full_train, [len(full_train) - n_val, n_val], generator=split_gen)

    common = dict(
        batch_size=t.batch_size,
        num_workers=d.num_workers,
        pin_memory=d.pin_memory,
        persistent_workers=d.num_workers > 0,
    )
    loader_gen = torch.Generator().manual_seed(t.seed)
    return DataLoaders(
        train=DataLoader(train, shuffle=True, generator=loader_gen, **common),
        val=DataLoader(val, shuffle=False, **common),
        test=DataLoader(test, shuffle=False, **common),
        sizes={"train": len(train), "val": len(val), "test": len(test)},
    )
