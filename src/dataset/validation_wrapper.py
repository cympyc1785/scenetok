from typing import Iterator, Optional

import torch
from torch.utils.data import Dataset, IterableDataset


class ValidationWrapper(Dataset):
    """Wraps a dataset so that PyTorch Lightning's validation step can be turned into a
    visualization step.
    """

    dataset: Dataset
    dataset_iterator: Optional[Iterator]
    length: int

    def __init__(self, dataset: Dataset, length: int) -> None:
        super().__init__()
        self.dataset = dataset
        self.length = length
        self.dataset_iterator = None

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        if isinstance(self.dataset, IterableDataset):
            if self.dataset_iterator is None:
                self.dataset_iterator = iter(self.dataset)
            try:
                return next(self.dataset_iterator)
            except StopIteration:
                self.dataset_iterator = iter(self.dataset)
            return next(self.dataset_iterator)

        dataset_length = len(self.dataset)
        if dataset_length == 0:
            raise ValueError(
                "ValidationWrapper received an empty dataset. "
                "Check the validation split path, meta.csv, and dataset filtering options."
            )

        # random_index = torch.randint(0, dataset_length, tuple())

        return self.dataset[index]
