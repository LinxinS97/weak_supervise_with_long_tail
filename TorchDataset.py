import bisect
import warnings

from torch import Tensor, IntTensor
from torch.utils.data import Dataset, IterableDataset
from typing import List, TypeVar, Optional, Tuple

T = TypeVar('T')


class TorchDataset(Dataset[Tuple[Tensor]]):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Args:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """
    tensors: Tuple[Tensor]

    def __init__(self, tensors: Tuple[Tensor, ...]) -> None:
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)

    def update_unlabeled(self, unlabeled_idx: IntTensor, new_label: Tensor, label_idx: Optional[int] = 3) -> None:
        self.tensors[label_idx][unlabeled_idx] = new_label


class ConcatTorchDataset(Dataset):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets (sequence): List of datasets to be concatenated
    """
    datasets: List[TorchDataset]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets) -> None:
        super(ConcatTorchDataset, self).__init__()
        # Cannot verify that datasets is Sized
        assert len(datasets) > 0, 'datasets should not be an empty iterable'  # type: ignore
        self.datasets = list(datasets)
        for d in self.datasets:
            assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def _get_idx(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return dataset_idx, sample_idx

    def __getitem__(self, idx):
        dataset_idx, sample_idx = self._get_idx(idx)
        return self.datasets[dataset_idx][sample_idx]

    def update_label(self, unlabeled_indices: IntTensor, new_labels: Tensor, label_idx: int) -> None:
        for idx, unlabeled_idx in enumerate(unlabeled_indices[0]):
            dataset_idx, sample_idx = self._get_idx(unlabeled_idx)
            self.datasets[dataset_idx].update_unlabeled(sample_idx, new_labels[0][idx], label_idx)
        pass

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes
