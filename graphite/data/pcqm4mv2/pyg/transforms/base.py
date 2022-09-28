from torch_geometric.data import Data
import abc


class BasePygGraphitePCQM4MTransform(metaclass=abc.ABCMeta):
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abc.abstractmethod
    def forward(self, g: Data) -> Data:
        pass
