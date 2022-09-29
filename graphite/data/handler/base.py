import abc


class DataHandlerBase(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractmethod
    def get_dataloaders(self):
        pass
