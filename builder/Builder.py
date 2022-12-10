import abc


class OperationBuilder(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def setModel(self):
        pass

    @abc.abstractmethod
    def setOption(self):
        pass

    @abc.abstractmethod
    def setTrainDataset(self):
        pass

    @abc.abstractmethod
    def setTestDataset(self):
        pass

    @abc.abstractmethod
    def setSavePath(self):
        pass

    @abc.abstractmethod
    def getOperation(self):
        pass


class TestBuilder(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def setDataset(self):
        pass

    @abc.abstractmethod
    def setOption(self):
        pass
