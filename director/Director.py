
class Director:
    def makeOperation(self, builder):
        builder.setModel()
        builder.setOption()
        builder.setTestDataset()
        builder.setTrainDataset()
        builder.setSavePath()

    def makeTest(self, builder):
        builder.setDataset()
        builder.setOption()