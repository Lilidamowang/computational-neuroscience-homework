import argparse

class Args:
   
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--device", type=str, default='cuda:0', help="CPU or GPU")
        parser.add_argument("--model_load_path", type=str, default='/home/data/yjgroup/lyl/result/RIM_PR_model/OKVQA/t=2/checkpoint_6.pth')
        parser.add_argument("--train", type=int, default=1, help='whether training')
        parser.add_argument("--model", type=str, default='Test1')
        parser.add_argument("--model_type", type=int, default=6)
        parser.add_argument("--save_path", type=str, default='/home/data/yjgroup/lyl/result/RIM_PR_model/OKVQA/t=2')

        self.args = parser.parse_args()

    def getDevice(self):
        return self.args.device
    
    def getModelLoadPath(self):
        return self.args.model_load_path

    def getTrain(self):
        return self.args.train
    
    def getModel(self):
        return self.args.model
    
    def getModelType(self):
        return self.args.model_type
    
    def getSavePath(self):
        return self.args.save_path
