import tensorflow as tf

from models.image_model_v15 import ImageModelVersion as IV15
from models.image_model_v28 import ImageModelVersion as IV28
from models.image_model_v16 import ImageModelVersion as IV16
from models.image_model_v31 import ImageModelVersion as IV31


DEBUG = True

class ImageModelAdapter:
    def __init__(self, version=10):
        self.version = version
        self.model = None
        if(self.version == 15):
            self.printDebug("=====15=====")
            self.model = IV15(9)
        elif(self.version == 16):
            self.printDebug("=====16=====")
            self.model = IV16(9)
        elif(self.version == 31):
            self.printDebug("=====31=====")
            self.model = IV31(4)
        else:
            self.printDebug("=====28=====")
            self.model = IV28(4)
        

    def getModel(self):
        return self.model

    def printDebug(self, message):
        if(DEBUG):
            print(message)
