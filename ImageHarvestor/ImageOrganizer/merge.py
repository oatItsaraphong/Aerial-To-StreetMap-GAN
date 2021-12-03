from os import listdir
import random
import shutil
from PIL import Image

input = {
    "test": "./SplitSet/test/aerial",
    "train": "./SplitSet/train/aerial",
    "valid": "./SplitSet/validation/aerial"
}

#D:\School\Graduate\Project\ImageExtractor\images

# Executive decistionto use street jpg and turn to 3 channel at the end
out = {
    "test": "./Merge/test",
    "train": "./Merge/train",
    "valid": "./Merge/validate"
}


size = {
    "test": 500,
    "train": 2500,
    "valid": 500
}

def getfile(outPath,filesName):
    mainset = listdir(filesName)
    for file in mainset:
        path = filesName + "/"+file
        image1 = Image.open(path)
        path = path.replace("aerial","street")
        image2 = Image.open(path)
        image1_size = image1.size
        image2_size = image2.size
        new_image = Image.new('RGB',(2*image1_size[0], image1_size[1]))
        new_image.paste(image1,(0,0))
        new_image.paste(image2,(image1_size[0],0))
        new_image.save(outPath +"/" + file,"JPEG")

getfile(out["train"], input["train"])
getfile(out["test"], input["test"])
getfile(out["valid"], input["valid"])