from os import listdir
import random
import shutil

target = {
    "test": "./SplitSet16/test",
    "train": "./SplitSet16/train",
    "valid": "./SplitSet16/validation"
}

#D:\School\Graduate\Project\ImageExtractor\images

# Executive decistionto use street jpg and turn to 3 channel at the end
input = {
    "aerial": "./images-16/aerial",
    "street": "./images-16/streetjpg"
}

size = {
    "test": 500,
    "train": 2500,
    "valid": 500
}

def subtract(mainlist, actlist):
    return list(set(mainlist)- set(actlist))

mainset = listdir(input["aerial"])

testset = random.sample(mainset, k= size["test"])
mainset = subtract(mainset, testset)

validset = random.sample(mainset, k= size["valid"])
mainset = subtract(mainset, validset)

print(len(mainset))

def getfile(outPath,filesName):
    for file in filesName:
        saveName = file.split("/")[-1].split('.')[0]

        aerialInput = input["aerial"] + "/" + file
        streetInput = input["street"] + "/" + saveName + ".jpg"

        saveName = file.split("/")[-1].split('.')[0]

        outAerial = outPath + "/aerial/" + saveName + ".jpeg"
        outStreet = outPath + "/street/" + saveName + ".jpeg"

        shutil.copy2(aerialInput, outAerial)
        shutil.copy2(streetInput, outStreet)

getfile(target["train"], mainset)
getfile(target["test"], testset)
getfile(target["valid"], validset)