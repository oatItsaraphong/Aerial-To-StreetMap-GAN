from os import listdir
from PIL import Image

PATHPNG = "../cacheimages/aerialimage"
PATHIN = "../cacheimages/aerialimageread"

def getfile(inPath,outPath):
    mainset = listdir(inPath)
    for file in mainset:
        path = inPath + "/"+file
        image1 = Image.open(path)
        image1_size = image1.size
        new_image = Image.new('RGB',(image1_size[0], image1_size[1]))
        new_image.paste(image1,(0,0))
        file = file.split('.')[0]
        file = file + ".jpeg"
        new_image.save(outPath +"/" + file,"JPEG")

getfile(PATHPNG,PATHIN)
