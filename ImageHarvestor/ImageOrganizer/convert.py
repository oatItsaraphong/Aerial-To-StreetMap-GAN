from PIL import Image
from os import listdir

inFile = './images-16/streetmap'
outFile = './images-16/streetjpg'

files = listdir(inFile)

for file in files:
    name = file.split("/")[-1].split('.')[0]
    #print(name)
    im1 = Image.open(inFile + "/" +file)
    im1 = im1.convert('RGB')
    im1.save(outFile +"/"+ name + '.jpg')

