
import os
from image_builder import ImageBuilder
import sys, getopt
from PIL import Image
from os import listdir

PATHPNG = "../cacheimages/aerialimage"
PATHIN = "../cacheimages/aerialimage"
PATHOUT = "../cacheimages/streetgen"

def main(argv):
    version = "15"

    #convert(PATHPNG, PATHIN)

    opts, args = getopt.getopt(argv,"hv:",["version="])
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -v <version>')
            sys.exit()
        elif opt in ("-v", "--version"):
            version = arg
    try:
        print("===Server Generate===========")

        pathOut = PATHOUT  + "v" + str(version)
        builder = ImageBuilder(int(version))

        fileList = os.listdir(PATHIN)
        for name in fileList:
            if not (os.path.exists(pathOut + "/" + name)):
                builder.generate_image(name)
        
    except Exception as e:
        print(str(e))

def convert(inFile, outFile):
    files = listdir(inFile)

    for file in files:
        name = file.split("/")[-1].split('.')[0]
        #print(name)
        im1 = Image.open(inFile + "/" +file)
        im1 = im1.convert('RGB')
        im1.save(outFile +"/"+ name + '.jpg', 'JPEG')


if __name__ == "__main__":
   main(sys.argv[1:])

