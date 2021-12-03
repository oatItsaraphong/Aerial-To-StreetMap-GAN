## ImageBuilder process

This is a python image generator process in which the image cfid is pass in and process.

The main process in which get send between the two system is using a file system where the node server will request the aerial image and store it in a predefine folder. The image generate will get all by the server to process the image base on the id. Then the imagebuilder will pick the file up the file an use it as a seed to start generating. 

### Issue
The main issue is that the generating image is very expensive and take so long that the browser actually excute timeout which lead to a bad leaflet output therefore we increase the speed by cache all the image before hand to reduce the issue. The server can be run in realtime if and only if the hardware is suffice.

### Components:

#### models.image_model_v##.py
This code contain generator and discriminator for each sepecific model. 

#### image_builder.py
This is the main code which generate the image. This will perform image format and clean up before passed into the generator. This code will call image_model** and use their gen and dis.

### image_model_adapter.py
This is an interface in which the model version got selected. 

### imagebuiler_cache.py
A program that will generate all the GAN image from all image the aerial map folder
`python imagebuilder_cache.py -v 15

### imagebuilder_server.py
A zerorpc server which will get all by the node express server to generate an image base on a given id
`python imagebuilder_server.py -p 4015 -v 15

### imagebuilder_base.py
An example code.
