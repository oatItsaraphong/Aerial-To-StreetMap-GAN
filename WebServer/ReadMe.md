## Webserver

This folder contain the backend to cache and generate the map image by either retrive it from third party or generate the map using GAN. The service also responsible for downloading the source map that will be use as a base to ML use to generate a streetmap. 

### Node Server
Index.js is a express server running in node 8.17.0. The service is version sensitive due to the library "zerorpc" using node-gyp which use to communicate with python imagebuilder. This server perform two main task for each call one is to retrive the image from third party and store it in local file as cache. 

#### Sample
localhost:3000/getmapimage?type=street&x=11303&y=26205&z=16

#### ClearCache
localhost:3000/getmapimage?type=street&x=11303&y=26205&z=16


### Python
imagebuilder.py is not a full web server it is a zerorpc server.
